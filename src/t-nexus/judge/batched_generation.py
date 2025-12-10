import argparse
import asyncio
import json
import os
from datetime import datetime

import pandas as pd
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Async response generation via OpenAI API')
parser.add_argument('--api_key', type=str, default=os.environ.get("OPENAI_API_KEY"), help='API key')
parser.add_argument('--base_url', type=str, default="https://openrouter.ai/api/v1", help='Base URL for API')
parser.add_argument('--model_name', type=str, required=True, help='Model name for API requests')
parser.add_argument('--n_parallel', type=int, default=4, help='Number of parallel requests')
parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for generation')
parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of tokens to generate')
parser.add_argument('--input_file', type=str, required=True, help='Path to JSONL file with prompts')
parser.add_argument('--system_field', type=str, default=None, help='Field name for system prompt in input file (optional)')
parser.add_argument('--user_field', type=str, required=True, help='Field name for user prompt in input file')
parser.add_argument('--output_folder', type=str, default="data", help='Folder name for saving results')
parser.add_argument('--output_filename', type=str, default=None, help='Output filename (optional, auto-generated if not specified)')
parser.add_argument('--response_format_field', type=str, default=None, help='Field name for response format in input file (optional)')
# parser.add_argument('--log_to_file_or_no', type=bool, default=False, help='')
args = parser.parse_args()

client = AsyncOpenAI(
    api_key=args.api_key,
    base_url=args.base_url
)

semaphore = asyncio.Semaphore(args.n_parallel)
write_lock = asyncio.Lock()

if args.output_filename is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model_name.replace("/", "_")
    args.output_filename = f"{model_short}_{timestamp}.jsonl"

output_path = os.path.join(args.output_folder, args.output_filename)
os.makedirs(args.output_folder, exist_ok=True)

async def write_to_file(record):
    async with write_lock:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def build_messages(row, system_field, user_field):
    messages = []
    if system_field:
        messages.append({"role": "system", "content": row[system_field]})
    messages.append({"role": "user", "content": row[user_field]})
    return messages


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type((
        json.JSONDecodeError,
        APIConnectionError,
        RateLimitError,
        InternalServerError
    ))
)
async def _generate_with_retry(
    client,
    messages: list,
    model_name: str,
    response_format: dict,
    temperature: float,
    max_tokens: int
):
    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

async def generate(row):
    messages = build_messages(row, args.system_field, args.user_field)
    async with semaphore:
        try:
            response_format = {"type": "text"} if args.response_format_field is None else row[args.response_format_field]
            content = await _generate_with_retry(
                client=client,
                messages=messages,
                model_name=args.model_name,
                response_format=response_format,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens
            )

            status = "OK"
        except Exception as e:
            content = str(e)
            status = "ERROR"

        record = {
            **row.to_dict(),
            "model": args.model_name,
            "content": content,
            "status": status
        }

        await write_to_file(record)
        return record


async def main():
    df = pd.read_json(args.input_file, lines=True)
    if args.system_field not in df or args.user_field not in df:
        raise ValueError(f'Файл с запросами должен содержать колонки {args.prompt_field} и {args.id_field}!')

    if args.response_format_field is not None and args.response_format_field not in df:
        raise ValueError()

    tasks = [generate(row) for _, row in df.iterrows()]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await f


if __name__ == '__main__':
    asyncio.run(main())