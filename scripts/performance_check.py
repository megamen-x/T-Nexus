#!/usr/bin/env python3
"""
Utility script to smoke-test HippoRAG end-to-end and record basic timings.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.t_nexus.ml.hipporag import ChatHistory, HippoRAG
from src.t_nexus.ml.utils.document_converter import collect_texts

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a HippoRAG performance smoke test.")
    parser.add_argument(
        "--config",
        default="src/t_nexus/ml/config/hipporag.yaml",
        help="Path to the HippoRAG YAML config.",
    )
    parser.add_argument(
        "--docs-file",
        action="append",
        default=[],
        help="Path to a file containing either a JSON list of docs or plain text.",
    )
    parser.add_argument("--query", required=True, help="Query used for retrieval/QA.")
    parser.add_argument(
        "--chat-mode",
        choices=["last_message", "full_history"],
        default="last_message",
        help="Conversation reduction strategy to test.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top_k.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of retrieval iterations to run for averaging.",
    )
    parser.add_argument(
        "--qa",
        action="store_true",
        help="If set, run the QA step (HippoRAG.rag) instead of retrieval-only.",
    )

    args = parser.parse_args()
    documents = collect_texts(args.docs_file)
    if not documents:
        raise SystemExit("Provide at least one document via --doc or --docs-file.")

    async def _run_async() -> None:
        hipporag = HippoRAG(args.config)

        index_start = time.perf_counter()
        chunk_ids = await hipporag.index_documents(documents)
        index_duration = time.perf_counter() - index_start
        print(f"Indexed {len(chunk_ids)} chunks in {index_duration:.2f}s")

        retrieval_times = []
        answers = []
        for _ in range(args.iterations):
            history = ChatHistory()
            history.add("user", args.query)
            start = time.perf_counter()
            if args.qa:
                answer, retrieval = await hipporag.rag(history, top_k=args.top_k)
                answers.append(answer)
            else:
                retrieval = await hipporag.retrieve(history, top_k=args.top_k, mode=args.chat_mode)
            retrieval_times.append(time.perf_counter() - start)
            print(retrieval)

        avg_retrieval = sum(retrieval_times) / len(retrieval_times)
        print(f"Ran {args.iterations} retrieval iteration(s); avg time {avg_retrieval:.2f}s")
        if args.qa and answers:
            print("Sample answer:", answers[-1])

        hipporag.close()

    asyncio.run(_run_async())


if __name__ == "__main__":
    main()

# uv run python -m scripts/performance_check --docs-file nano_data.csv --query "Мог бы ты пожалуйста мне рассказать, можно ли посмотреть на лимиты на прием платежей по qr коду посмотреть?"
