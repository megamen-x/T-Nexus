import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.t_nexus.backend.telegram.bot import create_bot
from src.t_nexus.backend.config import settings

def main():
    bot_token = settings.BOT_TOKEN
    rag_url = settings.RAG_URL
    transcription_url = settings.TRANSCRIPTION_URL
    db_path = settings.TG_DB_PATH

    if not bot_token:
        print("Error: BOT_TOKEN environment variable is not set!")
        print("Please set it before running the bot:")
        print("  export BOT_TOKEN='your-telegram-bot-token'")
        sys.exit(1)

    print("Starting T-Nexus Telegram Bot...")
    print(f"RAG URL: {rag_url}")
    print(f"Transcription URL: {transcription_url}")
    print(f"Database path: {db_path}")

    bot = create_bot(
        bot_token=bot_token,
        db_path=db_path,
        rag_url=rag_url,
        transcription_url=transcription_url
    )

    asyncio.run(bot.start_polling())


if __name__ == "__main__":
    main()