import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backend.telegram.bot import create_bot


def main():
    bot_token = os.getenv("BOT_TOKEN")
    rag_url = os.getenv("RAG_URL", "http://localhost:8001/")
    transcription_url = os.getenv("TRANSCRIPTION_URL", "http://localhost:8002/transcribe")
    db_path = os.getenv("TG_DB_PATH", "telegram_bot.db")

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