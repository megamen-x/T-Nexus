import os
import sys
import asyncio


start_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, start_path)

from src.t_nexus.backend.telegram.bot import create_bot
from src.t_nexus.backend.config import settings

def main():
    print(start_path)
    print(sys.path)
    bot_token = settings.BOT_TOKEN
    rag_url = settings.RAG_URL
    db_path = settings.TG_DB_PATH

    if not bot_token:
        print("Error: BOT_TOKEN environment variable is not set!")
        print("Please set it before running the bot:")
        print("  export BOT_TOKEN='your-telegram-bot-token'")
        sys.exit(1)

    print("Starting T-Nexus Telegram Bot...")
    print(f"RAG URL: {rag_url}")
    print(f"Database path: {db_path}")

    bot = create_bot(
        bot_token=bot_token,
        db_path=db_path,
        rag_url=rag_url,
    )

    asyncio.run(bot.start_polling())


if __name__ == "__main__":
    main()