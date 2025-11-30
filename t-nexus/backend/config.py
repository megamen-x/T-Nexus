import os

SECRET_KEY = os.getenv("SECRET_KEY", "t-nexus-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tnexus.db")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

BOT_TOKEN = os.getenv("BOT_TOKEN", "8377171332:AAFuNVmTDyqpz1fRDdYU_szfVxs_NwOhe_4")
RAG_URL = os.getenv("RAG_URL", "http://localhost:8001/")
TRANSCRIPTION_URL = os.getenv("TRANSCRIPTION_URL", "http://localhost:8002/transcribe")
TG_DB_PATH = os.getenv("TG_DB_PATH", "telegram_bot.db")