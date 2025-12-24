import os
from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    SECRET_KEY: str = "t-nexus-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    DATABASE_URL: str = "sqlite:///./tnexus.db"

    BOT_TOKEN: str =  "8377171332:AAFuNVmTDyqpz1fRDdYU_szfVxs_NwOhe_4"
    RAG_URL: str = "http://localhost:8000/"
    TG_DB_PATH: str = "telegram_bot.db"

settings = Settings()