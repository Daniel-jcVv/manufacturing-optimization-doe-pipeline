import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Centralized project settings loaded from .env file"""
    POSTGRES_HOST = os.getenv("DB_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "doe_optimization")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    DATA_DIR = os.getenv("DATA_DIR", "./data")

    @classmethod
    def postgre_uri(cls) -> str:
        return (
            f"postgresql+psycopg2://{cls.POSTGRES_USER}:"
            f"{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:"
            f"{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
        )

