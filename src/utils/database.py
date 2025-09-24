from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import Iterator, Optional
from .config import Settings 


def get_engine(uri: Optional[str] = None) -> Engine:
    """Create SQLAlchemy engine from PostgresSQL."""
    return create_engine(uri or Settings.postgre_uri(), future=True)


def test_connecton() -> bool:
    """Health check for database connection."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True


