import logging
import os
from collections.abc import Generator
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("vehicle_ai.database")

load_dotenv(PROJECT_ROOT / ".env", override=False)


class Base(DeclarativeBase):
    pass


def _database_url() -> str:
    value = os.environ.get("DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError(
            "DATABASE_URL is required. Set it in the environment or in the project .env file."
        )
    try:
        parsed = make_url(value)
    except Exception:
        raise RuntimeError(
            "DATABASE_URL is invalid. Use the documented PostgreSQL psycopg URL format."
        ) from None
    if (
        parsed.drivername != "postgresql+psycopg"
        or not parsed.host
        or not parsed.username
        or not parsed.password
        or not parsed.database
    ):
        raise RuntimeError(
            "DATABASE_URL must use postgresql+psycopg and specify a host, username, password, and database."
        )
    return value


try:
    engine = create_engine(
        _database_url(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=1800,
        connect_args={"connect_timeout": 5},
    )
except (ImportError, ModuleNotFoundError):
    raise RuntimeError(
        "The PostgreSQL psycopg driver is unavailable. Install project dependencies."
    ) from None

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def verify_database_connection() -> None:
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
    except SQLAlchemyError as error:
        logger.error(
            "PostgreSQL connection verification failed (%s).",
            type(error).__name__,
        )
        raise RuntimeError(
            "Could not connect to PostgreSQL. Verify the local service and DATABASE_URL."
        ) from None
