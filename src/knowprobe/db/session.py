"""Database session management."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy.orm import Session

from knowprobe.db.models import get_engine, get_session_factory


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Yield a database session with automatic commit/rollback."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class DatabaseSession:
    """Async-friendly database session manager."""

    def __init__(self) -> None:
        self._factory = get_session_factory()
        self._session: Session | None = None

    def __enter__(self) -> Session:
        self._session = self._factory()
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session is not None:
            if exc_type is None:
                self._session.commit()
            else:
                self._session.rollback()
            self._session.close()
            self._session = None
