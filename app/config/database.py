"""Database configuration and setup."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
from typing import Generator
import logging

from .settings import get_config

logger = logging.getLogger(__name__)

# Create declarative base for models
Base = declarative_base()

# Global variables for database components
engine = None
SessionLocal = None


def init_database(config=None) -> None:
    """Initialize database with configuration."""
    global engine, SessionLocal
    
    if config is None:
        config = get_config()
    
    # Create engine based on database URL
    if config.DATABASE_URL.startswith('sqlite'):
        # SQLite specific configuration
        engine = create_engine(
            config.DATABASE_URL,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=config.DEBUG
        )
    else:
        # Other databases (PostgreSQL, MySQL, etc.)
        engine = create_engine(
            config.DATABASE_URL,
            echo=config.DEBUG
        )
    
    # Create session factory
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
    
    logger.info(f"Database initialized with URL: {config.DATABASE_URL}")


def create_tables() -> None:
    """Create all database tables."""
    if engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def get_db_session() -> Generator:
    """Get database session for dependency injection."""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db() -> SessionLocal:
    """Get database session (direct access)."""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    return SessionLocal()


# Backward compatibility with existing code
def get_engine():
    """Get database engine (backward compatibility)."""
    if engine is None:
        init_database()
    return engine