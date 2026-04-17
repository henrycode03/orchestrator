"""Database initialization"""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.models import Base

# Create database engine with optimized pool settings
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=5,  # Keep 5 connections in pool
    max_overflow=10,  # Allow up to 10 additional connections
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_pre_ping=True,  # Verify connection before use
    connect_args=(
        {"check_same_thread": False, "timeout": 30}
        if "sqlite" in settings.DATABASE_URL
        else {}
    ),
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    _ensure_schema_updates()


def _ensure_schema_updates():
    """Apply lightweight schema updates for installs without migrations."""
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    if "tasks" in table_names:
        existing_columns = {column["name"] for column in inspector.get_columns("tasks")}
        statements = []

        if "plan_id" not in existing_columns:
            statements.append("ALTER TABLE tasks ADD COLUMN plan_id INTEGER")
        if "estimated_effort" not in existing_columns:
            statements.append(
                "ALTER TABLE tasks ADD COLUMN estimated_effort VARCHAR(50)"
            )
        if "plan_position" not in existing_columns:
            statements.append("ALTER TABLE tasks ADD COLUMN plan_position INTEGER")
        if "execution_profile" not in existing_columns:
            statements.append(
                "ALTER TABLE tasks ADD COLUMN execution_profile VARCHAR(30) DEFAULT 'full_lifecycle'"
            )

        if statements:
            with engine.begin() as connection:
                for statement in statements:
                    connection.execute(text(statement))

    if "sessions" in table_names:
        existing_columns = {
            column["name"] for column in inspector.get_columns("sessions")
        }
        statements = []

        if "execution_mode" not in existing_columns:
            statements.append(
                "ALTER TABLE sessions ADD COLUMN execution_mode VARCHAR(20) DEFAULT 'automatic'"
            )
        if "default_execution_profile" not in existing_columns:
            statements.append(
                "ALTER TABLE sessions ADD COLUMN default_execution_profile VARCHAR(30) DEFAULT 'full_lifecycle'"
            )
        if "last_alert_level" not in existing_columns:
            statements.append(
                "ALTER TABLE sessions ADD COLUMN last_alert_level VARCHAR(20)"
            )
        if "last_alert_message" not in existing_columns:
            statements.append("ALTER TABLE sessions ADD COLUMN last_alert_message TEXT")
        if "last_alert_at" not in existing_columns:
            statements.append("ALTER TABLE sessions ADD COLUMN last_alert_at DATETIME")

        if statements:
            with engine.begin() as connection:
                for statement in statements:
                    connection.execute(text(statement))


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """Get database session for background tasks (Celery workers)"""
    return SessionLocal()


# ============= TEST DATABASE HELPERS =============


def create_test_database(test_engine):
    """Create test database tables"""
    Base.metadata.create_all(bind=test_engine)


def cleanup_test_database(test_engine):
    """Drop all test database tables"""
    Base.metadata.drop_all(bind=test_engine)
