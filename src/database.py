from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


DATABASE_URL = "sqlite:///./conversations.db"

# 1. Create the engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)
# 2. Create a configured "SessionLocal" class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)
# 3. Create a Base class for your models
Base = declarative_base()

Base.metadata.create_all(bind = engine)

