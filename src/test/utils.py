from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from ..database import Base
from ..main import app
from fastapi.testclient import TestClient
import pytest
from ..models import Thread, Message

DATABASE_URL = "sqlite:///./test_conversations.db"

# 1. Create the engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)
# 2. Create a configured "SessionLocal" class
TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base.metadata.create_all(bind = engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

client = TestClient(app)

@pytest.fixture
def test_sql():

    thread = Thread(
        id = "thread_test",
        title = "Testing thread"
    )

    message = Message(
        thread_id = "thread_test",
        role = "user",
        content = "Hello, how are you doing"
    )

    message2 = Message(
        thread_id = "thread_test",
        role = "assistant",
        content = "Fine, thanks for asking"
    )

    db = TestingSessionLocal()
    db.add(thread)
    db.commit()
    db.add(message)
    db.commit()
    db.add(message2)
    db.commit()
    yield thread, message, message2
    with engine.connect() as connection:
        connection.execute(text("DELETE FROM threads;"))
        connection.execute(text("DELETE FROM messages;"))
        connection.commit()





