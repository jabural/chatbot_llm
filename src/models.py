from .database import Base
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship, sessionmaker, Session
from datetime import datetime, UTC

class Thread(Base):
    __tablename__ = "threads"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now(UTC))

    # One-to-many: A thread can contain multiple messages
    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Thread(id={self.id}, title={self.title})>"

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(Integer, ForeignKey("threads.id"), nullable=False)
    role = Column(String, nullable=False)   # e.g., "user", "assistant", "system"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now(UTC))

    # Relationship back to Thread
    thread = relationship("Thread", back_populates="messages")

    def __repr__(self):
        return f"<Message(thread_id={self.thread_id}, role={self.role})>"
