from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session
from typing import Annotated
from ..database import SessionLocal
from ..models import Thread
from datetime import datetime
from .auth import get_current_user


router = APIRouter(
    prefix='/history',
    tags=['history']
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]


class MessageResponse(BaseModel):
    role: str
    content: str
    created_at: datetime


class ThreadResponse(BaseModel):
    thread_id: str
    messages: List[MessageResponse]


@router.get("/", status_code=status.HTTP_200_OK, response_model=ThreadResponse)
async def get_thread_messages(
    db: db_dependency,
    user: user_dependency,
    thread_id: str = Query(..., description="Thread ID to retrieve messages for")
) -> ThreadResponse:
    """
    Retrieves all messages from a specific thread based on thread_id.
    """
    if user is None or user.get('user_role') != 'admin':
        raise HTTPException(status_code=401, detail='Authentication Failed')

    thread = db.query(Thread).filter_by(id=thread_id).first()

    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No thread found with ID: {thread_id}"
        )

    messages = [
        MessageResponse(role=msg.role, content=msg.content, created_at=msg.created_at)
        for msg in thread.messages
    ]
    return ThreadResponse(thread_id=str(thread.id), messages=messages)
