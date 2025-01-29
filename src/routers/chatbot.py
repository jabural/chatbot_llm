from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import Thread, Message
from ..core.llm import get_response_llm, add_message_sql

router = APIRouter(
    tags=['chatbot']
)

class Prompt(BaseModel):
    prompt: str = Field(..., title="Prompt", description="The prompt to send")
    thread: str = Field("abc123", title="Thread", description="The thread of the user")

@router.post("/chatbot", status_code=status.HTTP_200_OK)
async def handle_prompt(data: Prompt) -> Dict[str, str]:
    user_input: str = data.prompt
    thread: str = data.thread
    add_message_sql(thread, user_input, "user")  # store user message in DB

    # Build config & load conversation
    config = {"configurable": {"thread_id": thread}}
    session: Session = SessionLocal()

    try:
        messages = []
        db_thread = session.query(Thread).filter_by(id=thread).first()
        if db_thread:
            for msg in db_thread.messages:
                messages.append((msg.role, msg.content))

        response = get_response_llm(messages, config)
        add_message_sql(thread, response, "assistant")
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Value error: {str(ve)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error occurred: {str(e)}"
        )
    finally:
        session.close()

    return {"response": response}
