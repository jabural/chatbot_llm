import os
import json
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
from ..core.audio_utils import (
    decode_base64_to_array,
    save_audio_to_wav,
)
from ..core.llm import get_response_llm, get_transcription_audio_file, add_message_sql
from ..database import SessionLocal
from typing import Annotated
from ..models import Thread


router = APIRouter(
        tags=['ws']
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: db_dependency) -> None:
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            try:
                message: str = await websocket.receive_text()
                data: Dict[str, Any] = json.loads(message)

                thread: str = data.get("user_id", "abc123")
                base64_encoded_array: str = data.get("audio_data", "")
                samplerate: int = data.get("samplerate", 16000)
                channels: int = data.get("channels", 1)

                decoded_array: np.ndarray = decode_base64_to_array(base64_encoded_array)
                if channels > 1:
                    decoded_array = decoded_array.reshape(-1, channels)

                save_audio_to_wav(decoded_array, samplerate)
                user_input = await get_transcription_audio_file()
                os.remove("output.wav")
                add_message_sql(db, thread, user_input, "user")

                config = {"configurable": {"thread_id": thread}}

                messages = []
                db_thread = db.query(Thread).filter_by(id=thread).first()
                if db_thread:
                    for msg in db_thread.messages:
                        messages.append((msg.role, msg.content))
                    response = await get_response_llm(messages, config)
                    add_message_sql(db, thread, response, "assistant")

                if not response:
                    raise ValueError("No response content generated.")

                await websocket.send_text(json.dumps({
                    "status": "success",
                    "text_generated": response
                }))

            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                await websocket.send_text(json.dumps({
                    "status": "error",
                    "message": "Internal server error."
                }))
    finally:
        print("WebSocket connection closed")
