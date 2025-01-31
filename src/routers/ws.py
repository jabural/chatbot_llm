# app/routers/ws.py
import os
import json
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any
from ..core.audio_utils import (
    decode_base64_to_array,
    save_audio_to_wav,
)
from ..core.llm import get_response_llm, get_transcription_audio_file


router = APIRouter(
        tags=['ws']
)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
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
                user_input = get_transcription_audio_file()
                os.remove("output.wav")

                config = {"configurable": {"thread_id": thread}}
                response: str = get_response_llm(user_input, config)

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
