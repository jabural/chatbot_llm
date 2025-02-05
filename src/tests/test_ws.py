from ..main import app
import base64
import json
import numpy as np
from unittest.mock import patch
from .utils import override_get_db, client, test_sql  # noqa: F401
from ..routers.ws import get_db
import pytest


app.dependency_overrides[get_db] = override_get_db

@patch("src.core.llm.get_transcription_audio_file")
@patch("src.routers.chatbot.get_response_llm")
def test_websocket_interaction(mock_llm, mock_stt, test_sql):  # noqa: F811
    mock_stt.return_value = "Hello, my name is Jim."
    mock_llm.return_value = "Hi, how are you doing?"

    test_data = {
        "user_id": "abc123",
        "audio_data": base64.b64encode(np.zeros(16000, dtype=np.int16).tobytes()).decode("utf-8"),
        "samplerate": 16000,
        "channels": 1
    }

    with client.websocket_connect("/ws") as websocket:
        websocket.send_text(json.dumps(test_data))
        response = websocket.receive_text()
        response_data = json.loads(response)

        assert response_data["status"] == "success"
        assert "text_generated" in response_data
        assert isinstance(response_data["text_generated"], str)
