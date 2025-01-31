from ..main import app
import base64
import json
import numpy as np
from unittest.mock import patch
from .utils import override_get_db, client
from ..routers.chatbot import get_db

app.dependency_overrides[get_db] = override_get_db


@patch("src.core.llm.get_response_llm")
def test_conversation(mock_llm):
    mock_llm.return_value = "Hi, how are you doing?"
    """
    Check if the chatbot returns an answer to the prompt
    """
    request_data = {
        'prompt': 'Hello, my name is Jim.',
        'thread': 'abc123'
    }
    response = client.post('/chatbot', json=request_data)
    # Parse the response body as JSON
    response_data = response.json()
    # Extract the value of the 'response' key
    chatbot_response = response_data.get("response")
    assert isinstance(chatbot_response, str)


@patch("src.core.llm.get_transcription_audio_file")
@patch("src.core.llm.get_response_llm")
def test_websocket_interaction(mock_llm, mock_stt):
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
        print(response)
        response_data = json.loads(response)

        assert response_data["status"] == "success"
        assert "text_generated" in response_data
        assert isinstance(response_data["text_generated"], str)
