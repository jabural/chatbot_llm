from ..main import app
from unittest.mock import patch
from .utils import override_get_db, client, test_sql  # noqa: F401
from ..routers.chatbot import get_db

app.dependency_overrides[get_db] = override_get_db


@patch("src.routers.chatbot.get_response_llm")
def test_conversation(mock_llm, test_sql):  # noqa: F811
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


@patch("src.routers.chatbot.get_response_llm")
def test_conversation_value(mock_llm, test_sql):  # noqa: F811
    mock_llm.return_value = "Hi, how are you doing?"
    """
    Check if the chatbot returns an answer to the prompt
    """
    mock_llm.return_value = "Hi, how are you doing?"
    request_data = {
        'prompt': 1,
        'thread': 'abc123'
    }
    response = client.post('/chatbot', json=request_data)

    assert response.status_code == 422


@patch("src.routers.chatbot.get_response_llm")
def test_conversation_empty_prompt(mock_llm, test_sql):  # noqa: F811
    """
    Ensure sending an empty prompt is handled properly.
    """
    mock_llm.return_value = "Hi, how are you doing?"
    request_data = {
        'prompt': '',
        'thread': 'abc123'
    }
    response = client.post('/chatbot', json=request_data)
    assert response.status_code == 422


@patch("src.routers.chatbot.get_response_llm")
def test_conversation_no_thread(mock_llm, test_sql):  # noqa: F811
    """
    Ensure chatbot handles missing thread ID.
    """
    mock_llm.return_value = "Hi, how are you doing?"
    request_data = {
        'prompt': 'Hello!'
    }
    response = client.post('/chatbot', json=request_data)
    assert response.status_code == 200


@patch("src.routers.chatbot.get_response_llm", side_effect=ValueError("Test LLM failure"))
def test_conversation_value_error(mock_llm, test_sql):  # noqa: F811
    """
    Ensure chatbot handles ValueErrors from LLM.
    """
    mock_llm.return_value = "Hi, how are you doing?"
    request_data = {
        'prompt': 'Hello!',
        'thread': 'abc123'
    }
    response = client.post('/chatbot', json=request_data)

    assert response.status_code == 400
    response_data = response.json()
    assert "Value error" in response_data["detail"]


@patch("src.routers.chatbot.get_response_llm", side_effect=Exception("Unexpected LLM failure"))
def test_conversation_generic_exception(mock_llm, test_sql):  # noqa: F811
    """
    Ensure chatbot handles unexpected errors gracefully.
    """
    request_data = {
        'prompt': 'Hello!',
        'thread': 'abc123'
    }
    response = client.post('/chatbot', json=request_data)

    assert response.status_code == 500
    response_data = response.json()
    assert "Unexpected error occurred" in response_data["detail"]
