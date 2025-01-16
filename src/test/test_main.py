from fastapi.testclient import TestClient
from fastapi import status
from ..main import app

client = TestClient(app)


def test_return_health_check():
    """
    Check if the endpoint returns the correct json
    """
    response = client.get("/healthy")
    assert response.status_code == status.HTTP_200_OK
    print("I'm here")
    assert response.json() == {'status': 'healthy'}

def test_conversation():
    """
    Check if the chatbot returns an answer to the prompt
    """
    request_data={
        'prompt': 'Hello, my name is Jim.',
        'thread': 'abc123'
    }
    response = client.post('/chatbot', json=request_data)
    # Parse the response body as JSON
    response_data = response.json()
    # Extract the value of the 'response' key
    chatbot_response = response_data.get("response")
    assert isinstance(chatbot_response, str)
