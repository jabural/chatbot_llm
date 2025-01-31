from fastapi import status
from .utils import client


def test_return_health_check():
    """
    Check if the endpoint returns the correct json
    """
    response = client.get("/healthy")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {'status': 'healthy'}
