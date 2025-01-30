from fastapi.testclient import TestClient
from fastapi import status
from ..main import app
import numpy as np
from unittest.mock import patch
from .utils import *


def test_return_health_check():
    """
    Check if the endpoint returns the correct json
    """
    response = client.get("/healthy")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {'status': 'healthy'}