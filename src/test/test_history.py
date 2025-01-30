from fastapi.testclient import TestClient
from fastapi import status
from ..main import app
import numpy as np
from unittest.mock import patch
from ..routers.history import get_db
from .utils import *
from ..routers.history import ThreadResponse

app.dependency_overrides[get_db] = override_get_db

def test_get_by_thread(test_sql):
    app.dependency_overrides[get_db] = override_get_db
    response = client.get("/history/?thread_id=thread_test")
    print(response.status_code)
    print(response.json())
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["thread_id"] == "thread_test"
