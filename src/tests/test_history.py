from fastapi import status
from ..main import app
from ..routers.history import get_db
from .utils import override_get_db, client, test_sql  # noqa: F401

app.dependency_overrides[get_db] = override_get_db


def test_get_by_thread(test_sql):  # noqa: F811
    app.dependency_overrides[get_db] = override_get_db
    response = client.get("/history/?thread_id=thread_test")
    print(response.status_code)
    print(response.json())
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["thread_id"] == "thread_test"


def test_get_empty_thread(test_sql):  # noqa: F811
    app.dependency_overrides[get_db] = override_get_db
    response = client.get("/history/?thread_id=empty_thread")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "No thread found with ID: empty_thread"
