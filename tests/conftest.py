import time
import pytest
from main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    client = TestClient(app)
    yield client