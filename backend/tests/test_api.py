import pytest
from fastapi.testclient import TestClient
from main import app
import os
import io

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_invalid_file_type():
    # Send a text file instead of an image
    file_content = b"This is a text file, not an image."
    response = client.post(
        "/predict",
        files={"file": ("test.txt", file_content, "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_predict_missing_file():
    # Send request without file
    response = client.post("/predict")
    assert response.status_code == 422 # Unprocessable Entity (FastAPI default for missing required fields)

# Note: Testing the happy path of /predict would require mocking the PyTorch model 
# or using a small dummy image, which is recommended for future expansions of this test suite.
