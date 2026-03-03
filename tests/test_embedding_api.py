import pytest
from fastapi.testclient import TestClient

from embedding_api.main import app


@pytest.fixture
def client(mock_embedding_service):
    with TestClient(app) as test_client:
        yield test_client

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ready"] is True

def test_embed_v1_valid_input(client):
    response = client.post("/api/v1/embed", json={"texts": ["hello world"]})
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert data["model"] == "intfloat/multilingual-e5-large"

def test_embed_v1_batch_input(client, mock_embedding_service):
    mock_embedding_service.return_value.embed.return_value = [
        [0.1] * 1024,  
        [0.2] * 1024, 
    ]

    response = client.post("/api/v1/embed", json={"texts": ["hello", "world"]})
    assert response.status_code == 200
    data = response.json()
    assert len(data["embeddings"]) == 2

def test_embed_v1_empty_texts(client):
    response = client.post("/api/v1/embed", json={"texts": []})
    assert response.status_code == 422

def test_embed_v1_with_task_type(client):
    response = client.post("/api/v1/embed", json={"texts": ["hello world"], "task_type": "query"})
    assert response.status_code == 200

def test_embed_v1_invalid_task_type(client):
    response = client.post("/api/v1/embed", json={"texts": ["hello world"], "task_type": "invalid"})
    assert response.status_code == 422

def test_embed_v1_service_not_ready(client, mock_embedding_service):
    mock_embedding_service.return_value = None
    app.state.service = None
    response = client.post("/api/v1/embed", json={"texts": ["hello world"]})
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"].lower()

def test_metrics_endpoint(client):
    client.post("/api/v1/embed", json={"texts": ["hello"]})
    
    response = client.get("/metrics")
    assert response.status_code == 200
    
    assert "http_requests_total" in response.text
    assert 'handler="/api/v1/embed"' in response.text
    
    assert "embed_batch_size" in response.text
    assert "embed_inference_duration_seconds" in response.text
    assert "model_loaded" in response.text
