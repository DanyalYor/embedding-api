import pytest
from fastapi.testclient import TestClient
from embedding_api.main import app

@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    """Test health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200

def test_embed_valid_input(client, mock_embedding_service):
    """Test embed endpoint with valid input."""
    mock_embedding_service.return_value.embed.return_value = [[0.1, 0.2, 0.3]]
    
    response = client.post("/embed", json={"texts": ["hello world"]})

    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert data["embeddings"] == [[0.1, 0.2, 0.3]]
    assert data["model"] == "intfloat/multilingual-e5-large"


def test_embed_batch_input(client, mock_embedding_service):
    """Test embed endpoint with batch input."""
    mock_embedding_service.return_value.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    response = client.post("/embed", json={"texts": ["hello", "world"]})

    assert response.status_code == 200
    data = response.json()
    assert len(data["embeddings"]) == 2


def test_embed_empty_texts(client):
    """Test embed endpoint with empty texts returns validation error."""
    response = client.post("/embed", json={"texts": []})
    assert response.status_code == 422


def test_embed_with_task_type(client, mock_embedding_service):
    """Test embed endpoint with task_type parameter."""
    response = client.post("/embed", json={"texts": ["hello world"], "task_type": "query"})

    assert response.status_code == 200
    mock_embedding_service.return_value.embed.assert_called_once_with(["hello world"], "query")


def test_embed_invalid_task_type(client):
    """Test embed endpoint with invalid task_type returns error."""
    response = client.post("/embed", json={"texts": ["hello world"], "task_type": "invalid"})
    assert response.status_code == 422


def test_embed_service_not_ready(client, mock_embedding_service):
    """Test embed endpoint when service is not loaded."""
    mock_embedding_service.return_value = None
    app.state.service = None

    response = client.post("/embed", json={"texts": ["hello world"]})

    assert response.status_code == 503
    assert "not ready" in response.json()["detail"].lower()
