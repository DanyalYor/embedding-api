import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.embedding_api.main import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


@patch("embedding_api.main.EmbeddingService")
def test_health(mock_service, client):
    """Test health endpoint returns healthy status."""
    # Mock the service instance
    mock_instance = MagicMock()
    mock_service.return_value = mock_instance
    app.state.service = mock_instance

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ready"] is True


@patch("embedding_api.main.EmbeddingService")
def test_embed_valid_input(mock_service, client):
    """Test embed endpoint with valid input."""
    # Mock the service instance
    mock_instance = MagicMock()
    mock_instance.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_instance.model_name = "intfloat/multilingual-e5-large"
    mock_service.return_value = mock_instance
    app.state.service = mock_instance

    response = client.post("/embed", json={"texts": ["hello world"]})

    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert data["embeddings"] == [[0.1, 0.2, 0.3]]
    assert data["model"] == "intfloat/multilingual-e5-large"


@patch("embedding_api.main.EmbeddingService")
def test_embed_batch_input(mock_service, client):
    """Test embed endpoint with batch input."""
    mock_instance = MagicMock()
    mock_instance.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_instance.model_name = "intfloat/multilingual-e5-large"
    mock_service.return_value = mock_instance
    app.state.service = mock_instance

    response = client.post("/embed", json={"texts": ["hello", "world"]})

    assert response.status_code == 200
    data = response.json()
    assert len(data["embeddings"]) == 2


def test_embed_empty_texts(client):
    """Test embed endpoint with empty texts returns validation error."""
    # No mock needed - validation happens before service call
    response = client.post("/embed", json={"texts": []})

    assert response.status_code == 422


@patch("embedding_api.main.EmbeddingService")
def test_embed_with_task_type(mock_service, client):
    """Test embed endpoint with task_type parameter."""
    mock_instance = MagicMock()
    mock_instance.embed.return_value = [[0.5, 0.6, 0.7]]
    mock_instance.model_name = "intfloat/multilingual-e5-large"
    mock_service.return_value = mock_instance
    app.state.service = mock_instance

    response = client.post("/embed", json={"texts": ["hello world"], "task_type": "query"})

    assert response.status_code == 200
    # Verify service was called with correct task_type
    mock_instance.embed.assert_called_once_with(["hello world"], "query")


def test_embed_invalid_task_type(client):
    """Test embed endpoint with invalid task_type returns error."""
    # No mock needed - validation happens before service call
    response = client.post("/embed", json={"texts": ["hello world"], "task_type": "invalid"})

    assert response.status_code == 422


@patch("embedding_api.main.EmbeddingService")
def test_embed_service_not_ready(mock_service, client):
    """Test embed endpoint when service is not loaded."""
    app.state.service = None

    response = client.post("/embed", json={"texts": ["hello world"]})

    assert response.status_code == 503
    assert "not ready" in response.json()["detail"].lower()
