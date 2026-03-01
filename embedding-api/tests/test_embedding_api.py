import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Create test client."""
    from main import app
    return TestClient(app)


@patch("main.httpx.AsyncClient")
def test_health(mock_httpx, client):
    """Test health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@patch("main.httpx.AsyncClient")
def test_embed_valid_input(mock_httpx, client):
    """Test embed endpoint with valid input."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]]
    }
    mock_response.raise_for_status = MagicMock()
    
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_httpx.return_value.__aenter__.return_value = mock_client
    
    response = client.post("/embed", json={
        "texts": ["hello world"]
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert data["embeddings"] == [[0.1, 0.2, 0.3]]


@patch("main.httpx.AsyncClient")
def test_embed_batch_input(mock_httpx, client):
    """Test embed endpoint with batch input."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    }
    mock_response.raise_for_status = MagicMock()
    
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_httpx.return_value.__aenter__.return_value = mock_client
    
    response = client.post("/embed", json={
        "texts": ["hello", "world"]
    })
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["embeddings"]) == 2


@patch("main.httpx.AsyncClient")
def test_embed_empty_texts(mock_httpx, client):
    """Test embed endpoint with empty texts returns validation error."""
    response = client.post("/embed", json={
        "texts": []
    })
    
    assert response.status_code == 422


@patch("main.httpx.AsyncClient")
def test_embed_with_task_type(mock_httpx, client):
    """Test embed endpoint with task_type parameter."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [[0.5, 0.6, 0.7]]
    }
    mock_response.raise_for_status = MagicMock()
    
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_httpx.return_value.__aenter__.return_value = mock_client
    
    response = client.post("/embed", json={
        "texts": ["hello world"],
        "task_type": "query"
    })
    
    assert response.status_code == 200
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args.kwargs["json"]["task_type"] == "query"


@patch("main.httpx.AsyncClient")
def test_embed_invalid_task_type(mock_httpx, client):
    """Test embed endpoint with invalid task_type returns error."""
    response = client.post("/embed", json={
        "texts": ["hello world"],
        "task_type": "invalid"
    })
    
    assert response.status_code == 422
