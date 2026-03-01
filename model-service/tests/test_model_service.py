import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create test client with mocked service."""
    import main
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3]], 
        "attention_mask": [[1, 1, 1]]
    }
    
    mock_model = MagicMock()
    mock_output = MagicMock()
    mock_output.last_hidden_state = MagicMock()
    mock_output.last_hidden_state.mean.return_value = MagicMock()
    mock_output.last_hidden_state.mean.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
    mock_model.return_value = mock_output
    
    main.service = {
        "model": mock_model,
        "tokenizer": mock_tokenizer
    }
    
    with patch("main.normalize_embeddings", side_effect=lambda x: x):
        from main import app
        yield TestClient(app)


def test_health(client):
    """Test health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ready"] is True


def test_embed_valid_input(client):
    """Test embed endpoint with valid input."""
    response = client.post("/embed", json={
        "texts": ["hello world"]
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)
    assert len(data["embeddings"]) == 1


def test_embed_empty_texts(client):
    """Test embed endpoint with empty texts returns empty embeddings."""
    response = client.post("/embed", json={
        "texts": []
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["embeddings"] == []


def test_embed_multiple_texts(client):
    """Test embed endpoint with multiple texts."""
    response = client.post("/embed", json={
        "texts": ["hello", "world", "test"]
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)


def test_embed_with_query_task_type(client):
    """Test embed endpoint accepts query task_type."""
    response = client.post("/embed", json={
        "texts": ["hello world"],
        "task_type": "query"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["embeddings"]) == 1


def test_embed_with_passage_task_type(client):
    """Test embed endpoint accepts passage task_type."""
    response = client.post("/embed", json={
        "texts": ["hello world"],
        "task_type": "passage"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["embeddings"]) == 1


def test_embed_invalid_task_type(client):
    """Test embed endpoint returns error for invalid task_type."""
    response = client.post("/embed", json={
        "texts": ["hello world"],
        "task_type": "invalid"
    })
    
    assert response.status_code == 422


def test_embed_default_task_type(client):
    """Test embed endpoint accepts None task_type."""
    response = client.post("/embed", json={
        "texts": ["hello world"]
    })
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["embeddings"]) == 1
