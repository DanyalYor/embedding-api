from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from embedding_api.main import app


@pytest.fixture(autouse=True)
def mock_embedding_service():
    """Mock EmbeddingService for all tests to avoid loading the real model."""
    with patch("embedding_api.main.EmbeddingService") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.model_name = "intfloat/multilingual-e5-large"
        mock_instance.embed.return_value = [[0.1] * 1024]
        mock_cls.return_value = mock_instance
        
        app.state.service = mock_instance
        yield mock_cls

@pytest.fixture
def client():
    """Create test client with mocked service."""
    with TestClient(app) as test_client:
        yield test_client
