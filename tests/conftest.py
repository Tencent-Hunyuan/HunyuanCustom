"""Shared pytest fixtures and configuration for all tests."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, MagicMock

import pytest
import numpy as np


# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "model": {
            "name": "test_model",
            "version": "1.0.0",
            "device": "cpu",
            "dtype": "float32",
        },
        "paths": {
            "models": "/tmp/models",
            "data": "/tmp/data",
            "output": "/tmp/output",
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "epochs": 10,
        },
        "inference": {
            "batch_size": 1,
            "temperature": 1.0,
            "num_samples": 1,
        }
    }


@pytest.fixture
def sample_image_array() -> np.ndarray:
    """Create a sample image array for testing."""
    # Create a simple 224x224x3 RGB image
    return np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_video_frames() -> np.ndarray:
    """Create sample video frames for testing."""
    # Create 10 frames of 128x128x3 RGB video
    return np.random.randint(0, 255, size=(10, 128, 128, 3), dtype=np.uint8)


@pytest.fixture
def sample_audio_array() -> np.ndarray:
    """Create a sample audio array for testing."""
    # Create a 1-second audio signal at 16kHz sample rate
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate a simple sine wave at 440Hz (A4 note)
    frequency = 440.0
    audio = np.sin(2 * np.pi * frequency * t)
    return audio


@pytest.fixture
def mock_model():
    """Create a mock model object."""
    model = MagicMock()
    model.forward = MagicMock(return_value=torch.zeros(1, 512))
    model.eval = MagicMock()
    model.train = MagicMock()
    model.parameters = MagicMock(return_value=[])
    return model


@pytest.fixture
def mock_diffusion_pipeline():
    """Create a mock diffusion pipeline."""
    pipeline = MagicMock()
    pipeline.device = "cpu"
    pipeline.dtype = "float32"
    pipeline.__call__ = MagicMock(return_value={"images": [np.zeros((512, 512, 3))]})
    return pipeline


@pytest.fixture
def mock_text_encoder():
    """Create a mock text encoder."""
    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=np.random.randn(1, 77, 768))
    return encoder


@pytest.fixture
def mock_vae():
    """Create a mock VAE."""
    vae = MagicMock()
    vae.encode = MagicMock(return_value=MagicMock(latent_dist=MagicMock(sample=MagicMock(return_value=np.zeros((1, 4, 64, 64))))))
    vae.decode = MagicMock(return_value=MagicMock(sample=np.zeros((1, 3, 512, 512))))
    return vae


@pytest.fixture
def test_data_dir(temp_dir: Path) -> Path:
    """Create a test data directory with sample files."""
    data_dir = temp_dir / "test_data"
    data_dir.mkdir(exist_ok=True)
    
    # Create some sample files
    (data_dir / "sample.txt").write_text("test content")
    (data_dir / "config.json").write_text('{"key": "value"}')
    
    return data_dir


@pytest.fixture
def env_vars(monkeypatch):
    """Fixture to set and clean up environment variables."""
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))
    return _set_env


@pytest.fixture(autouse=True)
def cleanup_test_files(request):
    """Automatically clean up any test files created during tests."""
    yield
    # Cleanup code here if needed
    pass


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture and assert on log messages."""
    with caplog.at_level("DEBUG"):
        yield caplog


# Markers for conditional test execution
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_internet: mark test as requiring internet connection"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring pre-trained model"
    )


# Skip tests that require GPU if not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests based on markers."""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        if "requires_gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)


# Test utilities
class TestHelpers:
    """Collection of helper methods for tests."""
    
    @staticmethod
    def assert_shape(array: np.ndarray, expected_shape: tuple):
        """Assert that an array has the expected shape."""
        assert array.shape == expected_shape, f"Expected shape {expected_shape}, got {array.shape}"
    
    @staticmethod
    def assert_dtype(array: np.ndarray, expected_dtype):
        """Assert that an array has the expected dtype."""
        assert array.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {array.dtype}"
    
    @staticmethod
    def assert_range(array: np.ndarray, min_val: float, max_val: float):
        """Assert that all values in an array are within a range."""
        assert np.all(array >= min_val) and np.all(array <= max_val), \
            f"Array values should be in range [{min_val}, {max_val}]"


@pytest.fixture
def test_helpers():
    """Provide test helper methods."""
    return TestHelpers()


# Import torch only if available (for the mock_model fixture)
try:
    import torch
except ImportError:
    # Create a mock torch module for tests that don't actually need PyTorch
    torch = MagicMock()
    torch.zeros = lambda *args, **kwargs: np.zeros(args)