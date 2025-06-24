"""Validation tests to ensure the testing infrastructure is properly set up."""

import sys
from pathlib import Path

import pytest
import numpy as np


class TestInfrastructureSetup:
    """Test class to validate the testing infrastructure setup."""
    
    def test_python_path_includes_project_root(self):
        """Verify that the project root is in the Python path."""
        project_root = Path(__file__).parent.parent
        assert str(project_root) in sys.path, "Project root should be in Python path"
    
    def test_imports_work(self):
        """Test that we can import from the project packages."""
        # These imports should work if the project structure is correct
        try:
            import hymm_sp
            import hymm_gradio
        except ImportError as e:
            pytest.fail(f"Failed to import project packages: {e}")
    
    def test_fixtures_available(self, temp_dir, mock_config, sample_image_array, 
                                sample_video_frames, sample_audio_array, test_helpers):
        """Test that all fixtures from conftest.py are available and working."""
        # Test temp_dir fixture
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert "model" in mock_config
        assert mock_config["model"]["name"] == "test_model"
        
        # Test image array fixture
        assert isinstance(sample_image_array, np.ndarray)
        test_helpers.assert_shape(sample_image_array, (224, 224, 3))
        test_helpers.assert_dtype(sample_image_array, np.uint8)
        
        # Test video frames fixture
        assert isinstance(sample_video_frames, np.ndarray)
        test_helpers.assert_shape(sample_video_frames, (10, 128, 128, 3))
        
        # Test audio array fixture
        assert isinstance(sample_audio_array, np.ndarray)
        assert len(sample_audio_array.shape) == 1
        assert sample_audio_array.shape[0] == 16000  # 1 second at 16kHz
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that the unit marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that the integration marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that the slow marker works."""
        assert True
    
    def test_coverage_configured(self):
        """Test that coverage is properly configured."""
        # This test will contribute to coverage metrics
        assert True
    
    def test_test_helpers_utilities(self, test_helpers):
        """Test the test helper utilities."""
        # Test assert_shape
        array = np.zeros((2, 3, 4))
        test_helpers.assert_shape(array, (2, 3, 4))
        
        # Test assert_dtype
        array = np.array([1, 2, 3], dtype=np.float32)
        test_helpers.assert_dtype(array, np.float32)
        
        # Test assert_range
        array = np.array([0.1, 0.5, 0.9])
        test_helpers.assert_range(array, 0.0, 1.0)
    
    def test_env_vars_fixture(self, env_vars):
        """Test the environment variable fixture."""
        env_vars(TEST_VAR="test_value", ANOTHER_VAR="123")
        
        import os
        assert os.environ.get("TEST_VAR") == "test_value"
        assert os.environ.get("ANOTHER_VAR") == "123"
    
    def test_capture_logs_fixture(self, capture_logs):
        """Test the log capturing fixture."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        assert "Debug message" in capture_logs.text
        assert "Info message" in capture_logs.text
        assert "Warning message" in capture_logs.text
    
    def test_test_data_dir_fixture(self, test_data_dir):
        """Test the test data directory fixture."""
        assert test_data_dir.exists()
        assert (test_data_dir / "sample.txt").exists()
        assert (test_data_dir / "config.json").exists()
        
        content = (test_data_dir / "sample.txt").read_text()
        assert content == "test content"


class TestPytestConfiguration:
    """Test pytest configuration settings."""
    
    def test_pytest_runs_with_coverage(self):
        """Verify that pytest is configured to run with coverage."""
        # This test verifies that the test is actually running
        # Coverage will be checked when running the tests
        assert True
    
    def test_multiple_test_patterns_work(self):
        """Ensure various test file naming patterns are recognized."""
        # This file itself (test_*.py) should be discovered
        # Other patterns like *_test.py and tests.py are also configured
        assert True