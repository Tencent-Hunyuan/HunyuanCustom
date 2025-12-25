"""Example unit test file to demonstrate testing patterns."""

import pytest
import numpy as np


class TestExampleUnit:
    """Example unit test class."""
    
    @pytest.mark.unit
    def test_example_passes(self):
        """Example test that passes."""
        assert 1 + 1 == 2
    
    @pytest.mark.unit
    def test_using_fixtures(self, mock_config, test_helpers):
        """Example test using fixtures."""
        # Using mock_config fixture
        assert mock_config["model"]["device"] == "cpu"
        
        # Using test_helpers
        array = np.ones((3, 3))
        test_helpers.assert_shape(array, (3, 3))
    
    @pytest.mark.unit
    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8),
    ])
    def test_parametrized(self, input_val, expected):
        """Example parametrized test."""
        assert input_val * 2 == expected