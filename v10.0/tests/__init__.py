"""Tests for __init__.py (package verification)."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_package_imports():
    """Test that all modules can be imported."""
    # TODO: Add import tests as modules are implemented
    # from src.binary_format_converter import BinaryFormatConverter
    # from src.halfdka_features import HalfKAFeatureGenerator
    # from src.training_loss import MultiSignalLoss
    pass


def test_package_version():
    """Test package version is set."""
    import src
    assert hasattr(src, '__version__')
    assert src.__version__ == "10.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
