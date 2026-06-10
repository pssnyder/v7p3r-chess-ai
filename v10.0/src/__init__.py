"""V7P3R Chess AI v10.0 - Neural Network Training Framework.

This package contains modules for:
- Data serialization and streaming (binary formats)
- Feature extraction (HalfKA sparse indexing)
- Neural network architecture (perspective accumulators)
- Training loops and loss functions
- Model quantization and export

Modules:
    binary_format_converter: PGN/JSONL → binary format conversion
    position_filters: Dataset filtering and balancing
    pytorch_dataset: PyTorch IterableDataset for streaming
    halfdka_features: HalfKA feature index calculation
    accumulator_architecture: Perspective accumulator design
    training_loss: Multi-signal loss function (strength + character + WDL)
    train: Training loop orchestration
    quantize_model: INT8 quantization for production

Usage:
    from src.binary_format_converter import BinaryPositionRecord
    from src.halfdka_features import get_active_features
    from src.training_loss import MultiSignalLoss
"""

__version__ = "10.0.0"
__author__ = "V7P3R Chess AI"

# Lazy imports - fill in as modules are implemented
# from .binary_format_converter import BinaryPositionRecord, BinaryFormatConverter
# from .position_filters import PositionAnalyzer, DatasetFilter
# from .pytorch_dataset import ChessBinaryDataset
# from .halfdka_features import get_halfdka_index, get_active_features
# from .accumulator_architecture import PerspectiveAccumulator
# from .training_loss import MultiSignalLoss
# from .train import train_model
# from .quantize_model import quantize_model_int8
