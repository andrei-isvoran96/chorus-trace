"""Local singing transcription from short PCM WAV clips."""

from .audio import FeatureConfig
from .chorus import align_posteriors, evaluate_choruses, pool_posteriors
from .data import Example, read_manifest
from .evaluation import evaluate
from .prepare import prepare_benchmark
from .recognizer import Recognizer, TrainingConfig, train

__all__ = [
    "Example", "FeatureConfig", "Recognizer", "TrainingConfig", "align_posteriors",
    "evaluate", "evaluate_choruses", "pool_posteriors", "prepare_benchmark", "read_manifest", "train",
]
