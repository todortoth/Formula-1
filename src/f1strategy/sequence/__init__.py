"""Continuous data: dense time sampling of the race graph, compact sequence containers, Parquet datasets."""

from f1strategy.sequence.builder import SequenceBuilder
from f1strategy.sequence.dataset import ContinuityReport, RaceSequence
from f1strategy.sequence.parquet import ParquetSequence, ParquetSequenceWriter, write_dataset
from f1strategy.sequence.sampling import resolve_step, time_grid

__all__ = [
    "ContinuityReport",
    "ParquetSequence",
    "ParquetSequenceWriter",
    "RaceSequence",
    "SequenceBuilder",
    "resolve_step",
    "time_grid",
    "write_dataset",
]
