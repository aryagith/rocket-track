"""rocket-track: detect, track, and benchmark rockets in launch imagery."""

__version__ = "0.1.0"

from .pipeline import TrackPipeline
from .track_sort import SortTracker

__all__ = ["TrackPipeline", "SortTracker", "__version__"]
