"""Safe Stage: portable manipulation scenes, prompts, and inspection tools.

The default API uses only Python's standard library. USD and simulator modules
are imported only by the optional inspection or simulation entry points.
"""

from .catalog import CONDITIONS, Dataset, Scene, SafeStageError, find_data_root
from .validation import CheckReport, check_dataset

__all__ = [
    "CONDITIONS", "Dataset", "Scene", "SafeStageError", "find_data_root",
    "CheckReport", "check_dataset",
]
__version__ = "0.2.0"
