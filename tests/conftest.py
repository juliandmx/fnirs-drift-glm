"""Macht `drift_glm` fuer pytest importierbar, ohne `pip install -e .`."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
