"""Macht die Analyse-Module (pipeline, sweep, ...) fuer pytest importierbar."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
