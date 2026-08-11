"""Macht das Paket `drift_glm` fuer pytest importierbar, ohne es zu installieren.

Ohne diese Zeile muesste das Repo per `pip install -e .` eingerichtet werden. Fuer eine
Abschlussarbeit, die aus dem Ordner heraus laufen soll, ist der sys.path-Eintrag der
kleinere Eingriff -- und `pytest tests` funktioniert damit aus jedem Arbeitsverzeichnis.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
