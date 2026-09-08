"""Alle Pfade des Projekts an einer Stelle.

Vorher stand in jedem Modul `Path(__file__).parent / "results"`. Das ging nur, solange
alle Skripte flach im Repo-Wurzelverzeichnis lagen -- mit der Paketstruktur waere jede
dieser Zeilen einzeln falsch geworden (die Module liegen jetzt zwei Ebenen tiefer). Statt
sie zwanzigmal zu reparieren, gibt es sie einmal hier.

Der Nebeneffekt ist der eigentliche Gewinn: es ist jetzt nachvollziehbar, wohin ein Lauf
schreibt, ohne zwanzig Dateien zu lesen.
"""

from __future__ import annotations

from pathlib import Path

#: Wurzel des Repositorys (enthaelt README.md, figures/, results/, drift_glm/).
ROOT = Path(__file__).resolve().parents[1]

#: Ergebnistabellen und -arrays (CSV, NetCDF, Markdown). Versioniert.
RESULTS = ROOT / "results"

#: Fortschrittsdateien und Nachtlauf-Logs. NICHT versioniert (.gitignore) -- sie aendern
#: sich bei jedem Lauf. Getrennt von RESULTS, damit dort nur echte Ergebnisse liegen.
LOGS = RESULTS / "logs"

#: Erzeugte Abbildungen. Bewusst FLACH mit Nummernpraefix 00..30 in Lesereihenfolge --
#: die Nummer ist die Ordnung, und die Zuordnung Nummer <-> Dateiname ist in
#: BESPRECHUNG.md (Anhang B) dokumentiert. Unterordner wuerden diese Zuordnung brechen.
FIGURES = ROOT / "figures"

#: Schwesterordner des Repos. Dort liegen die grossen externen Datensaetze (der
#: Finger-Tapping-Datensatz) und der Cedalion-Clone -- bewusst ausserhalb des Code-Repos.
EXTERNAL = ROOT.parent


def ensure() -> None:
    """Ausgabeverzeichnisse anlegen, falls sie fehlen. Idempotent."""
    for d in (RESULTS, LOGS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
