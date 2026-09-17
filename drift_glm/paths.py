"""Alle Pfade des Projekts an einer Stelle."""

from __future__ import annotations

from pathlib import Path

# Wurzel des Repositorys (README.md, figures/, results/, drift_glm/).
ROOT = Path(__file__).resolve().parents[1]

# Ergebnistabellen und -arrays (CSV, NetCDF, Markdown), versioniert.
RESULTS = ROOT / "results"

# Fortschrittsdateien und Nachtlauf-Logs, nicht versioniert (.gitignore).
LOGS = RESULTS / "logs"

# Abbildungen, flach mit Nummernpraefix 00..30 in Lesereihenfolge; die Zuordnung
# Nummer <-> Dateiname steht in BESPRECHUNG.md (Anhang B).
FIGURES = ROOT / "figures"

# Schwesterordner des Repos mit den grossen externen Datensaetzen und dem Cedalion-Clone.
EXTERNAL = ROOT.parent


def ensure() -> None:
    """Ausgabeverzeichnisse anlegen, falls sie fehlen."""
    for d in (RESULTS, LOGS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
