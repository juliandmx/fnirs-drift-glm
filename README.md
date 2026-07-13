# Driftregressoren im fNIRS-GLM — Analyse-Code

Code zur Bachelorarbeit *„Vergleich verschiedener Driftregressoren im General Linear Model
zur Schätzung der hämodynamischen Antwortfunktion in fNIRS-Daten"* (TU Berlin, FG
Neurotechnologie).

> **Ausführliche, laienverständliche Erklärung** des gesamten Projekts (was, wie, warum,
> mit Cedalion-Bezügen und Abbildungen): siehe **[`DOKUMENTATION.md`](DOKUMENTATION.md)**.
> Diese README ist die knappe technische Referenz zum Ausführen/Reproduzieren.

## Überblick

Simulationsbasierte Vergleichsstudie: In echte Ruhe-fNIRS-Daten wird eine synthetische
HRF mit **bekannter Ground Truth** (räumlicher Blob) eingemischt; anschließend wird per GLM
(AR-IRLS) geschätzt, wie genau verschiedene **Driftregressor-Familien** die Aktivierung
zurückgewinnen — variiert über Driftfamilie × Analysefenster × Regressor-Konstellation ×
Zufalls-Seeds. Aufbauend auf dem Framework **Cedalion** (Schwester-Ordner `../cedalion/`,
Version 26.5.1).

## Voraussetzungen

- Conda-Umgebung **`cedalion`** (Python 3.11), mit Cedalion editierbar installiert.
- Zusätzlich genutzt: `numpy`, `pandas`, `xarray`, `scipy`, `statsmodels`, `matplotlib`
  (allesamt Cedalion-Abhängigkeiten).
- Umgebungs-Export (für exakte Reproduktion) liegt in `environment.lock.txt`.

Aktivieren (einmal pro Terminal):
```bash
source ~/anaconda3/etc/profile.d/conda.sh
```
Alle Skripte dann mit `conda run -n cedalion python <skript>` starten.

## Struktur

```
fnirs-drift-glm/         (dieses Repository — als Git-Repo neben ../cedalion/ nutzbar)
  pipeline.py          Kern: Ruhedaten laden, synthetische HRF (räumlicher Blob) einmischen,
                       Designmatrix bauen. Wird von allen anderen Skripten genutzt.
  sweep.py             Systematischer Vergleich (Familie × Fenster × Konstellation × Seeds).
  sweep_report.py      Abbildungen + Markdown-Ergebnistabellen aus den Sweep-Ergebnissen.
  flex_basis.py        Flexible HRF-Recovery-Basis -> Formfehler unabhängig von der Amplitude.
  detection.py         Signifikanz je Kanal + FDR-Korrektur -> Detektionsgüte vs. Ground Truth.
  demo_recovery.py     End-to-end-Demo (Zahlen): ein Durchlauf, Fehlerkennzahlen.
  demo_figures.py      Diagnose-Abbildungen (Designmatrix, Ein-Kanal-Fit, Scalp-Karten).
  tests/               Smoke-Tests (pytest).
  results/             Ausgaben: sweep_summary.csv, sweep_per_channel.nc, tables.md,
                       flex_basis_summary.csv, detection_summary.csv, *_progress.txt.
  figures/             Erzeugte Abbildungen (PNG).
  environment.lock.txt Versions-Sperrdatei (Reproduzierbarkeit).
  README.md            Diese Datei (knappe technische Referenz).
  DOKUMENTATION.md     Vollständige, laienverständliche Projektdokumentation.
../cedalion/           Cedalion-Framework (Schwester-Ordner) inkl. Notebooks (examples/).
```

## Ausführen

```bash
source ~/anaconda3/etc/profile.d/conda.sh
cd fnirs-drift-glm                          # in diesen Repo-Ordner wechseln
CR="conda run -n cedalion python"

# Schnelltests (Sekunden–Minuten)
$CR -m pytest tests -q            # Smoke-Tests
$CR sweep.py pilot                # alle Code-Pfade, ~5 min
$CR flex_basis.py test            # ~1 min
$CR detection.py test             # ~1 min

# Machbarkeitsnachweis + Diagnose-Abbildungen
$CR demo_recovery.py
$CR demo_figures.py

# Hauptstudie (Hintergrund empfohlen)
$CR sweep.py v3                   # ~3 h;  Fortschritt: results/sweep_progress.txt
$CR flex_basis.py                 # ~20 min; Fortschritt: results/flex_basis_progress.txt
$CR detection.py                  # ~60–75 min (Fits über ALLE Kanäle); results/detection_progress.txt

# Auswertung / Abbildungen
$CR sweep_report.py
```

## Reproduzierbarkeit

- **Seeds:** Alle Zufallsanteile (Stimulus-Platzierung) sind über `seed` gesteuert und
  reproduzierbar (`random` + `numpy` werden in `pipeline.build` geseedet).
- **WSL/Windows:** Das Repo liegt auf `/mnt/c/`. Dateien **nur aus der Linux-/WSL-Seite**
  bearbeiten — native Windows-Editoren schreiben CRLF-Zeilenenden und lassen dann scheinbar
  den gesamten Cedalion-Baum als „verändert" erscheinen.
- **Cedalion-Version:** Entwicklungszweig `dev`, Version 26.5.1. Die verwendeten
  API-Bausteine und die zugehörigen Referenz-Notebooks sind in `DOKUMENTATION.md` (Kap. 3–4)
  aufgeführt.
