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
  preprocess.py        Vorverarbeitungskette (OD, Motion Correction, Kanalmasken, Pruning).
  shortchannel.py      Long/Short-Split, Short-Regressoren, Global-Component-Subtraktion.
  imagespace.py        Bildraum-Fundament: Kopfmodell ICBM152, Adot, C3/C4-Blob,
                       Vorwärts-/Rückweg, Sichtbarkeitsmaske, Regularisierung.
  pipeline.py          Kern der Simulation: Ruhedaten laden, synthetische HRF im BILDRAUM
                       erzeugen, in den Kanalraum tragen, dort einmischen, Designmatrix.
  sweep.py             Systematischer Vergleich (Familie × Fenster × Konstellation × Seeds).
  sweep_report.py      Abbildungen + Markdown-Ergebnistabellen aus den Sweep-Ergebnissen.
  imageglm.py          GLM-Ergebnis (HRF / Residuum / Residuum+HRF) zurück auf den Kortex,
                       Gütemaße inkl. Lokalisationsfehler.
  flex_basis.py        Flexible HRF-Recovery-Basis -> Formfehler unabhängig von der Amplitude.
  detection.py         Signifikanz je Kanal + FDR-Korrektur -> Detektionsgüte vs. Ground Truth.
  realdata.py          Stufe 2 (Khan, 25 Probanden, 48 Kanäle): Loader, Inventar, Stimuli.
  realglm.py           Stufe 2, Kern: GLM je Familie, Gruppen-t-Test, Reproduzierbarkeit.
  realglm_report.py    Abbildungen zu den Khan-Daten.
  coregister.py        Landmarkenfreie Koregistrierung der NIRScout-Montage auf ICBM152.
  multisubject.py      Stufe 3 (Cedalion, 5 Probanden, echte Short Channels): Loader.
  msglm.py             Stufe 3, Kern: Familien × Systemik-Achse, Halbierungs-
                       Reproduzierbarkeit, kontralaterale Kontrolle, Bildraum.
  imagespace_report.py Abbildungen 19–23 (Bildraum und Multisubject).
  demo_recovery.py     End-to-end-Demo (Zahlen): ein Durchlauf, Fehlerkennzahlen.
  demo_figures.py      Diagnose-Abbildungen (Designmatrix, Ein-Kanal-Fit, Scalp-Karten).
  tests/               Smoke-Tests (pytest): test_smoke.py, test_imagespace.py.
  results/             Ausgaben: sweep_summary.csv, sweep_per_channel.nc, tables.md,
                       imageglm_summary.csv, msglm_summary.csv, *_progress.txt.
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

### Bildraum (Kopfmodell ICBM152)

```bash
$CR imagespace.py                     # Selbsttest: Adot, C3/C4-Seeds, Vorwärtsmodell
$CR imagespace.py check               # Regularisierung empirisch prüfen (rauschfrei)
$CR pipeline.py leakage               # wie viel HRF landet in den kurzen Kanälen?
$CR imageglm.py test                  # ~5 min;  voll: $CR imageglm.py  (mehrere Stunden)
$CR imagespace_report.py              # Abb. 19-23
```

> **Speicher.** Die nn22-Sensitivitätsmatrix belegt im Arbeitsspeicher rund 300 MB, eine
> `ImageRecon`-Instanz darauf ähnlich viel. Auf einer 8-GB-Maschine **immer nur einen**
> Bildraum-Lauf gleichzeitig starten – zwei parallele Läufe wurden hier vom OOM-Killer
> beendet. Die 28-Kanal-Montage der Stufe 3 ist dagegen unkritisch (~16 MB).

### Stufe 3: Multisubject-Fingertapping (5 Probanden, echte Short Channels)

```bash
$CR multisubject.py                   # Inventar aller 5 Probanden
$CR multisubject.py sub-01            # Inventar + Vorverarbeitung eines Probanden
$CR msglm.py test                     # 2 Probanden, 2 Familien (Timing)
$CR msglm.py                          # voll (Nachtlauf); Fortschritt: results/msglm_progress.txt
```

### Koregistrierung der Khan-Montage

```bash
$CR coregister.py                     # Registrierung + Blocker-Bericht (keine GPU nötig)
```

### Nachtlauf (alles auf einmal)

```bash
nohup ./run_v5.sh > results/run_v5.log 2>&1 &
tail -f results/imageglm_progress.txt results/msglm_progress.txt
```

Läuft **sequenziell**: `imageglm.py` → `msglm.py` → `imagespace_report.py`. Gemessene
Größenordnungen auf dieser Maschine:

| Schritt | Kosten | woran es hängt |
|---|---|---|
| `imageglm.py` | ~3 h | 420 s je AR-IRLS-Fit über 519 Kanäle, 24 Zellen, plus ~20 min Build |
| `msglm.py` | ~5 h | AR-IRLS kostet hier das ~200-Fache von OLS (23 240 Samples je Kanal) |
| `imagespace_report.py` | ~1 h | Abb. 19 und 20 brauchen je einen Build plus einen Fit |

Beide Auswertungen schreiben ihre CSV **nach jeder Zelle** – ein Abbruch verliert nur die
laufende Zelle. Der Report ist fehlertolerant: eine fehlgeschlagene Abbildung nimmt die
übrigen nicht mit.

## Reproduzierbarkeit

- **Seeds:** Alle Zufallsanteile (Stimulus-Platzierung) sind über `seed` gesteuert und
  reproduzierbar (`random` + `numpy` werden in `pipeline.build` geseedet).
- **WSL/Windows:** Das Repo liegt auf `/mnt/c/`. Dateien **nur aus der Linux-/WSL-Seite**
  bearbeiten — native Windows-Editoren schreiben CRLF-Zeilenenden und lassen dann scheinbar
  den gesamten Cedalion-Baum als „verändert" erscheinen.
- **Cedalion-Version:** Entwicklungszweig `dev`, Version 26.5.1. Die verwendeten
  API-Bausteine und die zugehörigen Referenz-Notebooks sind in `DOKUMENTATION.md` (Kap. 3–4)
  aufgeführt.
