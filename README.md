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
Alle Auswertungen laufen als Modul: `conda run -n cedalion python -m drift_glm.<paket>.<modul>`.

## Struktur

```
fnirs-drift-glm/              (dieses Repository — als Git-Repo neben ../cedalion/ nutzbar)
  drift_glm/                  das Python-Paket. Vier Schichten, Abhängigkeiten nur nach unten.
    paths.py                  ROOT / RESULTS / LOGS / FIGURES — alle Pfade an einer Stelle.
    core/                     Bausteine, die alle Auswertungen teilen
      preprocess.py             OD, Bewegungskorrektur, Kanalmasken, Pruning, Amplitudengrenzen
      shortchannel.py           Long/Short-Split, Short-Regressoren, Global-Comp-Subtraktion
      imagespace.py             Kopfmodell ICBM152, Adot, C3/C4-Blob, Vorwärts-/Rückweg
      pipeline.py               Simulation: Blob im Bildraum → Kanalraum → einmischen → Designmatrix
    data/                     Einlesen der Datensätze
      realdata.py               Stufe 2: Khan, 25 Probanden, 48 Kanäle
      multisubject.py           Stufe 3: Cedalion, 5 Probanden, echte Short Channels
      coregister.py             NIRScout-Montage landmarkenfrei auf ICBM152
    analysis/                 die Auswertungen — jede schreibt eine Tabelle nach results/
      sweep.py                  Kanalraum: Familie × Fenster × Konstellation × Seeds
      imageglm.py               Bildraum: HRF / Residuum / Residuum+HRF auf den Kortex
      realglm.py                Khan: Familien, Gruppen-t-Test, Reproduzierbarkeit
      msglm.py                  Multisubject: Familien × Systemik-Achse, Lateralisierung
      flex_basis.py             Formtreue mit flexibler HRF-Vorlage
      detection.py              Signifikanz je Kanal + FDR gegen die Ground Truth
      compare_preprocessing.py  Vorverarbeitungs-Varianten gegen die β-Rückgewinnung
      recon_check.py            rauschfreie Kontrolle des Bildraum-Kreises + Regularisierung
    reports/                  Abbildungen und Tabellen AUS den Ergebnissen
      sweep_report.py           Abb. 6–10 + results/tables.md
      realglm_report.py         Abb. 15–18
      imagespace_report.py      Abb. 19–23
      demo_figures.py           Abb. 1–5 (Kontrollabbildungen eines Durchlaufs)
      demo_recovery.py          End-to-end-Demo als Zahlen
  tests/                      Smoke-Tests (pytest): test_smoke.py, test_imagespace.py
  results/                    Ergebnistabellen (CSV, NetCDF, tables.md) — versioniert
    logs/                     Fortschrittsdateien und Nachtlauf-Logs — NICHT versioniert
  figures/                    Abbildungen, flach mit Nummernpräfix 01–23 in Lesereihenfolge
  run_v5.sh                   Nachtlauf: imageglm → msglm → Abbildungen, sequenziell
  environment.lock.txt        Versions-Sperrdatei (Reproduzierbarkeit)
  README.md                   Diese Datei (knappe technische Referenz)
  DOKUMENTATION.md            Vollständige, laienverständliche Projektdokumentation
../cedalion/                  Cedalion-Framework (Schwester-Ordner) inkl. Notebooks (examples/)
```

> **Warum ein Paket und nicht flache Skripte.** Vorher lagen 20 Module flach im
> Wurzelverzeichnis. Die Schichtung macht die Abhängigkeitsrichtung sichtbar und
> erzwingbar: `reports` darf `analysis` benutzen, `analysis` darf `data` und `core`,
> `core` kennt die oberen Schichten nicht. Wer das umdreht, baut einen Importzyklus — und
> der fällt sofort auf, statt sich als Reihenfolge-Abhängigkeit zu verstecken.
>
> **Aufruf deshalb immer als Modul**, nie als Datei: `python -m drift_glm.analysis.sweep`.
> Das ist nicht nur Kosmetik — `realglm.py` und `msglm.py` verteilen ihre Fits über
> `joblib`/`loky` auf Prozesse, und loky pickelt Funktionen aus `__main__` per *Wert*
> (woran cedalion-Objekte scheitern). Als Modul importiert werden sie per *Referenz*
> gepickelt, und die Kindprozesse importieren selbst.

## Ausführen

```bash
source ~/anaconda3/etc/profile.d/conda.sh
cd fnirs-drift-glm                          # in diesen Repo-Ordner wechseln
M="conda run -n cedalion python -m"         # alles läuft als Modul, nicht als Datei
```

### Schnelltests (Sekunden–Minuten)

```bash
$M pytest tests -q                          # 23 Smoke-Tests
$M drift_glm.analysis.sweep pilot           # alle Code-Pfade, ~5 min
$M drift_glm.analysis.flex_basis test       # ~1 min
$M drift_glm.analysis.detection test        # ~1 min
$M drift_glm.core.imagespace                # Selbsttest: Adot, C3/C4-Seeds, Vorwärtsmodell
$M drift_glm.analysis.recon_check           # Regularisierung empirisch prüfen (rauschfrei)
$M drift_glm.core.pipeline leakage          # wie viel HRF landet in den kurzen Kanälen?
$M drift_glm.data.multisubject              # Inventar der 5 Probanden
$M drift_glm.data.coregister                # Khan-Montage: Registrierung + Blocker-Bericht
```

### Hauptläufe (Hintergrund empfohlen)

Fortschritt liegt jeweils in `results/logs/<name>_progress.txt`.

```bash
$M drift_glm.analysis.sweep v4               # ~7,5 h  Kanalraum-Hauptstudie
$M drift_glm.analysis.imageglm               # ~3 h    Bildraum, Simulation
$M drift_glm.analysis.realglm                # ~3,6 h  Khan, 25 Probanden
$M drift_glm.analysis.msglm                  # ~5 h    Multisubject
$M drift_glm.analysis.flex_basis             # ~20 min
$M drift_glm.analysis.detection              # ~60–75 min (Fits über ALLE Kanäle)
$M drift_glm.analysis.compare_preprocessing  # Vorverarbeitungs-Varianten
```

### Abbildungen und Tabellen

```bash
$M drift_glm.reports.demo_recovery           # End-to-end-Demo als Zahlen
$M drift_glm.reports.demo_figures            # Abb. 1–5
$M drift_glm.reports.sweep_report            # Abb. 6–10 + results/tables.md
$M drift_glm.reports.realglm_report          # Abb. 15–18
$M drift_glm.reports.imagespace_report       # Abb. 19–23
```

> **Speicher.** Die nn22-Sensitivitätsmatrix belegt im Arbeitsspeicher rund 300 MB, eine
> `ImageRecon`-Instanz darauf ähnlich viel. Auf einer 8-GB-Maschine **immer nur einen**
> Bildraum-Lauf gleichzeitig starten – zwei parallele Läufe wurden hier vom OOM-Killer
> beendet. Die 28-Kanal-Montage der Stufe 3 ist dagegen unkritisch (~16 MB).

### Nachtlauf (alles auf einmal)

```bash
nohup ./run_v5.sh > results/logs/run_v5.log 2>&1 &
tail -f results/logs/*_progress.txt
```

Läuft **sequenziell** `imageglm` → `msglm` → Abbildungen, mit PID-Sperre gegen zwei
parallele Ketten. Einzelne Schritte überspringen: `./run_v5.sh msglm report`.

| Schritt | Kosten | woran es hängt |
|---|---|---|
| `imageglm` | ~3 h | 420 s je AR-IRLS-Fit über 519 Kanäle, 24 Zellen, plus ~20 min Build |
| `msglm` | ~5 h | AR-IRLS kostet hier das ~200-Fache von OLS (23 240 Samples je Kanal) |
| Abbildungen | ~1 h | Abb. 19 und 20 brauchen je einen Build plus einen Fit |

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
