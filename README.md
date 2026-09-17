# fnirs-drift-glm

Code zur Bachelorarbeit „Vergleich verschiedener Driftregressoren im General Linear
Model zur Schätzung der hämodynamischen Antwortfunktion in fNIRS-Daten" (TU Berlin,
Fachgebiet Neurotechnologie).

Das Paket vergleicht Driftregressor-Familien (Polynome, DCT, Legendre, B-Splines, kein
Driftmodell, Butterworth-Hochpass als Filteralternative) im fNIRS-GLM. Grundlage ist eine
Simulation auf Ruhedaten (nn22, 520 Kanäle): eine Aktivierung mit bekannter Ground Truth
wird als Gauß-Blob auf dem Kortex unter C3/C4 erzeugt, über die Sensitivitätsmatrix in den
Kanalraum projiziert und vor der Bewegungskorrektur in die optische Dichte eingemischt;
anschließend wird mit AR-IRLS geschätzt und gegen die Wahrheit gemessen. Dieselben Modelle
laufen zusätzlich auf zwei realen Fingertapping-Datensätzen (Khan, 25 Probanden;
Multisubject-Fingertapping, 5 Probanden mit echten kurzen Kanälen). Alle Auswertungen
bauen auf Cedalion (Version 26.5.1, Entwicklungszweig `dev`) auf.

Die technische Projektdokumentation (Daten, Pipeline, Versuchsraster, Gütemaße,
Ergebnisse, Einschränkungen) steht in [`DOKUMENTATION.md`](DOKUMENTATION.md).

## Voraussetzungen

- Conda-Umgebung `cedalion` (Python 3.11) mit editierbar installiertem Cedalion aus dem
  Schwesterordner `../cedalion/` (Commit `a7eb625`, siehe `environment.lock.txt`).
- Weitere Abhängigkeiten sind Cedalion-Abhängigkeiten: `numpy`, `pandas`, `xarray`,
  `scipy`, `statsmodels`, `matplotlib`, `joblib`.
- Datensätze: nn22 und Multisubject-Fingertapping werden über `cedalion.data` geladen;
  der Khan-Datensatz liegt als SNIRF unter `../FingerTappingDataset_Published2025/`.
- Arbeitsspeicher: ein AR-IRLS-Kanalfit mit `dct:0.02` hält 2,9 GB, die
  nn22-Sensitivitätsmatrix rund 300 MB. Auf einer 8-GB-Maschine läuft nur ein großer
  Lauf gleichzeitig.

Alle Module werden als Modul aufgerufen (`python -m drift_glm.<paket>.<modul>`), nicht als
Datei: `realglm.py` und `msglm.py` verteilen Fits über `joblib`/`loky`, und das
funktioniert nur, wenn die Worker-Funktionen importierbar sind.

## Struktur

```
fnirs-drift-glm/
  drift_glm/                    Python-Paket, vier Schichten: core -> data -> analysis -> reports
    paths.py                    ROOT, RESULTS, LOGS, FIGURES, EXTERNAL
    figstyle.py                 Farben und Reihenfolge der Driftfamilien für alle Abbildungen
    core/                       importiert nichts aus den anderen Schichten
      preprocess.py             int2od, Bewegungskorrektur, Qualitätsmasken, Pruning, od2conc
      shortchannel.py           Long/Short-Split, Short-Regressoren, Global-Component-Subtraktion
      imagespace.py             ICBM152, Sensitivitätsmatrix, C3/C4-Blob, Vorwärts- und Rückweg
      pipeline.py               Simulation: Ground Truth, Injektion, Designmatrix
      fitstats.py               R², adj. R², Residual-RMS im Datenraum
    data/
      realdata.py               Khan-Datensatz (25 Probanden, 48 Kanäle)
      multisubject.py           Multisubject-Fingertapping (5 Probanden, 8 kurze Kanäle)
      coregister.py             landmarkenfreie Registrierung der NIRScout-Montage
    analysis/                   jede Auswertung schreibt eine Tabelle nach results/
      sweep.py                  Kanalraum-Hauptstudie (Presets pilot, v4)
      flex_basis.py             Formtreue mit flexibler HRF-Basis (Abb. 12, 13)
      detection.py              Signifikanz je Kanal, FDR, Detektionsgüte (Abb. 14)
      residuals.py              Residualspuren, Spektren, HRF je Familie (Abb. 27, 28)
      compare_preprocessing.py  Vorverarbeitungsvarianten gegen die beta-Rückgewinnung (Abb. 11)
      imageglm.py               GLM-Ergebnis im Bildraum (HRF, Residuum, Residuum+HRF)
      recon_check.py            rauschfreie Kontrolle der Rekonstruktion und Regularisierung
      realglm.py                Khan: GLM je Familie, Gruppen-t-Test, Reproduzierbarkeit
      msglm.py                  Multisubject: Familien x Systemik-Achse, Lateralisierung
      mshrf.py                  Multisubject: HRF je Familie, Fit-Metriken (Abb. 29, 30)
    reports/
      flowchart.py              Abb. 00 (PDF und PNG)
      demo_figures.py           Abb. 1-5, 24
      sweep_report.py           Abb. 6-10, 25, 26 und results/tables.md
      realglm_report.py         Abb. 15-18
      imagespace_report.py      Abb. 19-23
  tests/                        55 pytest-Tests in acht Modulen (Smoke, Bildraum, Fit-Metriken,
                                Filterung, Reliabilität, Designrang, Provenienz, Khan-Zuordnung)
  results/                      CSV, NetCDF, tables.md (versioniert); logs/ nicht versioniert
  figures/                      Abbildungen 00-30, flach mit Nummernpräfix
  run_v6.sh                     sequenzielle Kette aller Schritte (flock-Sperre, Abbruch bei Fehler)
  environment.lock.txt          Versionsstand der Umgebung (maschinenspezifisch)
  environment.yml               portable Umgebungsbeschreibung
  README.md, DOKUMENTATION.md
../cedalion/                    Cedalion-Quelltext (dev), editierbar installiert
../FingerTappingDataset_Published2025/   Khan-Datensatz (SNIRF)
```

## Ausführen

```bash
source ~/anaconda3/etc/profile.d/conda.sh
cd fnirs-drift-glm
M="conda run -n cedalion python -m"
```

### Schnelltests (Sekunden bis Minuten)

```bash
$M pytest tests -q                          # 31 Tests
$M drift_glm.analysis.sweep pilot           # alle Code-Pfade, ~5 min
$M drift_glm.analysis.flex_basis test       # ~1 min
$M drift_glm.analysis.detection test        # ~1 min
$M drift_glm.core.imagespace                # Selbsttest: Adot, C3/C4-Seeds, Vorwärtsmodell
$M drift_glm.analysis.recon_check           # rauschfreie Rekonstruktion, Regularisierung
$M drift_glm.core.pipeline leakage          # Anteil der Injektion in den kurzen Kanälen
$M drift_glm.data.multisubject              # Inventar der 5 Probanden
$M drift_glm.data.realdata                  # Inventar des Khan-Datensatzes
$M drift_glm.data.coregister                # Khan-Montage: Registrierung und Bericht
```

### Hauptläufe

Fortschritt je Lauf in `results/logs/<name>_progress.txt`. Der Sweep schreibt seine CSV
am Ende (Fortschrittsdatei nach jedem Fit); `imageglm` und `msglm` schreiben nach jeder
Zelle, `msglm` setzt einen abgebrochenen Lauf fort.

```bash
$M drift_glm.analysis.sweep v4 --replace     # ~7 h    Kanalraum-Hauptstudie, 1800 Zellen (archiviert alte Ergebnisse)
$M drift_glm.analysis.sweep butterxy         # ~25 min Kontrollarm (Daten + Design gefiltert), danach: sweep merge
$M drift_glm.analysis.sweep migrate-dct      # einmalig: adj. R² der DCT-Zellen auf eine Konstante umgerechnet
$M drift_glm.analysis.flex_basis             # ~1 h
$M drift_glm.analysis.detection              # ~1,5 h  Fits über alle Kanäle
$M drift_glm.analysis.residuals              # ~20 min
$M drift_glm.analysis.compare_preprocessing  # ~15 min
$M drift_glm.analysis.imageglm               # ~3 h    Bildraum, 24 Zellen à 420 s Fit
$M drift_glm.analysis.realglm                # ~3,6 h  Khan, 3864 Fits
$M drift_glm.analysis.realglm supplement     # ~30 min Ergänzungsarme lowpass:0.5, butterxy:0.01; danach: realglm merge
$M drift_glm.analysis.khan_lateralisation    # ~10 min Hemisphären-Kontrolle der Gruppenkarten (Khan Tab. 3)
$M drift_glm.analysis.msglm                  # ~5 h    81-Zellen-Raster (OLS voll, AR-IRLS none/poly:3/butter)
$M drift_glm.analysis.msglm butterxy         # ~30 min OLS-Kontrollarm, eigene CSV
$M drift_glm.analysis.msglm reliability --reuse-full-from results/archive/<msglm_summary_...>.csv --output results/msglm_summary.csv --no-resume
                                             #          nur die gemeinsamen Even/Odd-Fits neu, Vollfit aus dem Archiv
$M drift_glm.analysis.mshrf                  # ~25 min
```

### Abbildungen und Tabellen

```bash
$M drift_glm.reports.flowchart               # Abb. 00
$M drift_glm.reports.demo_figures            # Abb. 1-5, 24
$M drift_glm.reports.sweep_report            # Abb. 6-10, 25, 26 und results/tables.md
$M drift_glm.reports.realglm_report          # Abb. 15-18 (Abb. 17 ~13 min; "quick" lässt sie weg)
$M drift_glm.reports.imagespace_report       # Abb. 19-23 (~1 h; Unterbefehle hrf, cortex, tables)
```

### Kette

```bash
nohup ./run_v6.sh > results/logs/run_v6.log 2>&1 &
tail -f results/logs/*_progress.txt
./run_v6.sh msglm tables                     # nur einzelne Schritte
```

Schritte in dieser Reihenfolge: `demo residuals mshrf sweep sweepxy sweeprep flex detection
imageglm report msglm realglm tables`. Die Kette läuft sequenziell mit `flock`-Sperre, weil der
Speicher der Engpass ist (2,9 GB je dct:0.02-AR-IRLS-Kanalfit); ein fehlgeschlagener Schritt
beendet sie (`results/logs/run_v6_status.json`). Jede Analyse legt eine `.meta.json` mit
Code-/Daten-Fingerabdruck neben ihre CSV und archiviert ersetzte Dateien unter
`results/archive/`.

## Reproduzierbarkeit

- Seeds: die Stimulus-Platzierung ist der einzige Zufallsanteil; `pipeline.build` seedet
  `random` und `numpy` mit dem übergebenen `seed`. Der Sweep verwendet die Seeds 0-3,
  `detection` 0-1, `imageglm` und `residuals` den Seed 0.
- Cedalion: Entwicklungszweig `dev`, Version 26.5.1, Commit `a7eb625`
  (`environment.lock.txt`). Verwendete API: `glm.fit(..., noise_model="ar_irls")` mit
  `.sm.params`, `drift_regressors`, `drift_cosine_regressors`, `drift_legendre_regressors`,
  `GaussianKernels`, `build_spatial_activation`, `get_precomputed_sensitivity`,
  `image_to_channel_space`, `ImageRecon`.
- WSL/Windows: das Repository liegt auf `/mnt/c/`. Dateien nur aus der Linux-Seite
  bearbeiten; native Windows-Editoren schreiben CRLF-Zeilenenden, wodurch der gesamte
  Cedalion-Baum als geändert erscheint.
- Ergebnisstand: Sweep, flex, detection, imageglm und die Abbildungen 00-30 stammen vom
  Lauf am 8./9. September 2026 (Bildraum-Ground-Truth nach dem Fix der
  Kanalreihenfolge); `realglm_summary.csv` und `msglm_summary.csv` sind Realdaten und
  davon nicht betroffen.
