# Projektdokumentation fnirs-drift-glm

Bachelorarbeit „Vergleich verschiedener Driftregressoren im General Linear Model zur
Schätzung der hämodynamischen Antwortfunktion in fNIRS-Daten", TU Berlin, Fachgebiet
Neurotechnologie. Stand der Ergebnisse: 9. September 2026.

Inhalt

1. [Ziel und Fragestellung](#1-ziel-und-fragestellung)
2. [Daten](#2-daten)
3. [Pipeline](#3-pipeline)
4. [Versuchsraster](#4-versuchsraster)
5. [Gütemaße](#5-gütemaße)
6. [Ergebnisse](#6-ergebnisse)
7. [Dateien und Ausführung](#7-dateien-und-ausführung)
8. [Bekannte Einschränkungen](#8-bekannte-einschränkungen)

## 1 Ziel und Fragestellung

Das GLM erklärt eine fNIRS-Zeitreihe als gewichtete Summe aus HRF-Regressor,
Driftregressoren und optionalen Störregressoren. Der Koeffizient des HRF-Regressors
(beta) ist die Zielgröße. Untersucht wird, welche Driftregressor-Familie unter welchen
Bedingungen (Fensterlänge, weitere Regressoren, Bewegungskorrektur) die genaueste
beta-Schätzung liefert und wie sich die Wahl auf Form, Detektion und Ort der geschätzten
Aktivierung auswirkt.

Verglichen werden Polynome (`poly:n`), Legendre-Polynome (`legendre:n`), eine diskrete
Cosinus-Basis (`dct:f`), kubische B-Splines (`bspline:k`), ein reines Offset-Modell
(`none`) und als Filteralternative ein Butterworth-Hochpass (`butter:0.01`), bei dem die
Zeitreihe vor dem Fit gefiltert wird und die Designmatrix nur den Offset enthält. In allen
Regressor-Armen wird nicht gefiltert; der Drift wird ausschließlich über die Designmatrix
modelliert. Schätzer ist AR-IRLS.

Die Arbeit hat zwei Stränge. In der Simulation wird eine Aktivierung mit bekannter
Amplitude in Ruhedaten eingemischt; alle Fehlermaße beziehen sich auf diese Ground Truth.
Auf zwei realen Fingertapping-Datensätzen gibt es keine Wahrheit; dort werden
Reproduzierbarkeit, Signifikanz nach FDR, HbO/HbR-Plausibilität, Modellfit (adj. R²,
Residuen) und beim Multisubject-Datensatz die kontralaterale Organisation der Motorik als
Kriterien verwendet.

## 2 Daten

### 2.1 nn22-Ruhedatensatz (Simulation)

`cedalion.data.get_nn22_resting_state()`: eine Person in Ruhe, 368 s, 567 Kanäle,
760/850 nm, ca. 9 Hz, Accelerometer und Gyroskop als Aux, Dunkelsignal. Nach dem
Positivitäts-Gate (6 Kanäle mit Amplitude 0) und dem Pruning (41 Kanäle) bleiben 520
Kanäle. Der Rauschboden aus dem Dunkelsignal liegt bei 9,5e-6 V; die Amplitudenuntergrenze
1e-3 V liegt beim 105-Fachen davon, und zwischen dem 50- und dem 105-Fachen fällt kein
weiterer Kanal heraus. Der kürzeste Quell-Detektor-Abstand beträgt 15,6 mm; bei der
Grenze 1,8 cm zählen 37 Kanäle (15,8-17,8 mm) als kurz, bei 1,9 cm wären es 114. Für die
Montage liegt eine vorberechnete Sensitivitätsmatrix auf ICBM152 vor.

### 2.2 Khan-Datensatz (25 Probanden)

Khan, Nazeer & Mirtaheri (2026), „Open access individual finger movement dataset with
fNIRS", SNIRF auf figshare. Einzelfinger-Tapping der rechten Hand, 10 s Block, 10 s Ruhe
(S25: 15 s), 15 Blöcke je Aufnahme, 350 s. NIRScout, 16 Quellen x 16 Detektoren = 48
Kanäle bei einheitlich 3 cm, 760/850 nm, 3,9063 Hz (S25: 10,1725 Hz). 25 Probanden mit je
2-3 Durchgängen, 69 Aufnahmen, 74 Durchgangspaare. Keine kurzen Kanäle, kein
Bewegungs-Aux, keine Landmarken in `geo3d`. Verwendet wird ausschließlich die
unverarbeitete Fassung (`*_TRIM`); die von den Autoren gefilterte Fassung
(`*_CC_filtered`) hat bereits einen Hochpass bei 0,01 Hz angewandt und wird von
`realdata.find_files()` ausgeblendet.

### 2.3 Multisubject-Fingertapping (5 Probanden)

BIDS-NIRS-Tapping über `cedalion.data` (Referenz-Notebook `50b`). 5 Probanden, 28 Kanäle,
davon 8 kurze bei 7-8 mm und 20 lange bei 33-41 mm (Grenze 15 mm), 7,8125 Hz, 2974,5 s je
Aufnahme, je 30 Trials `control`, `Tapping/Left`, `Tapping/Right` (Blockdauer 5 s), kein
Aux. Landmarken LPA/Nasion/RPA sind vorhanden, die Sensitivitätsmatrix ist vorberechnet
(`get_precomputed_sensitivity("fingertapping", "icbm152")`).

## 3 Pipeline

### 3.1 Vorverarbeitung (`core/preprocess.py`)

Reihenfolge: Rohamplitude -> `int2od` mit gespeicherter Baseline -> Bewegungskorrektur
auf der OD -> `od2int` zurück zur Amplitude -> Qualitätsmasken auf der korrigierten
Amplitude -> Pruning -> `od2conc`. Die Kanalqualität (dunkel, gesättigt) ist nur auf der
Amplitude definiert, die Korrektur arbeitet auf der OD; die Baseline erlaubt den Rückweg
(`amp = baseline * exp(-od)`).

| Parameter | Wert |
|---|---|
| Bewegungskorrektur | `wavelet` (Standard); `tddr+wavelet` als zweite Achsenstufe |
| SNR-Schwelle | 3 |
| Amplitudenfenster | 1e-3 bis 0,84 V |
| Quell-Detektor-Abstand | 0 bis 4,5 cm |
| DPF | 6 |

Wavelet lässt das Driftband unter 0,01 Hz unangetastet, TDDR entfernt einen großen Teil
davon und dämpft auch eine eingemischte HRF (Abschnitt 6.3). Die synthetische Aktivierung
wird deshalb in die OD eingemischt, bevor die Bewegungskorrektur läuft. Für die
Filterfamilien wird `freq_filter` auf der Konzentration angewandt.

### 3.2 Ground Truth (`core/imagespace.py`, `core/pipeline.py`)

Die Aktivierung entsteht auf dem Kortex des ICBM152-Kopfmodells als geodätische
Gauß-Glocke (sigma 20 mm) um die Vertices unter den 10-20-Punkten C3 und C4
(`build_spatial_activation`), HbR mit dem Faktor -0,4. Über die Sensitivitätsmatrix
(`image_to_channel_space`) entsteht daraus eine OD-Karte je Kanal und Wellenlänge; sie
wird so kalibriert, dass der HbO-Peak im Kanalraum 0,6 µM beträgt (HbR -0,24 µM). Für die
Kalibrierung zählen nur Kanäle mit mindestens 10 % der maximalen Hirnsensitivität, weil
`od2conc` durch den Quell-Detektor-Abstand teilt und kurze Kanäle sonst mit unphysiologisch
großen Konzentrationen das Maximum stellen. Die Kanalkarte wird nach Namen ausgerichtet,
weil `od2conc` die Kanäle nicht in Eingabereihenfolge zurückgibt (Abschnitt 8).

Zeitverlauf: Blöcke von 10 s mit zufälligen Abständen von 25-35 s (`build_stim_df`,
geseedet), Gamma-Basis mit tau = 0 s, sigma = 3 s, T = 0 s (die Blockbreite entsteht aus
der Faltung in `hrf_regressors`, T wird nicht auf die Stimulusdauer gesetzt). Der
HRF-Regressor ist auf Peak 1 normiert; damit ist der GLM-Koeffizient direkt die
Peak-Konzentrationsänderung in µM, und Injektion und Fit verwenden denselben Regressor.
Die OD-Aktivierung wird zu den Ruhedaten addiert, danach folgt die zweite Hälfte der
Vorverarbeitung.

### 3.3 Driftfamilien (`analysis/sweep.py: drift_dm`)

| Familie | Umsetzung | Spalten (368 s) |
|---|---|---|
| `poly:n` | `drift_regressors(drift_order=n)` | n + 1 |
| `legendre:n` | `drift_legendre_regressors(order=n)` | n + 1 |
| `dct:f` | `drift_cosine_regressors(fmax=f)` plus Offset | 1 + Cosinus-Terme bis f |
| `bspline:k` | eigene kubische B-Spline-Basis mit k Splines, geklemmt, enthält den Offset | k |
| `none` | nur Offset | 1 |
| `butter:0.01` | Hochpass 0,01 Hz auf der Konzentration, Designmatrix nur Offset | 1 |
| `lowpass:0.5`, `bandpass:0.01-0.5` | Filter, nur Offset; nur im Pilot und bei Khan | 1 |

`poly:n` und `legendre:n` spannen denselben Raum auf und liefern identische Fits; sie
unterscheiden sich in der numerischen Konditionierung unter AR-IRLS (Abschnitt 6.1).
`dct:f` enthält keinen Cosinus-Term, wenn f unter der ersten Frequenz 1/T liegt:
`dct:0.005` und `dct:0.01` bei 90 s sowie `dct:0.005` bei 180 s sind gleich `none`.

### 3.4 Konstellationen

| Name | Zusätzliche Regressoren |
|---|---|
| `baseline` | keine |
| `motion` | Accelerometer- und Gyroskopkanäle, z-normiert |
| `global` | Mittel aller Kanäle |
| `short_avg` | Mittel der kurzen Kanäle (nn22: 15,8-17,8 mm; Multisubject: 7-8 mm) |
| `short_maxcorr` | je langem Kanal der am stärksten korrelierende kurze Kanal |
| `short_closest` | je langem Kanal der räumlich nächste kurze Kanal (nur msglm) |

Bei `short_*` wird nur über die langen Kanäle ausgewertet. `msglm` unterscheidet
zusätzlich, ob der systemische Regressor in der Designmatrix bleibt (`_dm`) oder vor dem
Fit abgezogen wird (`_sub`, Global-Component-Subtraktion nach Notebook 50b).

### 3.5 Schätzer und Fit-Metriken

`glm.fit(..., noise_model="ar_irls")` mit AR-Ordnung 30, Koeffizienten aus `.sm.params`,
Standardfehler aus `.sm.regressor_variances()`. OLS läuft im Pilot, auf den Realdaten als
zweites Rauschmodell und in `msglm`/`mshrf` als Hauptschätzer (AR-IRLS kostet dort das
rund 200-Fache). `core/fitstats.py` berechnet Residuum, Residual-RMS, R² und adj. R² im
Datenraum über `glm.predict`, also nicht im prewhitened Raum.

### 3.6 Bildraum (`core/imagespace.py`, `analysis/imageglm.py`)

Rückweg über `ImageRecon` mit `alpha_meas = 0,01 / median(C_meas)` (aus den Daten
geschätzt), `alpha_spatial = 0,001`, nur Hirnvertices. Ausgewertet wird nur innerhalb der
Sichtbarkeitsmaske (Vertexsensitivität über 1 % des Maximums, 9817 Vertices bei nn22).
Drei Projektionen des GLM-Ergebnisses werden zu einer Amplitudenkarte je Kanal verdichtet
(Mittel 4-10 s nach Onset) und rekonstruiert: `hrf` (beta-Karte), `residual`
(blockgemitteltes Residuum), `cleaned` (Residuum plus HRF-Anteil). Die Ground Truth
durchläuft denselben Rückweg als rauschfreie Obergrenze (`truth_ref`).

## 4 Versuchsraster

### 4.1 Kanalraum-Sweep (Preset `v4`)

| Achse | Stufen |
|---|---|
| Driftfamilie | poly:1-5, dct:0.005/0.01/0.02, legendre:1/3/5, none, butter:0.01, bspline:5/8 (15) |
| Fenster | 90, 180, 368 s |
| Konstellation | baseline, motion, global, short_avg, short_maxcorr |
| Bewegungskorrektur | wavelet, tddr+wavelet |
| Seeds | 0-3 |

1800 Zellen, 6,8 h Laufzeit, Median 11 s je Zelle. Ausgewertet werden je Zelle die 20
Kanäle mit der größten wahren Amplitude. Ausgaben: `results/sweep_summary.csv`,
`results/sweep_per_channel.nc`, `results/sweep_meta.json`; `reports/sweep_report.py`
erzeugt daraus Abb. 6-10, 25, 26 und `results/tables.md`. Das Preset `pilot` (8 Familien
inkl. `lowpass:0.5` und `bandpass:0.01-0.5`, 90 s, 4 Konstellationen, Seed 0, AR-IRLS und
OLS) prüft alle Code-Pfade in etwa 5 min.

### 4.2 Zusatzanalysen auf nn22

| Modul | Raster | Ausgabe |
|---|---|---|
| `flex_basis` | 8 Familien, 180 s, baseline, Seeds 0-3, 20 Kanäle, `GaussianKernels` als Recovery-Basis | `flex_basis_summary.csv`, Abb. 12, 13 |
| `detection` | 6 Familien, 180 s, baseline, Seeds 0-1, alle 519 Kanäle, FDR q = 0,05, aktiv wenn wahres beta über 10 % des Peaks (0,06 µM): 51 HbO-, 26 HbR-Kanäle | `detection_summary.csv`, Abb. 14 |
| `residuals` | 9 Familien, 90 und 368 s, Seed 0, 20 Kanäle; Spektren, Driftband-Anteil unter 0,02 Hz, HRF je Familie mit flexibler Basis | `residuals_summary.csv`, Abb. 27, 28 |
| `compare_preprocessing` | ohne Korrektur / wavelet / tddr+wavelet, je baseline und motion, 180 s, 3 Seeds, gemeinsame Kanalbasis; HRF-Retention je Verfahren | `preprocessing_comparison.csv`, `hrf_retention.csv`, Abb. 11 |
| `imageglm` | 12 Familien x baseline/global, 368 s, Seed 0, AR-IRLS über alle Kanäle (420 s je Fit), 3 Projektionen plus truth_ref | `imageglm_summary.csv`, Abb. 19-21 |
| `recon_check` | rauschfreier Kreis Kortex -> Kanal -> Kortex für alpha_spatial aus / 0,001 / 0,01 | Konsolenausgabe |

### 4.3 Realdaten

| Modul | Raster | Ausgabe |
|---|---|---|
| `realglm` (Khan) | 14 Familien x baseline/global x AR-IRLS/OLS über 69 Aufnahmen = 3864 Fits; Gruppen-t-Test über 25 Probanden, FDR, Reproduzierbarkeit über 74 Durchgangspaare | `realglm_summary.csv`, `realglm_group_map.nc`, Abb. 15-18 |
| `msglm` (Multisubject) | 12 Familien x 6 Systemik-Stufen mit OLS (72 Zellen); AR-IRLS für none/poly:3/dct:0.02/butter:0.01 x none/short_avg_dm/short_avg_sub (12 Zellen, 9 gerechnet); je Zelle Vollfit plus zwei Trial-Hälften, 20 lange Kanäle; Lateralisierungsindex und Bildraum-Lokalisation | `msglm_summary.csv`, Abb. 22, 23 |
| `mshrf` (Multisubject) | 12 Familien x 5 Probanden, OLS, short_avg in der Designmatrix, flexible HRF-Basis; ROI je Hand die 3 stärksten kontralateralen Kanäle aus dem Block-Mittel der Daten | `mshrf_summary.csv`, Abb. 29, 30 |

## 5 Gütemaße

- Simulation, Kanalraum: Bias, Varianz und RMSE von beta_hat gegen das wahre beta je
  Kanal über die Seeds, danach Median über die 20 Kanäle (`bias_med`, `var_med`,
  `rmse_med`). Plausibilität: Korrelation der HbO- und HbR-Karten über Kanäle
  (`hbo_hbr_corr`) und rückgewonnenes Verhältnis HbR/HbO (`hbr_hbo_ratio_med`,
  eingemischt -0,4).
- Modellfit ohne Wahrheit: R², adj. R² (Strafterm für die Spaltenzahl der Designmatrix),
  Residual-RMS in µM, Anteil der Residualleistung unter 0,02 Hz (`lowfreq_frac`). Für
  Filter-Arme bezieht sich R² auf die gefilterte Zeitreihe und ist mit den
  Regressor-Familien nicht direkt vergleichbar. `resid_err_corr` ist die Korrelation
  zwischen Residual-RMS und |beta_hat - beta| über Seeds x Kanäle einer Zelle.
- Form: Pearson-Korrelation zwischen rückgewonnener und eingemischter HRF-Kurve
  (skaleninvariant) und relativer Amplitudenfehler am wahren Peak.
- Detektion: Sensitivität, Spezifität, Präzision und Youden-J der nach FDR signifikanten
  Kanäle gegen die wahre Aktivmaske.
- Bildraum: Korrelation zwischen rekonstruierter Karte und Wahrheit innerhalb der Maske
  (`img_r`), Abstand des rekonstruierten Maximums zum nächsten wahren Zentrum
  (`img_loc_err_mm`), Anteil der rekonstruierten Masse innerhalb von 30 mm um ein wahres
  Zentrum (`img_hit_frac`), Verhältnis der Peaks (`img_peak_ratio`).
- Realdaten: Reproduzierbarkeit als Median-Korrelation der beta-Karten zwischen
  Durchgängen eines Probanden (Khan) bzw. zwischen geraden und ungeraden Trials
  (Multisubject); Zahl signifikanter Kanäle nach FDR; HbR/HbO-Verhältnis und
  -Korrelation; Lateralisierungsindex als Differenz des mittleren beta kontralateral
  minus ipsilateral (positiv = Erwartung erfüllt); Abstand des rekonstruierten Maximums
  zur erwarteten Landmarke (C3 für die rechte, C4 für die linke Hand).

## 6 Ergebnisse

Alle Zahlen stammen aus dem Lauf vom 8./9. September 2026 (Bildraum-Ground-Truth nach
dem Fix der Kanalreihenfolge), sofern nichts anderes angegeben ist. Vollständige Tabellen:
`results/tables.md` (Sweep, adj. R², Residualkorrelation) und die CSV-Dateien in
`results/`. Die Abschnitte „Flexible Recovery-Basis" und „Detektion" in `tables.md`
stammen noch aus einem älteren Lauf; maßgeblich sind `flex_basis_summary.csv` und
`detection_summary.csv`.

### 6.1 Kanalraum-Sweep (Abb. 6-10, 25, 26)

Fensterlänge (baseline, wavelet, Median über die 15 Familien):

| | 90 s | 180 s | 368 s |
|---|---|---|---|
| HbO RMSE [µM] | 0,367 | 0,193 | 0,084 |
| HbO Bias [µM] | +0,106 | +0,084 | -0,026 |
| HbR RMSE [µM] | 0,061 | 0,041 | 0,031 |
| HbR Bias [µM] | +0,018 | -0,028 | -0,016 |
| HbR/HbO-Verhältnis | -0,30 | -0,39 | -0,46 |
| corr(HbO, HbR) | -0,83 | -0,93 | -0,96 |

Konstellation (wavelet, Mittel über Familien und Fenster):

| Konstellation | HbO RMSE | HbO Bias | HbR RMSE | HbR Bias |
|---|---|---|---|---|
| baseline | 0,221 | +0,051 | 0,044 | -0,012 |
| motion | 0,219 | +0,049 | 0,044 | -0,009 |
| global | 0,076 | -0,017 | 0,049 | -0,015 |
| short_avg | 0,069 | -0,022 | 0,073 | -0,031 |
| short_maxcorr | 0,082 | +0,010 | 0,051 | -0,015 |

Bewegungskorrektur (baseline, Mittel über Familien und Fenster): HbO RMSE 0,221 (wavelet)
gegen 0,133 (tddr+wavelet), HbO Bias +0,051 gegen -0,073; HbR RMSE 0,044 gegen 0,048,
HbR Bias -0,012 gegen +0,034 (bei wahrem beta -0,24). Der Bias ist über die Familien
nahezu konstant; der Unterschied zwischen den Balken in Abb. 10 stammt aus der
Vorverarbeitung. Der niedrigere HbO-RMSE unter TDDR entsteht aus der Dämpfung der
eingemischten HRF (Abschnitt 6.3), die der systemischen Überschätzung entgegenwirkt; bei
HbR, wo keine Überschätzung vorliegt, verschlechtert TDDR den Bias.

Familien: bei 368 s liegen alle Familien für HbO zwischen 0,082 und 0,086 µM
(butter:0.01, legendre:3, poly:3 bei 0,082; none 0,083). Bei 90 s liegen sie zwischen
0,357 (butter:0.01) und 0,406 (poly:5); bspline:8 fällt auf 0,489. Beste Zellen je Fenster
mit systemischem Regressor: dct:0.01 + short_avg 0,080 (90 s), bspline:5 + short_avg 0,052
(180 s), dct:0.02 + short_maxcorr 0,041 (368 s). Für HbR beträgt die Spannweite der
Familien bei 368 s 0,030-0,034 µM.

Modellfit (adj. R², baseline, wavelet, Median über Familien): HbO -0,45 / 0,18 / 0,21 und
HbR 0,15 / 0,45 / 0,55 für 90 / 180 / 368 s. Höchste Werte hat dct:0.02 (HbO 0,03 / 0,35 /
0,38; HbR 0,15 / 0,57 / 0,77). Unter AR-IRLS werden rohe Polynome hoher Ordnung bei kurzen
Fenstern numerisch instabil: poly:5 bei 90 s hat adj. R² -95 und Residual-RMS 5,1 µM,
bspline:8 -6416 und 39,7 µM; der beta-RMSE dieser Zellen bleibt in der Größenordnung der
übrigen (0,406 bzw. 0,489). Legendre-Polynome liefern dieselben Fits wie rohe Polynome,
weil beide denselben Raum aufspannen.

Residuen gegen Ground-Truth-Abweichung (Abb. 26): zwischen den 45 Zellen (baseline,
wavelet) ist die Spearman-Korrelation von Residual-RMS und beta-RMSE für HbO 0,62, für HbR
-0,02. Innerhalb der Zellen liegt `resid_err_corr` für HbO bei 0,06-0,26, für HbR bei
-0,05-0,31.

### 6.2 Form, Detektion, Residualspektren (Abb. 12-14, 27, 28)

Formtreue mit flexibler Recovery-Basis (180 s, baseline, n = 80 je Zelle):

| Familie | HbO Form | HbO Amplitudenfehler | HbR Form | HbR Amplitudenfehler |
|---|---|---|---|---|
| none | 0,791 | -2,6 % | 0,882 | -2,6 % |
| poly:1 | 0,792 | -2,8 % | 0,914 | +5,6 % |
| poly:3 / legendre:3 | 0,791 | -1,7 % | 0,930 | +10,6 % |
| poly:5 | 0,788 | -2,3 % | 0,941 | +13,3 % |
| dct:0.01 | 0,790 | -1,4 % | 0,930 | +10,2 % |
| bspline:5 | 0,784 | -4,5 % | 0,928 | +9,4 % |
| butter:0.01 | 0,796 | -2,0 % | 0,918 | +4,9 % |

Detektion nach FDR (q = 0,05; 180 s; Seeds 0-1; 51 aktive HbO- und 26 aktive HbR-Kanäle
von 519):

| Familie | HbO Sens. | HbO Spez. | HbO Youden | HbR Sens. | HbR Spez. | HbR Youden |
|---|---|---|---|---|---|---|
| none | 0,37 | 0,84 | 0,22 | 0,44 | 0,95 | 0,39 |
| poly:1 | 0,36 | 0,85 | 0,21 | 0,63 | 0,89 | 0,52 |
| poly:3 / legendre:3 | 0,35 | 0,83 | 0,18 | 0,67 | 0,91 | 0,59 |
| dct:0.01 | 0,35 | 0,84 | 0,19 | 0,65 | 0,93 | 0,58 |
| bspline:5 | 0,36 | 0,83 | 0,19 | 0,69 | 0,90 | 0,59 |

Für HbO liegen alle Familien bei Sensitivität 0,35-0,37 und Spezifität 0,83-0,85; für HbR
heben Driftregressoren die Sensitivität von 0,44 (none) auf 0,63-0,69 bei Spezifität
0,89-0,93.

Residualspektren (`residuals_summary.csv`, AR-IRLS, Seed 0): Anteil der Residualleistung
unter 0,02 Hz bei 368 s für HbO 0,22 (none, poly:1-5, legendre:3, bspline:5), 0,195
(dct:0.01), 0,116 (butter:0.01), 0,056 (dct:0.02). Bei 90 s liegt poly:5 bei 0,957
(Residual-RMS 6,1 µM), none bei 0,072.

### 6.3 Vorverarbeitung (Abb. 11)

`hrf_retention.csv` (Lauf vom 10. September 2026, Seeds 0-1, korrigierte Injektion):
Anteil der eingemischten HRF-Amplitude, der die Bewegungskorrektur überlebt: ohne
Korrektur 100,000 %, wavelet 100,000-100,002 %, tddr 51,5-60,0 % (HbO) und 52,8-60,2 %
(HbR), tddr+wavelet identisch mit tddr. TDDR entfernt also 40-49 % der Antwort (nicht
"ein Drittel"; ältere Notizen nannten 56-69 % Retention aus dem Lauf vom 5. August).

`preprocessing_comparison.csv` (Lauf vom 5. August 2026 mit Kanalraum-Blob, 180 s, 3
Seeds, baseline; wahres beta 0,395 / -0,158 µM): Bias HbO +0,199 (ohne Korrektur), +0,249
(wavelet), +0,058 (tddr+wavelet); Bias HbR +0,018, +0,018, +0,071. Die Motion-Regressoren
in der Designmatrix ändern diese Werte um höchstens 0,011.

### 6.4 Bildraum (Abb. 19-21, 24)

Rauschfreie Obergrenze (`truth_ref`, 520 Kanäle, alpha_spatial 0,001): r = 0,80,
Lokalisationsfehler 11,9 mm, 82 % der rekonstruierten Masse innerhalb von 30 mm um ein
wahres Zentrum, Peak-Verhältnis 1,93.

Projektion der geschätzten HRF (368 s, Seed 0, AR-IRLS, 12 Familien):

| | HbO baseline | HbO global | HbR baseline | HbR global |
|---|---|---|---|---|
| img_r | 0,34-0,43 | 0,39-0,43 | 0,21-0,24 | 0,30-0,33 |
| Lokalisationsfehler [mm] | 74-86 | 86 | 78-96 (none: 19) | 76-78 |
| Massenanteil innerhalb 30 mm | 0,17-0,21 | 0,22-0,24 | 0,15-0,17 | 0,18-0,19 |
| Kanalraum r derselben Zellen | 0,91-0,93 | 0,93-0,96 | 0,90-0,92 | 0,92-0,93 |

Das rekonstruierte Maximum liegt in 23 der 24 HbO-Zellen an derselben Stelle (84-86 mm
von den wahren Zentren, unabhängig von der Familie); die Ausnahme ist dct:0.02/baseline mit
74 mm. Die Residuum-Projektion hat r zwischen -0,03 und 0,05, die Projektion
Residuum + HRF zwischen -0,01 und 0,29 bei Lokalisationsfehlern von 61-117 mm. Die
Reihung der Familien im Bildraum ist flach: img_r variiert innerhalb einer Konstellation um
höchstens 0,09.

### 6.5 Khan-Datensatz (Abb. 15-18)

Reproduzierbarkeit (Median-Korrelation der HbO-beta-Karten über 74 Durchgangspaare,
baseline):

| Familie | AR-IRLS | OLS |
|---|---|---|
| dct:0.02 | 0,566 | 0,564 |
| dct:0.01 | 0,545 | 0,547 |
| bspline:8 | 0,535 | 0,535 |
| poly:5 | 0,533 | 0,516 |
| poly:3 / legendre:3 | 0,526 | 0,440 |
| dct:0.005 | 0,522 | 0,498 |
| none | 0,494 | 0,461 |
| butter:0.01 | 0,443 | 0,364 |
| bandpass:0.01-0.5 | n. a. | 0,389 |

Spannweite über die Familien: 0,123 mit AR-IRLS, 0,200 mit OLS. Mit AR-IRLS überstehen bei
`baseline` 27-31 von 48 Kanälen die FDR-Korrektur, mit `global` 10-19. HbO und HbR sind
über Kanäle mit -0,86 bis -0,89 korreliert; das HbR/HbO-Verhältnis liegt bei -0,18
(baseline) und -0,29 (global, AR-IRLS) bzw. -0,36 (global, OLS). Kombinationen aus
Tiefpass 0,5 Hz und AR-IRLS liefern beta um 1e-7 µM und werden nicht in Ranglisten
aufgenommen (Abschnitt 8).

### 6.6 Multisubject-Datensatz (Abb. 22, 23, 29, 30)

`msglm_summary.csv` (OLS, HbO, Tapping-Bedingungen, Mittel über beide Hände):
Halbierungs-Reproduzierbarkeit 0,21-0,42 ohne systemischen Regressor, 0,25-0,52 mit
short_avg in der Designmatrix, 0,33-0,52 mit short_avg-Subtraktion, 0,46-0,58 mit
global_dm. Der Lateralisierungsindex ist für die linke Hand in allen Zellen positiv
(+0,10 bis +0,18 µM), für die rechte Hand nahe null (-0,02 bis +0,10 µM). Im Bildraum liegt
das rekonstruierte Maximum mit global_dm 12-16 mm von der erwarteten Landmarke, in den
übrigen Systemik-Stufen überwiegend 84-90 mm (etwa gleich weit von beiden Landmarken),
in einzelnen Zellen 16-31 mm. Von den 12 AR-IRLS-Zellen sind 9 gerechnet.

`mshrf_summary.csv` (OLS, short_avg, flexible HRF-Basis, Median über 5 Probanden), adj.
R² HbO / HbR: none 0,41 / 0,34; poly:1 0,60 / 0,57; poly:3 0,83 / 0,65; poly:5 0,83 /
0,69; dct:0.005 0,88 / 0,77; dct:0.01 0,93 / 0,79; dct:0.02 0,96 / 0,86; bspline:5 0,83 /
0,68; bspline:8 0,84 / 0,73; butter:0.01 0,26 / 0,07 (gefilterte Zeitreihe). Die
Spaltenzahl der Designmatrix reicht dabei von 32 (none) bis 149 (dct:0.02).

### 6.7 Abbildungen

| Abb. | Datei | Modul |
|---|---|---|
| 00 | `00_pipeline_flowchart.pdf/.png` | `reports/flowchart.py` |
| 1-5 | Designmatrix, Ein-Kanal-Fit, beta_hat gegen beta, Scalp-Abweichung absolut/relativ | `reports/demo_figures.py` |
| 6-10 | RMSE je Familie, Bias/Varianz, Konstellation, Plausibilität, Motion-Achse | `reports/sweep_report.py` |
| 11 | Vorverarbeitungsvarianten | `analysis/compare_preprocessing.py` |
| 12, 13 | Formtreue, HRF-Kurven flexible Basis | `analysis/flex_basis.py` |
| 14 | Detektion nach FDR | `analysis/detection.py` |
| 15-18 | Khan: Reproduzierbarkeit, signifikante Kanäle, Gruppenkarte, Plausibilität | `reports/realglm_report.py` |
| 19-23 | HRF je Kanal, Kortex wahr/rekonstruiert, Familien im Bildraum, Lateralisierung, Global-Component | `reports/imagespace_report.py` |
| 24 | Scalp Ground Truth neben Schätzung | `reports/demo_figures.py` |
| 25, 26 | adj. R², Residuen gegen GT-Abweichung | `reports/sweep_report.py` |
| 27, 28 | Residualspuren und -spektren, HRF je Familie (Simulation) | `analysis/residuals.py` |
| 29, 30 | HRF je Familie und Modellfit (Multisubject) | `analysis/mshrf.py` |

## 7 Dateien und Ausführung

Das Paket `drift_glm/` hat vier Schichten mit Abhängigkeiten nur nach unten: `core`
(Vorverarbeitung, kurze Kanäle, Bildraum, Simulation, Fit-Metriken) -> `data`
(Khan, Multisubject, Koregistrierung) -> `analysis` (Auswertungen, schreiben nach
`results/`) -> `reports` (Abbildungen und Tabellen aus `results/`). `paths.py` hält alle
Pfade, `figstyle.py` die Farben und Reihenfolge der Familien. `tests/` enthält 31
pytest-Tests (`test_smoke.py` 12, `test_imagespace.py` 12, `test_fitstats.py` 7), darunter
den Regressionstest `test_activation_matches_beta_true_map` für die Kanalreihenfolge.

`run_v6.sh` führt alle Schritte sequenziell mit PID-Sperre aus (`demo residuals mshrf
sweep sweeprep flex detection imageglm report msglm tables`); der Sweep schreibt seine CSV
am Ende, `imageglm` und `msglm` nach jeder Zelle, `msglm` setzt fort. Aufrufe, Laufzeiten
und Reproduzierbarkeitshinweise stehen in der [`README`](README.md).

## 8 Bekannte Einschränkungen

- Kanalreihenfolge nach `od2conc`: die Funktion gibt Kanäle nicht in Eingabereihenfolge
  zurück. Eine positionale Zuweisung der Ground-Truth-Karte in `pipeline.build` verteilte
  die Injektion auf falsche Kanäle; alle Bildraum-Ergebnisse zwischen dem 11. August und
  dem 8. September 2026 waren dadurch ungültig. Die Karte wird jetzt nach Kanalnamen
  ausgerichtet (`test_activation_matches_beta_true_map`); alle Simulationsergebnisse in
  Abschnitt 6 stammen aus dem Neulauf.
- TDDR dämpft die eingemischte HRF auf 51-60 % (HbO wie HbR) und entfernt einen großen
  Teil des Driftbands. Deshalb ist `wavelet` Standard und TDDR eine eigene Achse. Der
  kleinere HbO-RMSE unter TDDR entsteht dadurch, dass die Dämpfung die systemische
  Überschätzung kompensiert; der Bias ist deshalb je Chromophor zu lesen.
- Tiefpass 0,5 Hz und AR-IRLS schließen einander bei 3,9 Hz Abtastrate aus: der Filter
  entfernt rund drei Viertel des Spektrums, der Whitening-Filter müsste dort unbegrenzt
  verstärken, beta kollabiert auf etwa 1e-7 µM. `lowpass:*` und `bandpass:*` werden nur mit
  OLS ausgewertet (`realglm.AR_IRLS_INCOMPATIBLE`).
- Der Global-Regressor mittelt über alle Kanäle einschließlich der aktivierten und enthält
  damit einen Teil der gesuchten Antwort. Die kurzen Kanäle von nn22 (15,8-17,8 mm) sind
  keine echten Short-Separation-Kanäle und sehen noch Kortex: der stärkste erhält 18,9 %
  des eingemischten Peaks, im Mittel bleiben 1,4 % (short_avg) gegen 3,8 % (global)
  (`pipeline leakage`, Werte vor dem Neulauf, zu prüfen). Die Grenze 1,8 cm ist
  empfindlich (1,9 cm: 114 statt 37 kurze Kanäle).
- Bildraum-Regularisierung: mit `alpha_spatial = 0,001` überschätzt schon die rauschfreie
  Rekonstruktion den Peak um den Faktor 1,93 bei 11,9 mm Lokalisationsfehler; mit 0,01
  sinkt der Fehler auf 3,0 mm (`recon_check`, zu prüfen). Ohne Sichtbarkeitsmaske landet
  auch die rauschfreie Wahrheit 65-110 mm neben ihrem Zentrum bei r = 0,00 (zu prüfen).
  Alle Bildraum-Zahlen gelten nur innerhalb der Maske und für diese Regularisierung; die
  Schätzungen aus dem GLM lokalisieren in 23 von 24 HbO-Zellen an einer familienunabhängigen
  falschen Stelle. `imageglm` verwendet einen Seed.
- `msglm_summary.csv` enthält 81 von 84 Zellen; dct:0.02 x AR-IRLS (x none / short_avg_dm /
  short_avg_sub) fehlt (Laufzeit etwa 19 h wegen 2,9 GB je Kanalfit, strikt sequenziell).
  `./run_v6.sh msglm tables` rechnet sie nach und überspringt die fertigen Zellen.
- Khan ohne Landmarken und Sensitivitätsmatrix: `geo3d` enthält nur Optoden. Die
  landmarkenfreie Registrierung (`data/coregister.py`) legt die Optoden im Median 2,2 mm auf
  die ICBM152-Kopfhaut, kann aber links und rechts nicht unterscheiden; die
  Sensitivitätsmatrix bräuchte eine Photonensimulation mit CUDA-GPU (nicht vorhanden). Der
  Bildraum und die kontralaterale Kontrolle sind für diesen Datensatz nicht verfügbar; die
  Konstellationen `short_*` und `motion` entfallen (keine kurzen Kanäle, kein Aux).
- Abb. 20 (`20_cortex_truth_vs_recon.png`) enthält nur leere Achsenrahmen: das
  Offscreen-Rendering der Kortexoberfläche (pyvista) liefert unter WSL keine Pixel. Die
  Datei muss auf einem System mit funktionierendem OpenGL neu erzeugt werden
  (`imagespace_report cortex`).
- Simulation auf einer einzelnen Person (nn22), Auswertung über die 20 stärksten Kanäle je
  Zelle; `dct:0.005`/`dct:0.01` bei 90 s und `dct:0.005` bei 180 s sind mit `none`
  identisch und `poly:n` mit `legendre:n`, sodass das 15-Familien-Raster bei kurzen
  Fenstern weniger unabhängige Modelle enthält als Zeilen.
