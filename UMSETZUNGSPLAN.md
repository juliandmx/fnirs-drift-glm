# Umsetzungsplan — Anmerkungen aus dem Betreuungsgespräch

> Quelle: Gesprächsnotizen als WhatsApp-Verlauf, `anmerkungen/chat_1..4.png` + `anmerkungen/pic_1..3.jpg`.
> Erstellt: 2026-08-01. Zeitrahmen: **4 Arbeitstage**, Nachtläufe für Sweeps erlaubt.
> Vorbedingung geklärt: die realen **Tetris-DOT-Daten liegen vor** (Stufe 2 der Arbeit startet).
> Verwandt: [`PROJEKTPLAN.md`](../PROJEKTPLAN.md), [`DOKUMENTATION.md`](DOKUMENTATION.md).

---

## 0. Vollständige Anmerkungsliste

Jede Notiz aus den sieben Bildern, mit Gruppe und Tag. Nichts wird verworfen.

| # | Anmerkung (Quelle) | Gruppe | Tag |
|---|---|---|---|
| A1 | Pipeline: **SNR-Threshold = 3** (chat_1, 11:08) | Preprocessing | 1 |
| A2 | Sehr dunkle und gesättigte Channels rausfiltern (chat_1, 11:13) | Preprocessing | 1 |
| A3 | NinjaNIRS-Grenzen: **upper 0,84 / lower 1e-3** (chat_1, 11:15) | Preprocessing | 1 |
| A4 | `quality.mean_amp` nimmt Zeitreihe + Thresholds (chat_1 11:17, pic_1) | Preprocessing | 1 |
| A5 | **Motion Correction fehlt: TDDR, Wavelet** — dazu gibt es ein Notebook (chat_1, 11:18) | Preprocessing | 1 |
| A6 | Motion-Artefakte = scharfer Spike **oder** ruckartige Verschiebung (chat_1, 11:19) | Preprocessing | 1 |
| A7 | Motion Correction **auf OD** (chat_1, 11:19) | Preprocessing | 1 |
| A8 | Von amp zu OD **auch Baseline ausgeben** (chat_1, 11:20) | Preprocessing | 1 |
| A9 | **Reihenfolge:** erst Motion Correction → zurück zu Amplitude → schlechte Channels markieren → wieder mit OD weiter (chat_1, 11:21) | Preprocessing | 1 |
| B6 | Motion-Daten vorhanden ⇒ evtl. **keine** Motion Correction nötig (sonst gedoppelt) → **überprüfen** (chat_2, 11:32) | Preprocessing | 1→2 |
| B7 | **cedalion neu pullen** (chat_2, 11:34) | Preprocessing | 1 |
| B8 | Normalisierungsfunktion im neuen Update? (chat_2, 11:35) | Preprocessing | 1 |
| B9 | Im Resting State gibt es **`dark signal`** (chat_2, 11:38) | Preprocessing | 1 |
| C1 | Short Channel = Abstand < 1,5 cm; flache „Banane" misst nur Gewebe/Scalp (chat_3 11:48, pic_2) | Short-Channel | 2 |
| C2 | Short Channels **als Regressor in der Designmatrix** (chat_3, 11:49) | Short-Channel | 2 |
| C3 | → damit **Scalp-Aktivität rausregressieren** (chat_3, 11:49) | Short-Channel | 2 |
| C4 | Geht nur, wenn Short Channels im Datensatz vorhanden sind (chat_3, 11:49) | Short-Channel | 2 |
| C5 | Kürzester Abstand im Vorgabedatensatz **15,5 mm** → **1,8 cm als Threshold testen** (chat_3, 11:52) | Short-Channel | 2 |
| D1 | Je Long-Channel den **am stärksten korrelierenden** Short-Channel — oder **Short-Average** (chat_4 11:53, pic_3) | Short-Channel | 2 |
| D2 | **Analyse nur über die Long-Channels** (chat_4, 11:53) | Short-Channel | 2 |
| B10 | Relativer Scalp-Plot: **Limit auf 100 %** setzen (chat_2, 11:41) | Plots | 2 |
| B11 | Abb. 10: **über alle Bilder gleiche Skala** (chat_2, 11:42) | Plots | 2 |
| B12 | **OLS-Rauschmodell** auch mal verwenden (chat_2, 11:43) | Sweep | 3 |
| B13 | Testweise **Hochpass 0,01 / Tiefpass 0,5 Hz**, im Sweep **ganz am Ende**, im **Konzentrationsraum**: entweder Driftregressor **oder** Hochpassfilter (chat_2, 11:45) | Sweep | 3 |
| — | **Reale Tetris-DOT-Daten** einbinden (Stufe 2) | Reale Daten | 3 |
| B1 | Künstliche Aktivierung **nicht** im Image Space, sondern im **Channel Space** einfügen — dort, wo Resting State und Aktivierung zusammengeführt werden (chat_2, 11:25) | Image Space | 4 |
| B2 | Ergebnisse (**pred-HRF & Residual**) in den Image Space überführen und mit der eigenen HRF vergleichen (chat_2, 11:25) | Image Space | 4 |
| B3 | **Trotzdem auch im Channel Space** vergleichen (chat_2, 11:25) | Image Space | 4 |
| B4 | Code für Image Reconstruction **von Elsa** (chat_2, 11:27) | Image Space | 1 (anfragen) / 4 |
| B5 | Bei Image Recon **auf jeden Fall** dunkle und gesättigte Kanäle rausfiltern (chat_2, 11:28) | Image Space | 1 + 4 |
| D3 | Vergleich: synthetische Aktivierung **auf dem Kortex** visualisieren, und die Ergebnisse im Image Space ebenfalls auf dem Kortex (chat_4, 12:01) | Image Space | 4 |

---

## 1. Vorab-Recherche: was schon geklärt ist

Damit während der Umsetzung nichts doppelt recherchiert wird. Alles gegen cedalion `dev` v26.5.1
verifiziert.

### 1.1 Preprocessing-Reihenfolge — die Vorgabe deckt sich exakt mit Cedalion

`examples/tutorial/3_signal_processing.ipynb` nennt als Lernziel:
*„quality assessment → OD conversion → motion correction → filtering → haemoglobin concentration"*,
und Zelle 38 sagt explizit: *„The correction algorithms operate on optical densities. After
correction, the corresponding corrected amplitudes are derived."* Das ist **genau A7 + A9**.

Praktisch umgesetzt in Notebook 3 (Zellen 28/39/41/59) und `25_intro_quality_workshop.ipynb`:

1. Metriken auf der **Rohamplitude**
2. `od = int2od(amp)`
3. **Motion Correction auf OD**: `tddr` zuerst (Sprünge), dann `wavelet` (Spikes) — Begründung in
   NB 25: *„apply TDDR first to correct jumps, then apply Wavelet motion artifact correction"* (**A5, A6**)
4. **Zurück zur Amplitude**: `amp_corr = od2int(od_corrected, baseline)` (**A8, A9**)
5. Metriken auf `amp_corr` neu berechnen
6. **Pruning ganz am Schluss** („Final channel selection")
7. Weiter auf OD → `od2conc`

### 1.2 Baseline-Rückgabe (A8) — existiert bereits

`src/cedalion/nirs/cw.py:16` — der Parameter heißt **`return_baseline`**, nicht `baseline=`:

```python
od, baseline = cedalion.nirs.cw.int2od(rec["amp"], return_baseline=True)
# baseline = amplitudes.mean("time"), dims (channel, wavelength)
amp_corr  = cedalion.nirs.cw.od2int(od_corrected, baseline)   # = baseline * exp(-od)
```

Die Notebooks schreiben äquivalent `od2int(od_wavelet, rec["amp"].mean("time"))`.

### 1.3 Qualitätsmetriken (A1–A4)

| Funktion | Signatur | Quelle |
|---|---|---|
| `quality.snr(amplitudes, snr_thresh=2.0)` | → `(snr, mask)`; `snr = mean/std` | `quality.py:539` |
| `quality.mean_amp(amplitudes, amp_range)` | → `(mean_amp, mask)`; **kein Default** | `quality.py:569` |
| `quality.sd_dist(amplitudes, geo3D, sd_range=(0,4.5) cm)` | → `(sd_dist, mask)` | `quality.py:599` |
| `quality.prune_ch(amplitudes, masks: list, operator, flag_drop=True)` | → `(gepruned, drop_list)` | `quality.py:30` |

Konvention: `CLEAN = True`, `TAINTED = False`. `operator="all"` = alle Masken mit `&` verknüpfen.
**`xrutils.combine_masks` existiert nicht** — Masken mit `&` kombinieren oder `prune_ch` nutzen.

**Auf nn22 gemessen** (567 Kanäle, 8,99 Hz, 368,1 s):

- `mean_amp` mit `(1e-3, 0.84) V`: 64 von 1134 (Kanal × Wellenlänge) außerhalb → **46 Kanäle fallen raus**
- `snr`-Threshold 3 → 557 Kanäle bleiben (bisher genutzter Wert 10 → 544). Der neue Wert ist
  **permissiver**; die eigentliche Arbeit macht jetzt `mean_amp`. Das ist konsistent: SNR ≙ Rauschanteil,
  `mean_amp` ≙ dunkel/gesättigt — zwei verschiedene Defekte.

### 1.4 `dark signal` (B9) — bestätigt vorhanden

`rec.aux_ts` von `get_nn22_resting_state()` enthält:
`['digital', 'analog-1', 'analog-2', 'dark signal', 'temparature', 'gyroscope', 'accelerometer']`
— Schreibweise mit **Leerzeichen**, nicht `dark_signal`. Damit lässt sich der Rauschboden des
Detektors bestimmen und die „dunkel"-Grenze datengetrieben statt per Faustwert (1e-3) belegen.

### 1.5 Normalisierungsfunktion im neuen Update (B8) — **Antwort: nein**

Repo-weite Suche: es gibt in v26.5.1 **keine** neue allgemeine Normalisierungsfunktion für Zeitreihen.
Was existiert:

- `motion.normalize_signal(signal, wavelet='db2')` (`motion.py:695`) — reiner **Wavelet-Helfer**,
  Homer3-`NormalizationNoise`: skaliert ein 1D-numpy-Array auf sein Rausch-Sigma (MAD).
  Kein Baseline-Abzug, kein z-Score. Einziger Aufrufer: `wavelet()` selbst.
- `int2od(..., return_baseline=True)` (`cw.py:16`) — existiert seit Mai 2025, **nicht** neu.
- z-Score gibt es nur inline in `sci`/`psp` und in Beispiel-Notebooks, nicht als exportierte Funktion.

⇒ Der Punkt ist damit erledigt; im Gespräch war vermutlich `return_baseline` / `od2int` gemeint.

### 1.6 cedalion-Update (B7)

Lokal `c7cbd53`, `origin/dev` ist **3 Commits weiter**: `e1c6e45` (Garbage Collection in einem
Notebook), `07442b4` (Referenz in einem Notebook), `a7eb625` (Doku/Zitation). **Kein API-Bruch** —
das Update ist unkritisch, sollte aber trotzdem gemacht werden (und CRLF-Falle beachten).

### 1.7 Short-Channel-Regression (C1–C5, D1, D2) — vollständig nativ

`cedalion.nirs.split_long_short_channels(ts, geo3d, distance_threshold=1.5*units.cm)`
(`nirs/common.py:105`) → gibt **(long, short)** zurück, unit-aware.

Drei fertige Regressor-Funktionen in `models/glm/design_matrix.py`:

| Zeile | Funktion | entspricht Notiz |
|---|---|---|
| 621 | `closest_short_channel_regressor(ts_long, ts_short, geo3d)` | räumlich nächster |
| 661 | `max_corr_short_channel_regressor(ts_long, ts_short)` | **„am meisten korreliert"** (D1) |
| 708 | `average_short_channel_regressor(ts_short)` | **„short avg"** (D1) |
| 730 | `global_mean_regressor(ts)` | bisheriges Surrogat |

`closest_*` und `max_corr_*` liefern `channel_wise`-Designmatrizen (je Kanal ein eigener Regressor
namens `"short"`), `average_*` liefert einen gemeinsamen `common`-Regressor.

**Distanzverteilung nn22 (gemessen):** min 15,55 mm · Median 28,78 mm · max 42,29 mm.
Mit Threshold **1,8 cm → 39 „Short"-Kanäle**; mit 1,6 cm nur 5; mit 2,0 cm bereits 230.
Der Vorschlag 1,8 cm ist damit der einzige Wert, der eine brauchbare, aber noch klar
abgegrenzte Short-Gruppe liefert. ⇒ **C5 ist auf nn22 direkt umsetzbar.**

> **Präzedenzfall im Framework:** `36_glm_workshop.ipynb` macht auf `fingertappingDOT` genau
> dasselbe mit 22,5 mm und schreibt dazu: *„The montage has longer (3-3.5cm) and shorter
> (~1.7-2.2cm) distance channels. Define a cut-off at 22.5 mm"*. Das sind ebenfalls keine echten
> Short-Separation-Kanäle (< 10 mm), sondern kurze Distanzkanäle — dieselbe Situation wie hier.
> **In der Arbeit als Limitation benennen:** 15,5–18 mm sehen noch etwas Kortex, der Regressor
> entfernt also potenziell auch echtes Signal.

### 1.8 Image Reconstruction (B2, B4, B5, D3)

**Wichtig: `cedalion.imagereco` heißt in v26.5.1 `cedalion.dot`.** `pseudo_inverse_stacked` gibt es
nicht mehr, ersetzt durch die Klasse `ImageRecon`. Das alte Notebook
`examples/misc/finger_tapping_full_pipeline.ipynb` ist veraltet — **nicht** als Vorlage nutzen.

```python
import cedalion.data, cedalion.dot as dot

# Sensitivitätsmatrix für GENAU unseren Ruhedatensatz — fertig vorberechnet, keine Fluence nötig
Adot = cedalion.data.get_precomputed_sensitivity("nn22_resting", "colin27")

recon = dot.ImageRecon(Adot, recon_mode="mua2conc", alpha_meas=0.01, brain_only=False)
img   = recon.reconstruct(od_like)      # (channel, wavelength, ...) -> (chromo, vertex, ...)
```

- `reconstruct` verträgt beliebige Zusatzdimensionen **und auch β-Bilder ohne Zeitachse**
  (in `47_image_reconstruction_regularizations.ipynb` explizit vorgeführt).
- `y` darf eine **Teilmenge** der Adot-Kanäle sein, aber keine unbekannten enthalten
  (`image_recon.py:1479`, Kommentar: *„y may contain less channels then W due to pruning"*).
  ⇒ **Pruning vor der Rekonstruktion ist der vorgesehene Weg** (= B5).
- Gegenrichtung: `dot.forward_model.image_to_channel_space(Adot, img, spectrum="prahl")`
  (`forward_model.py:972`), vorher `Adot.sel(vertex=Adot.is_brain, channel=od.channel)`.
- Kortex-Visualisierung (D3): `cedalion.vis.anatomy.image_recon_view` /
  `image_recon_multi_view` (erwartet Dims `("vertex","chromo","time")`) oder
  `plot_brain_in_axes` für Matplotlib-Figuren.

**Warum dunkle/gesättigte Kanäle gerade hier kritisch sind (Antwort auf die offene Frage in chat_1):**
Im Kanalraum verfälscht ein defekter Kanal nur sich selbst. In der Rekonstruktion geht die
Kanalvarianz als Gewicht in die Pseudoinverse ein — `W = λ_R·A·inv(λ_R·F + λ_meas·C_meas)`
(`image_recon.py:1377`). Ein **gesättigter** Kanal ist geklippt und dadurch künstlich rauscharm,
bekommt also **maximales** Gewicht; ein **dunkler** Kanal hat in `int2od` extremes Rauschen.
Da jede Zeile von `W` über *alle* Vertices verteilt, wirkt sich der Fehler nicht lokal aus,
sondern schmiert über das gesamte Sensitivitätsprofil dieses Kanals.
Cedalion sagt das explizit in `24_downweighting_noisy_channels.ipynb`:
*„the metric cannot account for saturation, and we should manually drop / downweight the
saturated channel."*

**Was Cedalion an Image Recon mitbringt** (Stand v26.5.1) — der Algorithmus ist vollständig da:

| Datei | Inhalt |
|---|---|
| `src/cedalion/dot/image_recon.py` | `ImageRecon` (Pseudoinverse `W`, Tikhonov, Tiefen-Prior), `GaussianSpatialBasisFunctions`, Regularisierungs-Presets `REG_*`/`SBF_*`, `estimate_alpha_meas` |
| `src/cedalion/dot/forward_model.py` | `ForwardModel` (Fluence → `compute_sensitivity` → Adot), `image_to_channel_space`, `parcel_sensitivity` |
| `src/cedalion/dot/head_model.py` | `TwoSurfaceHeadModel`, `get_standard_headmodel("colin27"\|"icbm152")` |
| Notebooks | `head_models/40_image_reconstruction`, `head_models/47_image_reconstruction_regularizations`, `tutorial/5_image_reconstruction`, `machine_learning/50b_advanced_finger_tapping_lda_classification` |

**Der einzige montageabhängige Teil ist die Sensitivitätsmatrix Adot** — die hängt an den
Optodenpositionen, nicht am Algorithmus. Vorberechnet verfügbar (`data/__init__.py:69-84`):

```
sensitivity_nn22_resting_{colin27,icbm152}.nc          <- unser Ruhedatensatz
sensitivity_ninja_cap_56x144_{colin27,icbm152}.nc      <- die generische NinjaNIRS-Kappe
sensitivity_ninja_uhd_cap_164x496_{colin27,icbm152}.nc <- Ultra-HD-Kappe
sensitivity_fingertapping{,DOT}_{colin27,icbm152}.nc
sensitivity_{lumo,kernel,artinis}_testdataset_colin27.nc
```

`ninja_cap_56x144` ist **genau die Kappe des nn22-Datensatzes** (56 Quellen × 144 Detektoren →
567 Kanäle). ⇒ Wurden die Tetris-Daten mit derselben NinjaNIRS-Kappe aufgenommen, ist auch dort
alles vorhanden und **es wird kein externer Code gebraucht**.

Selbst rechnen wäre der Rückfallweg und ist teuer: `compute_fluence_mcx` simuliert eine
MCX-Rechnung **pro Optode** (`nphoton=1e8`, CUDA) — bei 200 Optoden also hunderte GPU-Läufe;
Referenzskript `examples/head_models/46_precompute_fluence.ipynb`.

⇒ **Der Image-Space-Teil ist nicht blockiert.** Bei Elsa geht es nur noch darum, welche Montage/
Koregistrierung und welche Regularisierungsparameter sie für Tetris verwendet — damit die
Ergebnisse vergleichbar sind, nicht damit sie überhaupt entstehen.

---

## 2. Der Plan — 4 Tage

Konvention: jeder nummerierte Schritt = **ein Commit**. Branch bleibt `main`.
Commit-Messages stehen jeweils in Backticks; committen macht der Autor selbst.

---

### TAG 1 — Preprocessing-Kette nach Betreuungsvorgabe

*Deckt A1–A9, B6 (Vorbereitung), B7, B8, B9, B5 (Kanalseite).*
Das ist die Grundlage: alle bisherigen v3-Zahlen entstanden mit der alten Kette und werden dadurch
ersetzt.

| # | Schritt | Commit-Message |
|---|---|---|
| 1.0 | Arbeitsbaum aufräumen (2 offene Änderungen; `pipeline.py` ist nur ein Leerzeichen → verwerfen) | `Doku: Hinweis auf spaetere DOT-Daten praezisiert` |
| 1.1 | cedalion auf `origin/dev` ziehen, Smoke-Tests, `environment.lock.txt` aktualisieren | `Umgebung: cedalion-Clone aktualisiert und Versionsstand festgehalten` |
| 1.2 | Neues Modul `preprocess.py`; `int2od(..., return_baseline=True)` | `preprocess: Modul angelegt, OD-Umrechnung mit Baseline-Rueckgabe` |
| 1.3 | Motion Correction auf OD: TDDR → Wavelet, über Parameter abschaltbar (`none`/`tddr`/`wavelet`/`tddr+wavelet`) | `preprocess: Motion Correction (TDDR, Wavelet) auf Optical Density` |
| 1.4 | Rückrechnung zur korrigierten Amplitude via `od2int(od, baseline)` | `preprocess: korrigierte Amplitude aus OD und Baseline rekonstruiert` |
| 1.5 | Qualitätsmasken auf der **korrigierten** Amplitude: `snr(3)`, `mean_amp((1e-3, 0.84) V)`, `sd_dist` | `preprocess: Kanalmasken SNR=3, mean_amp 1e-3..0.84 V, SD-Distanz` |
| 1.6 | `prune_ch(..., "all")`, dann **auf OD weiterarbeiten** → `od2conc` | `preprocess: Pruning nach der Korrektur, Weiterverarbeitung auf OD` |
| 1.7 | `pipeline.build()` auf `preprocess.py` umstellen, alte Inline-Kette entfernen | `pipeline: nutzt die neue Preprocessing-Kette` |
| 1.8 | `dark signal`-Aux auswerten: Rauschboden je Detektor, Vergleich mit der 1e-3-Grenze | `preprocess: dark-signal-Aux als Rauschboden-Referenz ausgewertet` |
| 1.9 | Vergleichslauf alt/neu (`demo_recovery.py`), Zahlen dokumentieren | `Doku: Effekt der neuen Preprocessing-Kette auf die Beta-Rueckgewinnung` |
| 1.10 | Smoke-Tests auf die neue Kette anpassen | `tests: Smoke-Tests auf die neue Preprocessing-Kette` |

**Nicht-Code-Aufgabe (früh am Tag):** Elsa fragen, **welche Montage/Koregistrierung und welche
Regularisierung** sie für die Tetris-Daten verwendet (B4). Der Image-Recon-Code selbst wird nicht
gebraucht — er steckt vollständig in `cedalion.dot`, und die Sensitivitätsmatrizen für nn22 wie für
die generische NinjaNIRS-Kappe sind vorberechnet (Abschnitt 1.8). Es geht nur um Vergleichbarkeit
mit ihrem Setup.

**Nachtlauf 1:** v3-Raster (13 Familien × {90, 180, 368 s} × 4 Konstellationen × 4 Seeds) mit der
**neuen** Preprocessing-Kette, einmal **mit** und einmal **ohne** Motion Correction.
Das liefert am Morgen von Tag 2 direkt die Antwort auf **B6** (Doppelung Motion-Correction ↔
Motion-Regressoren) und die Referenz, wie stark das neue Preprocessing die v3-Zahlen verschiebt.

---

### TAG 2 — Short-Channel-Regression + Plot-Korrekturen

*Deckt C1–C5, D1, D2, B10, B11 und die Auswertung von B6.*

| # | Schritt | Commit-Message |
|---|---|---|
| 2.1 | Distanz-Histogramm nn22, Split bei 1,8 cm dokumentiert (39 Short / 528 Long) | `short-channel: Distanzanalyse und Long-Short-Split bei 1,8 cm` |
| 2.2 | **Injektion nur in Long-Channels** — der Blob darf die Short-Kanäle nicht treffen, sonst saugt der SC-Regressor die HRF weg (dasselbe Artefakt wie beim Global-Mean in Sweep v1) | `pipeline: synthetische HRF nur in Long-Channels injizieren` |
| 2.3 | Neue Sweep-Konstellationen `short_avg`, `short_maxcorr`, `short_closest`; `global` bleibt als Vergleich erhalten | `sweep: Short-Channel-Regressoren als eigene Konstellationen` |
| 2.4 | Auswertung/Fit-Subset auf Long-Channels beschränken | `sweep: Auswertung auf die Long-Channels beschraenkt` |
| 2.5 | Abb. 05: Farbgrenzen des relativen Scalp-Plots hart auf ±100 % (statt Perzentil) | `figures: relativen Scalp-Plot auf plus/minus 100 Prozent begrenzt` |
| 2.6 | Abb. 10: gemeinsame y-Skala über alle Teilbilder | `figures: gemeinsame Skala ueber alle Teilbilder des Familienvergleichs` |
| 2.7 | Nachtlauf 1 auswerten: Motion Correction vs. Motion-Regressoren — doppelt oder komplementär? | `Doku: Motion-Correction gegen Motion-Regressoren geprueft` |

**Nachtlauf 2:** v4-Hauptsweep — 15 Familien × 3 Fenster × {baseline, motion, global, short_avg,
short_maxcorr} × Seeds, AR-IRLS. Grobschätzung aus v3 (720 Fits ≈ 3 h): rund **8–10 h**, passt in
eine Nacht.

---

### TAG 3 — Reale Tetris-DOT-Daten + Sweep-Erweiterungen

*Deckt die neuen Daten (Stufe 2), B12, B13.*

> **Hier brauche ich die Tetris-Daten.** Wenn du sie mir schon an Tag 1 gibst, ziehe ich das
> Inventar (3.1) vorweg — das kostet nichts und entschärft Überraschungen beim Format.

| # | Schritt | Commit-Message |
|---|---|---|
| 3.1 | Daten einlesen, Inventar: Montage, Kanalzahl, Distanzverteilung, Wellenlängen, aux, Events (Tetris/Ruhe) | `daten: Loader und Inventar fuer die realen Tetris-DOT-Daten` |
| 3.2 | Preprocessing-Kette anwenden; **Amplitudengrenzen prüfen** — 0,84/1e-3 sind NinjaNIRS-spezifisch und gelten evtl. nicht für dieses Gerät | `daten: Preprocessing-Kette auf die Tetris-Daten angewandt` |
| 3.3 | Short-Channels der realen Montage bestimmen, Threshold begründen | `daten: Short-Channel-Erkennung auf der Tetris-Montage` |
| 3.4 | GLM Tetris vs. Ruhe je Driftfamilie; Signifikanz + Benjamini-Hochberg-FDR über alle Kanäle (`detection.py` wiederverwenden) | `real: GLM Tetris gegen Ruhe mit FDR-Korrektur ueber alle Kanaele` |
| 3.5 | Sweep-Achse `noise_model` ∈ {`ar_irls`, `ols`} | `sweep: OLS als zweites Rauschmodell aufgenommen` |
| 3.6 | Filter-Arm **am Ende** des Sweeps: Hochpass 0,01 Hz / Tiefpass 0,5 Hz im **Konzentrationsraum**, als *Alternative* zu Driftregressoren (nicht zusätzlich) | `sweep: Filter-Arm mit Hochpass 0,01 Hz und Tiefpass 0,5 Hz` |

**Nachtlauf 3:** Driftfamilien-Sweep auf den realen Tetris-Daten (volles Raster) + der neue
Filter-/OLS-Arm auf den Simulationsdaten.

---

### TAG 4 — Image Space + Abschluss

*Deckt B1, B2, B3, B4, B5, D3.*

| # | Schritt | Commit-Message |
|---|---|---|
| 4.1 | `cedalion.dot` einbinden: Adot (`get_precomputed_sensitivity("nn22_resting","colin27")`), Kopfmodell, `ImageRecon` | `imagespace: Sensitivitaetsmatrix und Kopfmodell eingebunden` |
| 4.2 | Kanalraum-Injektion belegen und begründen (B1) — die Pipeline macht das bereits so; Nachweis + Doku statt Umbau | `Doku: Injektion im Kanalraum begruendet und belegt` |
| 4.3 | **pred-HRF + Residuum** rekonstruieren: HRF-Anteil aus dem GLM plus Residuum, in den Bildraum überführen | `imagespace: HRF-Anteil und Residuum in den Bildraum ueberfuehrt` |
| 4.4 | Vergleich mit der injizierten HRF — **im Bildraum und im Kanalraum** (B2 + B3), gleiche Metriken (RMSE/Bias/Form) | `imagespace: Vergleich rekonstruierte gegen injizierte HRF` |
| 4.5 | Kortex-Visualisierung: injizierte Aktivierung und geschätztes Ergebnis nebeneinander (D3) | `figures: Kortexdarstellung der injizierten und der geschaetzten Aktivierung` |
| 4.6 | `sweep_report.py` auf v4 aktualisieren, `results/tables.md` neu erzeugen | `report: Abbildungen und Ergebnistabellen auf v4 aktualisiert` |
| 4.7 | `DOKUMENTATION.md` Kapitel 5/6/9 mit v4-Zahlen und den eingearbeiteten Anmerkungen | `Doku: Kapitel 5/6/9 mit v4-Ergebnissen und Betreuungsanmerkungen` |

---

## 3. Entscheidungen, die ich getroffen habe

Ohne Rückfrage, mit Begründung — bitte widersprechen, falls anders gewünscht.

1. **`global_mean_regressor` bleibt im Sweep**, zusätzlich zur echten Short-Channel-Regression.
   Der v3-Kernbefund („der systemische Regressor halbiert den HbO-RMSE") wird dadurch direkt
   gegen die echte SC-Regression vergleichbar — das ist ein eigenständiges Ergebnis, kein Ballast.
2. **Die synthetische HRF wird nicht mehr in die Short-Channels injiziert** (Schritt 2.2).
   Sonst enthält der SC-Regressor die HRF und regressiert sie weg — exakt das Artefakt, das in
   Sweep v1 beim Global-Mean auftrat (siehe `DOKUMENTATION.md`, v1→v2).
3. **Motion Correction ist eine vollwertige Sweep-Achse** (Entscheidung des Autors, 2026-08-01),
   Stufen `wavelet` und `tddr+wavelet`. Auslöser ist ein Messergebnis aus Schritt 1.3:

   | Band | TDDR | Wavelet | TDDR+Wavelet |
   |---|---|---|---|
   | **Drift < 0,01 Hz** | **55,6 %** | 100,0 % | **55,5 %** |
   | 0,01–0,1 Hz | 54,3 % | 99,9 % | 54,3 % |
   | 0,1–0,5 Hz | 49,7 % | 96,8 % | 49,0 % |
   | Kardial > 0,5 Hz | 98,3 % | 85,4 % | 83,4 % |

   (Restleistung nach der Korrektur, Median über Kanal × Wellenlänge; reproduzierbar über
   `preprocess.band_power_ratio`.)

   **TDDR dämpft das Driftband auf 55,6 %** — es entfernt fast die Hälfte dessen, was die
   Driftregressoren modellieren sollen, und wirkt unterhalb 0,5 Hz wie ein breitbandiger Dämpfer.
   Das kollidiert mit dem Betreuungshinweis vom 2026-07-11 (bei Drift-Modellierung nicht
   hochpassfiltern). Wavelet ist im Driftband dagegen exakt neutral. Weil beide Vorgaben von der
   Betreuung stammen, wird nicht stillschweigend eine gewählt, sondern die Achse mitgeführt und
   der Effekt beziffert. Gekreuzt mit der Konstellations-Achse (`motion` = Motion-Regressoren in
   der Designmatrix) beantwortet das zugleich **B6** (Doppelung Korrektur ↔ Regressoren).

   **Nachtrag (Schritt 1.9, nach dem Umbau der Einmischung):** TDDR dämpft nicht nur den
   Drift, sondern die **eingemischte HRF selbst auf 70 %** (Wavelet: 100 %). Der Effekt auf
   die Schätzung (3 Seeds, 180 s, 20 aktivste Kanäle):

   | | Bias HbO | RMSE HbO | Bias HbR | RMSE HbR |
   |---|---|---|---|---|
   | ohne Motion Correction | +0,199 | 0,298 | +0,018 | 0,100 |
   | wavelet | +0,249 | 0,384 | +0,018 | 0,106 |
   | tddr+wavelet | +0,059 | 0,192 | +0,071 | 0,103 |

   (Wahrheit HbO +0,397 / HbR −0,159.) TDDR sieht bei HbO am besten aus — aber nur, weil die
   30 % Dämpfung die **50-%-Überschätzung** durch systemische Physiologie zufällig kompensiert.
   Dass es kein besseres Schätzen ist, zeigt HbR: dort fehlt der Gegenfehler, und TDDR
   verschlechtert den Bias von +0,018 auf +0,071. ⇒ Beim Auswerten des Sweeps **Bias getrennt
   nach Chromophor** betrachten, nicht nur RMSE — sonst wird diese Kompensation als Erfolg
   fehlgedeutet. Auffällig außerdem: **Wavelet allein ist schlechter als gar keine Korrektur**
   (Streuung 0,262 vs 0,161) — im Sweep im Auge behalten.

   Kostenrechnung: v3 waren 720 Fits in 3,03 h (15,2 s/Fit). v4 mit 5 Konstellationen sind
   900 Fits ≈ 3,8 h; mit der Motion-Achse ×2 also **1800 Fits ≈ 7,6 h**. Der Umbau der
   Einmischung kostet zusätzlich zwei Motion-Correction-Läufe pro Build (≈ +45 min), weil das
   Preprocessing jetzt am Seed hängt. Zusammen **≈ 8,4 h** — passt in eine Nacht. Eine dritte
   Stufe `none` käme auf ≈ 12,5 h.
4. **Der Butterworth-Arm bleibt Alternative, nicht Ergänzung** — konsistent zum Betreuungshinweis
   vom 2026-07-11 und zu B13 („entweder Driftregressor oder Highpassfilter").
5. **Der Tiefpass 0,5 Hz** wird nur im Filter-Arm getestet, nicht in den Driftregressor-Armen —
   sonst dämpft er den Drift, den die Regressoren modellieren sollen.

## 4. Risiken

- **TDDR und Wavelet sind Python-Schleifen** über Kanal × Wellenlänge (`motion.py:543`, `:769`),
  Wavelet zusätzlich mit Zero-Padding auf Zweierpotenzen. Bei 567 Kanälen ist das potenziell teuer.
  Das Notebook 22 reduziert dafür auf 500 s. **Zu Beginn von Tag 1 messen**; falls zu langsam,
  einmal vorverarbeiten und das Ergebnis cachen (die Motion Correction hängt nicht vom Seed ab).
- **Tetris-Datenformat** — Inventar (3.1) möglichst vorziehen. Kernfrage für Tag 4: **wurde mit der
  NinjaNIRS-Kappe 56×144 aufgenommen?** Wenn ja, ist die Adot vorberechnet und der Bildraum für die
  realen Daten kostet keine Extraarbeit. Wenn nein, läuft Tag 4 auf nn22 (Adot sicher vorhanden) und
  der reale Bildraum wird nachgezogen, sobald die passende Adot da ist.
- **Kein Blocker durch externen Code**: Image Recon ist vollständig in `cedalion.dot` (Abschnitt 1.8).
- **Der Neu-Sweep entwertet alle v3-Zahlen** in `DOKUMENTATION.md` Kapitel 6. Erst nach Nachtlauf 2
  aktualisieren, sonst wird zweimal geschrieben.
