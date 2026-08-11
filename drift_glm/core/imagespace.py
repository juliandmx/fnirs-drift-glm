"""Bildraum: Kopfmodell, Sensitivitaetsmatrix, Vorwaerts- und Rueckweg.

Dieses Modul ist die einzige Stelle, an der das Kopfmodell und die
Sensitivitaetsmatrix beschafft werden. Alles andere (pipeline.py, imageglm.py,
multisubject.py) greift hier zu, damit Kopfmodell, Regularisierung und
Vertex-Reihenfolge nicht an mehreren Stellen unabhaengig festgelegt werden.

WARUM ES DIESEN RAUM BRAUCHT. Im Kanalraum ist ein Messwert einem Quell-Detektor-Paar
zugeordnet, nicht einem Ort im Gehirn. Solange man nur Driftfamilien vergleicht, reicht
das. Sobald aber die Frage lautet "welche Hirnregion war aktiv?", muss das inverse
Problem geloest werden -- die Zuordnung Kanal -> Kortex ist nicht eindeutig, weil jeder
Kanal ueber ein ganzes Sensitivitaetsprofil integriert.

DIE BEIDEN RICHTUNGEN, und beide werden hier gebraucht:

  Bildraum -> Kanalraum  (`to_channel_space`, Vorwaertsmodell, exakt)
      Damit wird die synthetische Aktivierung erzeugt: ein Blob auf dem Kortex bei
      C3/C4, ueber Adot in Optical Density je Kanal umgerechnet. Erst DANACH kommen die
      Ruhedaten dazu (Betreuungsvorgabe) -- so liegt die Ground Truth dort, wo sie
      physiologisch hingehoert, und die Einmischung passiert dort, wo sie bei echten
      Daten passiert.

  Kanalraum -> Bildraum  (`recon_operator`, inverses Problem, regularisiert)
      Damit wird das GLM-Ergebnis bewertet: die geschaetzte HRF (bzw. das Residuum)
      wird auf den Kortex zurueckprojiziert und dort gegen die Wahrheit gemessen.

Kopfmodell ist durchgaengig **ICBM152** (Betreuungsvorgabe 2026-08-05: "fuer alle
Kopfmodelle ICBM152 - nicht Colin"). Der Unterschied ist nicht kosmetisch: Colin27 ist
ein einzelnes, hochaufgeloest gemitteltes Gehirn, ICBM152 ein nichtlinearer Mittelwert
ueber 152 Probanden und damit der uebliche Bezugsraum fuer Gruppenaussagen.

Aufruf:  conda run -n cedalion python -m drift_glm.core.imagespace [datensatz]
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import xarray as xr

import cedalion
import cedalion.data
import cedalion.dot as dot
import cedalion.nirs
import cedalion.sim.synthetic_hrf as synhrf
from cedalion import units
from cedalion.sigproc.quality import measurement_variance

# Betreuungsvorgabe 2026-08-05: durchgaengig ICBM152, NICHT Colin27.
HEAD_MODEL = "icbm152"

# Regularisierung, Betreuungsvorgabe 2026-08-05 -- weicht bewusst von Notebook 50b ab
# (dort alpha_meas=0.01 fest, alpha_spatial=None, apply_c_meas=False):
#
#   alpha_meas    wird NICHT fest gesetzt, sondern aus den Daten geschaetzt
#                 (`dot.estimate_alpha_meas`, s. `recon_operator`).
#   alpha_spatial 0.001 -- steuert die effektive Tiefe: kleinere Werte unterdruecken
#                 Aktivitaet, die auf der Kopfhaut rekonstruiert wuerde, staerker.
#
# GEMESSEN, rauschfrei ueber `recon_check` auf der 28-Kanal-Montage (bekanntes Muster
# hinein, Guete auf den sichtbaren Vertices heraus):
#
#   alpha_spatial   alpha_meas    r     Peak/wahr   Ort [mm]
#   None            0.001      +0.578     1.51         8.6
#   None            8.86 (est) +0.494     1.36         8.6
#   0.001           0.001      +0.505     2.02        14.6     <- Vorgabe
#   0.001           8.86 (est) +0.450     1.82        14.6     <- Vorgabe + est
#   0.01            0.001      +0.650     0.97        13.2
#   0.01            8.86 (est) +0.590     0.89        14.0
#
# Drei Dinge stehen darin, und keines davon ist trivial:
#
#   1. `alpha_spatial` dominiert. Der Lokalisationsfehler haengt praktisch nur daran
#      (8.6 mm gegen ~14 mm), `alpha_meas` verschiebt ihn kaum.
#   2. **Die Vorgabe 0.001 ist von den drei getesteten Werten der schlechteste**: kleinste
#      Korrelation UND eine um Faktor 2 zu grosse Amplitude. Der Grund ist die Bauart der
#      Tiefenkorrektur: R_j = 1/(sum_i A_ij^2 + alpha_spatial * max), also teilt ein
#      kleiner Wert durch eine kleine Vertexsensitivitaet und blaest schwach gesehene
#      Vertices auf. Mit `brain_only=True` gibt es keine Kopfhaut, die davon profitieren
#      wuerde -- verstaerkt werden nur tiefe Hirnvertices, die die Sonde kaum sieht.
#   3. Cedalions eigener Optimierungs-Preset (`REG_PAPER_MUA_SBF`) benutzt 1e-2, also den
#      hier besten Wert.
#
# Die Vorgabe bleibt trotzdem der DEFAULT -- so wie bei TDDR wird eine Betreuungsvorgabe
# nicht stillschweigend uebersteuert, sondern befolgt, gemessen und zur Entscheidung
# gestellt (siehe BESPRECHUNG.md). Der Parameter ist in `recon_operator` durchgereicht,
# der Gegenlauf kostet also nur ein Argument.
#
# Auf der 520-Kanal-Montage von nn22 ist alles deutlich besser (alpha_spatial=None:
# r = 0.76..0.82, Ort 3.6 mm) -- die Zahlen oben sind die einer bewusst duennen Montage.
ALPHA_SPATIAL = 0.001

#: K in `alpha_meas = K / median(C_meas)`. Cedalion-Default; laut Docstring lag K in
#: frueheren Analysen bei 0.01..0.1.
K_ALPHA_MEAS = 0.01

#: Aktivierungsorte: primaerer Motorkortex links/rechts im 10-20-System
#: (Betreuungsnotiz: "c3, c4 (vor allem halt motorkortex)").
MOTOR_LANDMARKS = ("C3", "C4")

#: Logischer Datensatzname -> Schluessel der vorberechneten Sensitivitaetsmatrix.
#: Fuer die NIRScout-Montage des Khan-Datensatzes gibt es keine -- siehe `coregister.py`.
ADOT_KEYS = {
    "nn22_resting": "nn22_resting",
    "multisubject_fingertapping": "fingertapping",
}


@lru_cache(maxsize=2)
def head(model: str = HEAD_MODEL, ras: bool = True):
    """Kopfmodell (Hirn- und Kopfhautoberflaeche, Landmarken), gecacht.

    `ras=True` transformiert in den RAS/MNI-Raum. Das ist nicht optional, wenn darauf
    ein raeumlicher Blob gebaut wird: die ijk-Oberflaeche ist EINHEITENLOS (Voxel), und
    `build_spatial_activation` rechnet `spatial_scale / surface.units` -- eine Angabe in
    cm laesst sich dort nicht sinnvoll umrechnen. Im RAS-Raum sind die Vertices in mm.

    Die Vertex-REIHENFOLGE ist von der Transformation unberuehrt. Deshalb passt ein auf
    `head(ras=True).brain` gerechneter Blob unveraendert auf die Brain-Vertices der
    Sensitivitaetsmatrix.
    """
    h = dot.get_standard_headmodel(model)
    return h.apply_transform(h.t_ijk2ras) if ras else h


@lru_cache(maxsize=4)
def adot(dataset: str, model: str = HEAD_MODEL) -> xr.DataArray:
    """Vorberechnete Sensitivitaetsmatrix, dims (channel, vertex, wavelength).

    Traegt `is_brain` (die ersten Vertices sind die Hirn-, die letzten die
    Kopfhautvertices) und `parcel` (Schaefer2018) als Vertex-Koordinaten.
    """
    key = ADOT_KEYS.get(dataset, dataset)
    return cedalion.data.get_precomputed_sensitivity(key, model)


def brain_adot(Adot: xr.DataArray, channels=None) -> xr.DataArray:
    """Adot auf die Hirnvertices -- und optional auf vorhandene Kanaele -- beschneiden.

    Zwei getrennte Gruende:

      * `is_brain`: Kopfhautvertices sind fuer die Ground Truth uninteressant, und der
        Vorwaertsweg soll ausdruecklich nur Hirnaktivitaet in den Kanalraum tragen.
      * Kanalauswahl: nach dem Pruning enthaelt die Zeitreihe WENIGER Kanaele als Adot.
        Das ist der vorgesehene Weg ("y may contain less channels then W due to
        pruning"); unbekannte Kanaele in der Zeitreihe waeren dagegen ein Fehler.
    """
    A = Adot.sel(vertex=Adot.is_brain)
    if channels is not None:
        labels = [str(c) for c in np.atleast_1d(channels)]
        known = set(map(str, A.channel.values))
        missing = [c for c in labels if c not in known]
        if missing:
            raise KeyError(f"{len(missing)} Kanaele nicht in Adot: {missing[:5]} ...")
        A = A.sel(channel=labels)
    return A


def seed_vertex(head_ras, label: str) -> int:
    """Index des Hirn-Vertex, der einer Landmarke am naechsten liegt.

    Die Landmarke sitzt auf der Kopfhaut; gesucht ist der darunterliegende Kortexpunkt.
    Ueber den KD-Baum der Hirnoberflaeche, wie in Tutorial 7.
    """
    _, idx = head_ras.brain.mesh.kdtree.query(
        head_ras.landmarks.sel(label=label).pint.magnitude)
    return int(np.atleast_1d(idx)[0])


def spatial_activation(
    head_ras=None,
    labels=MOTOR_LANDMARKS,
    *,
    spatial_scale=2.0 * units.cm,
    intensity_scale=1.0 * units.micromolar,
    hbr_scale: float = -0.4,
) -> xr.DataArray:
    """Gauss-Bloebe auf dem Kortex, je Landmarke einer. Dims (vertex, chromo, trial_type).

    Die raeumliche Ausdehnung entsteht aus GEODAETISCHEN Abstaenden auf der
    Hirnoberflaeche, nicht aus euklidischen: ein Blob laeuft dadurch nicht ueber einen
    Sulcus hinweg auf den gegenueberliegenden Gyrus, obwohl der euklidisch nah waere.

    `spatial_scale` ist die Standardabweichung der Gauss-Kurve (2 cm wie in Tutorial 7),
    `hbr_scale` das Verhaeltnis HbR/HbO -- negativ, weil eine Aktivierung HbO anhebt und
    HbR senkt.

    Die Amplitude ist hier NOCH NICHT physiologisch kalibriert: `intensity_scale` ist der
    Peak *je Vertex*, und ein Kanal integriert ueber viele Vertices. Die Kalibrierung auf
    eine Peak-Konzentration im Kanalraum passiert in `calibrate_to_channel_peak`.
    """
    head_ras = head_ras if head_ras is not None else head()
    imgs = [
        synhrf.build_spatial_activation(
            head_ras.brain, seed_vertex(head_ras, lab),
            spatial_scale=spatial_scale, intensity_scale=intensity_scale,
            hbr_scale=hbr_scale)
        for lab in labels
    ]
    return xr.concat(imgs, dim="trial_type").assign_coords(
        trial_type=[f"Stim {lab}" for lab in labels])


def to_channel_space(Adot_brain: xr.DataArray, img: xr.DataArray,
                     spectrum: str = "prahl") -> xr.DataArray:
    """Vorwaertsmodell: Konzentrationsbild -> Optical Density je Kanal.

    Duenner Wrapper um `dot.forward_model.image_to_channel_space`, damit das Spektrum an
    genau einer Stelle steht -- es muss dasselbe sein wie in der Beer-Lambert-Umrechnung
    der Pipeline (`preprocess.to_conc`, ebenfalls "prahl"), sonst sind Hin- und Rueckweg
    inkonsistent.
    """
    return dot.forward_model.image_to_channel_space(Adot_brain, img, spectrum=spectrum)


#: Anteil der maximalen Hirnsensitivitaet, ab dem ein Kanal als "sieht den Kortex" gilt.
CORTEX_CHANNEL_FRAC = 0.10


def cortex_channels(Adot_brain: xr.DataArray, frac: float = CORTEX_CHANNEL_FRAC
                    ) -> np.ndarray:
    """Boolesche Maske ueber die Kanaele: welche sehen den Kortex ueberhaupt?

    Kriterium ist die aufsummierte Hirnsensitivitaet je Kanal, bezogen auf den besten
    Kanal -- nicht der Quell-Detektor-Abstand. Das ist montageunabhaengig und braucht
    keine Schwelle in Millimetern.

    Auf der 28-Kanal-Montage trennt das scharf: die langen Kanaele liegen bei 14..83, die
    acht 7-mm-Kanaele bei 0.08..0.58 -- gut zwei Groessenordnungen darunter.
    """
    A = np.abs(np.asarray(Adot_brain.values, dtype=float))
    ax = tuple(i for i, d in enumerate(Adot_brain.dims) if d != "channel")
    s = A.sum(axis=ax)
    return s >= frac * s.max()


def calibrate_to_channel_peak(od_channel: xr.DataArray, geo3d, target_uM: float = 1.0,
                              dpf: float = 6.0, spectrum: str = "prahl",
                              channel_mask: np.ndarray | None = None) -> float:
    """Faktor, der das Kanalraum-Muster auf `target_uM` Peak-Konzentration bringt.

    WARUM DIESER SCHRITT NOETIG IST: `spatial_activation` setzt den Peak *je Vertex*.
    Rechnet man das ueber Adot in den Kanalraum und von dort per Beer-Lambert zurueck in
    eine Konzentration, kommt eine ANDERE Zahl heraus -- der Kanal integriert ueber ein
    ganzes Sensitivitaetsprofil, und die Kanal-Konzentration ist eine effektive Groesse
    ueber das ganze beleuchtete Volumen, keine lokale. Tutorial 7 macht denselben Schritt
    mit einem `rescale_factor`.

    Erst dadurch ist die eingemischte Amplitude wieder in µM interpretierbar und mit den
    Kanalraum-Zahlen aus Kapitel 6 sowie mit physiologischen Erwartungswerten
    (~0.1..1 µM) vergleichbar.

    `channel_mask` schliesst Kanaele von der Peak-Suche aus, und das ist keine Feinheit.
    **Die "Konzentration" eines kurzen Kanals ist keine Amplitude.** `od2conc` teilt durch
    Kanalabstand x DPF; bei 7 mm gegen 37 mm ist dieser Nenner fuenfmal kleiner, dieselbe
    OD wird also zu einer fuenfmal groesseren Konzentration. Auf der 28-Kanal-Montage
    fuehrte das dazu, dass das GLOBALE Maximum der Kanalraum-Wahrheit auf einem 7,3-mm-Kanal
    lag (0,600 µM) -- vor dem besten langen Kanal (0,231 µM), obwohl dessen
    Hirnsensitivitaet 230-mal groesser ist. Ohne die Maske wuerde die Kalibrierung an
    diesem Artefakt verankert, und alle langen Kanaele bekaemen nur einen Bruchteil der
    Zielamplitude. Sinnvoll ist `cortex_channels(Adot_brain)`.
    """
    ts = od_channel
    if "time" not in ts.dims:                        # od2conc braucht eine Zeitachse
        ts = ts.expand_dims("time").assign_coords(time=[0.0])
        ts.time.attrs["units"] = "second"
    dpf_da = xr.DataArray([dpf] * ts.sizes["wavelength"], dims="wavelength",
                          coords={"wavelength": ts.wavelength})
    conc = cedalion.nirs.cw.od2conc(ts, geo3d, dpf_da, spectrum=spectrum)
    conc = np.abs(conc.pint.to("uM").pint.dequantify())
    if channel_mask is not None:
        conc = conc.isel(channel=np.flatnonzero(np.asarray(channel_mask, bool)))
    peak = float(conc.max())
    if not np.isfinite(peak) or peak == 0.0:
        raise ValueError("Kanalraum-Muster ist leer -- passt die Adot zur Montage?")
    return target_uM / peak


#: Memo fuer `ground_truth`. Der Vorwaertsweg ist teuer -- `compute_stacked_sensitivity`
#: baut bei nn22 eine Matrix (2 x 567) x (2 x 25 000), also ~450 MB -- haengt aber nur von
#: Montage, Blob-Parametern und Kanalmenge ab. Im Sweep wird `pipeline.build` ~1800 mal
#: aufgerufen; ohne diesen Memo waere der Vorwaertsweg der Kostentreiber des ganzen Laufs.
_GT_MEMO: dict = {}


def ground_truth(
    dataset: str,
    channels,
    geo3d,
    *,
    labels=MOTOR_LANDMARKS,
    spatial_scale_mm: float = 20.0,
    hbr_scale: float = -0.4,
    target_uM: float = 0.6,
    separate_trial_types: bool = False,
    dpf: float = 6.0,
    model: str = HEAD_MODEL,
) -> dict:
    """Ground Truth im Bildraum, plus ihr Abbild im Kanalraum.

    Das ist der Kern der Betreuungsvorgabe vom 2026-08-05: die Aktivierung wird im
    Bildraum bei C3/C4 erzeugt und von dort in den Kanalraum getragen -- nicht umgekehrt.

    Rueckgabe (dict):
        `beta_true_map` -- (channel, chromo) in µM, dequantifiziert. Der Peak der
            eingemischten Aktivierung je Kanal. Weil der HRF-Regressor auf Peak 1 normiert
            ist, ist das exakt der GLM-Koeffizient, den ein fehlerfreies Verfahren
            zurueckgeben muesste.
        `img`         -- (vertex, chromo[, trial_type]) in µM: die Wahrheit auf dem
            Kortex. Nur sie ist die eigentliche Ground Truth; `beta_true_map` ist ihr
            Schatten im Kanalraum.
        `chan_od`     -- (channel, wavelength[, trial_type]): dasselbe Muster in Optical
            Density, also die Groesse, die tatsaechlich eingemischt wird.
        `seeds`       -- {Landmarke: Vertexindex}, fuer den Lokalisationsfehler.
        `factor`      -- der Kalibrierfaktor (s. `calibrate_to_channel_peak`).

    `separate_trial_types=False` fasst C3 und C4 zu EINEM Regressor zusammen (bilaterale
    Aktivierung, ein gemeinsamer Zeitverlauf). Das ist der Default, weil der ganze
    Driftfamilien-Vergleich auf genau einem HRF-Regressor aufsetzt. Mit `True` bleiben die
    beiden Seiten getrennt -- das ist die Variante fuer die Lateralisierungsfrage, die im
    Kanalraum grundsaetzlich nicht beantwortbar ist (fehlende Landmarken im realen
    Datensatz, siehe BESPRECHUNG Abb. 17).
    """
    labels = tuple(labels)
    ch = tuple(str(c) for c in np.atleast_1d(channels))
    key = (dataset, model, labels, float(spatial_scale_mm), float(hbr_scale),
           float(target_uM), bool(separate_trial_types), float(dpf), ch)
    if key in _GT_MEMO:
        return _GT_MEMO[key]

    hd = head(model)
    img = spatial_activation(hd, labels, spatial_scale=spatial_scale_mm * units.mm,
                             intensity_scale=1.0 * units.micromolar,
                             hbr_scale=hbr_scale)
    if not separate_trial_types:
        # Bilateral als EIN Regressor: die beiden Bloebe ueberlappen nicht (verschiedene
        # Hemisphaeren), die Summe ist also einfach das gemeinsame raeumliche Muster.
        img = img.sum("trial_type")

    Ab = brain_adot(adot(dataset, model), channels=ch)
    chan_od = to_channel_space(Ab, img)

    # Nur Kanaele, die den Kortex sehen, duerfen die Amplitude festlegen -- siehe
    # `calibrate_to_channel_peak`.
    sees = cortex_channels(Ab)
    factor = calibrate_to_channel_peak(chan_od, geo3d, target_uM=target_uM, dpf=dpf,
                                       channel_mask=sees)
    img, chan_od = img * factor, chan_od * factor

    # Kanalraum-Wahrheit in Konzentration: exakt der Peak, den das GLM schaetzen soll.
    ts = chan_od.expand_dims("time").assign_coords(time=[0.0])
    ts.time.attrs["units"] = "second"
    dpf_da = xr.DataArray([dpf] * ts.sizes["wavelength"], dims="wavelength",
                          coords={"wavelength": ts.wavelength})
    beta = cedalion.nirs.cw.od2conc(ts, geo3d, dpf_da, spectrum="prahl")
    beta = beta.isel(time=0, drop=True).pint.to("uM").pint.dequantify()

    out = dict(beta_true_map=beta, img=img.pint.to("uM"), chan_od=chan_od,
               seeds={lab: seed_vertex(hd, lab) for lab in labels}, factor=factor,
               sees_cortex=sees)
    _GT_MEMO[key] = out
    return out


def c_meas_of(od: xr.DataArray) -> xr.DataArray:
    """Messvarianz je Kanal x Wellenlaenge als Rauschproxy fuer die Rekonstruktion.

    Ein verrauschter Kanal soll in der Pseudoinversen weniger Gewicht bekommen. Genau das
    ist der Grund, warum gesaettigte Kanaele hier gefaehrlich sind: das Klippen macht sie
    kuenstlich rauscharm, also bekaemen sie MAXIMALES Gewicht und wuerden ihren Fehler
    ueber ihr ganzes Sensitivitaetsprofil verteilen. Die Amplitudengrenzen in
    `preprocess.quality_masks` sind der Grund, warum das nicht passiert.
    """
    return measurement_variance(od.pint.dequantify(), calc_covariance=False)


def recon_operator(Adot: xr.DataArray, od: xr.DataArray, *,
                   alpha_spatial: float | None = ALPHA_SPATIAL,
                   k_alpha_meas: float = K_ALPHA_MEAS, brain_only: bool = True):
    """`ImageRecon` nach Betreuungsvorgabe, plus das dazu passende `c_meas`.

    Rueckgabe `(recon, c_meas)`; `c_meas` muss an `recon.reconstruct(y, c_meas)`
    weitergegeben werden.

    Drei Abweichungen von Notebook 50b, alle aus der Betreuungsvorgabe 2026-08-05:

      1. `alpha_meas` wird ueber `dot.estimate_alpha_meas(c_meas)` = `K / median(c_meas)`
         aus den Daten geschaetzt statt fest auf 0.01 gesetzt. Sinn: die Vorgabe
         balanciert Mess- und Bildregularisierung auf ein festes Verhaeltnis K, egal wie
         verrauscht der konkrete Datensatz ist. Ein fester Wert waere je Datensatz
         unterschiedlich stark.
      2. `alpha_spatial = 0.001` statt `None`. `None` heisst reine Tikhonov-Regularisierung
         auf der Messseite; mit `alpha_spatial` wird zusaetzlich die Tiefenabhaengigkeit
         korrigiert, sodass Aktivitaet nicht faelschlich auf der Kopfhaut landet.
      3. `apply_c_meas=True`, weil `c_meas` uebergeben wird. Ohne das Flag ignoriert
         `ImageRecon` die Kovarianz STILLSCHWEIGEND -- dann waere die geschaetzte
         `alpha_meas` auf eine Groesse geeicht, die gar nicht benutzt wird.

    `brain_only=True` beschraenkt die Rekonstruktion auf Hirnvertices. Das ist zugleich
    die Sparmassnahme: die Pseudoinverse W hat die Groesse (Vertices x Chromophore) x
    (Kanaele x Wellenlaengen), bei nn22 also 50 000 x 1134 -- mit Kopfhautvertices waere
    sie 40 % groesser.
    """
    c_meas = c_meas_of(od)
    alpha_meas = float(dot.estimate_alpha_meas(
        np.asarray(c_meas.pint.dequantify().values, dtype=float).ravel(),
        K=k_alpha_meas))
    recon = dot.ImageRecon(
        Adot,
        recon_mode="mua2conc",     # je Wellenlaenge mua, dann in Konzentrationen -- der
                                   # in NB 50b/47 verwendete Modus
        brain_only=brain_only,
        alpha_meas=alpha_meas,
        alpha_spatial=alpha_spatial,
        apply_c_meas=True,
        spatial_basis_functions=None,
    )
    return recon, c_meas


def sensitive_parcels(Adot: xr.DataArray, *, dOD_thresh: float = 0.001, min_ch: int = 1,
                      dHbO: float = 10.0, dHbR: float = -3.0) -> list[str]:
    """Parzellen, die die Montage ueberhaupt sehen kann (Schaefer2018).

    Eine Parzelle gilt als sichtbar, wenn eine plausible Konzentrationsaenderung darin
    (dHbO/dHbR in µM) in mindestens `min_ch` Kanaelen eine messbare OD-Aenderung
    erzeugen wuerde. Parzellen, die die Sonde nicht sieht, gehoeren nicht in eine
    Auswertung -- dort ist jeder rekonstruierte Wert reine Regularisierung.
    """
    _, mask = dot.ForwardModel.parcel_sensitivity(
        Adot, chan_droplist=None, dOD_thresh=dOD_thresh, minCh=min_ch,
        dHbO=dHbO, dHbR=dHbR)
    return mask.where(mask, drop=True)["parcel"].values.tolist()


def to_parcels(img: xr.DataArray, parcels: list[str] | None = None) -> xr.DataArray:
    """Vertex-Bild -> Parzellen-Mittelwerte, optional auf sichtbare Parzellen beschraenkt.

    Aggregation reduziert die Zahl der Freiheitsgrade von ~25 000 Vertices auf ~600
    Parzellen. Das ist bei einer duennen Montage kein Informationsverlust, sondern
    ehrlicher: die Ortsauflaesung einer 3-cm-Einzelabstandsmontage liegt bei ~2-3 cm,
    also weit ueber der Vertexdichte.
    """
    out = img.groupby("parcel").mean()
    if parcels is not None:
        out = out.sel(parcel=out.parcel.isin(parcels))
    return out


#: Sichtbarkeitsschwelle fuer Vertices, als Zehnerlogarithmus der auf das Maximum
#: normierten Gesamtsensitivitaet. -2 = ein Prozent des besten Vertex. Cedalions
#: Sensitivitaets-Plots benutzen dieselbe Skala (`low_th=-3` in NB 50b).
SENS_LOG_THRESHOLD = -2.0


def sensitivity_mask(Adot: xr.DataArray, log_threshold: float = SENS_LOG_THRESHOLD
                     ) -> np.ndarray:
    """Boolesche Maske ueber die Hirnvertices: was die Montage ueberhaupt sieht.

    WARUM DAS FUER DIE BEWERTUNG UNVERZICHTBAR IST. Die Rekonstruktion gibt fuer JEDEN
    Vertex einen Wert zurueck, auch fuer solche, zu denen kein einziges Photon gelangt
    ist. Dort ist das Ergebnis kein Messwert, sondern reine Regularisierung: die
    Tiefenkorrektur (`alpha_spatial`) teilt durch die Vertexsensitivitaet und blaest
    unsichtbare Vertices dadurch systematisch auf. Nimmt man sie in eine Korrelation oder
    einen Lokalisationsfehler mit hinein, misst man das Verhalten des Regularisierers und
    nicht die Guete der Schaetzung -- der Fehler landet dann zuverlassig irgendwo tief im
    Gehirn, wo nie etwas gemessen wurde.

    Notebook 50b loest dasselbe Problem eine Stufe grober, ueber `parcel_sensitivity`:
    Parzellen, die die Sonde nicht sieht, gehen nicht in die Auswertung ein. Diese Maske
    ist die Vertex-Variante davon und laesst sich zusaetzlich mit `to_parcels` kombinieren.
    """
    A = np.abs(np.asarray(brain_adot(Adot).values, dtype=float))
    ax = tuple(i for i, d in enumerate(brain_adot(Adot).dims) if d != "vertex")
    s = A.sum(axis=ax)
    s = s / s.max()
    with np.errstate(divide="ignore"):
        return np.log10(s) > log_threshold


def reference_od(dataset: str):
    """Eine vorverarbeitete OD-Zeitreihe des Datensatzes -- fuer `c_meas` und Diagnosen.

    Ohne Motion Correction, weil hier nur die Rauschgroesse je Kanal gebraucht wird und
    die Korrektur sie veraendern wuerde.
    """
    from drift_glm.core import preprocess as prep

    if dataset == "nn22_resting":
        import cedalion.data as cdata
        return prep.finish(prep.to_od_stage(cdata.get_nn22_resting_state()),
                           motion_method="none")
    if dataset == "multisubject_fingertapping":
        from drift_glm.data import multisubject as ms
        return ms.preprocess_recording(ms.load(ms.paths()[0]),
                                       motion_method="none")[0]
    raise ValueError(f"reference_od kennt den Datensatz {dataset!r} nicht")


def recon_check(dataset: str = "multisubject_fingertapping", *,
                alpha_meas_grid=("est", 0.001, 1.0),
                alpha_spatial_grid=(None, 0.001, 0.01),
                target_uM: float = 0.6) -> "list[dict]":
    """Rauschfreie Kontrolle des Kreises Bildraum -> Kanalraum -> Bildraum.

    Das eingemischte Muster ist bekannt, also laesst sich die Rekonstruktion OHNE
    Ruhedaten und OHNE GLM pruefen: was kommt zurueck, wenn man exakt das Wahre
    hineingibt? Was hier nicht funktioniert, kann spaeter nicht an der Driftfamilie
    liegen. Zugleich belegt der Lauf die Wahl der Regularisierungsparameter empirisch,
    statt sie nur zu uebernehmen.

    Gemessen wird ausschliesslich auf den sichtbaren Vertices (`sensitivity_mask`).

    Default ist die 28-Kanal-Montage, NICHT nn22: dort ist eine `ImageRecon`-Instanz
    ~16 MB statt ~300 MB, und die Aussage ueber die Regularisierung ist dieselbe. Auf
    dieser Maschine (7,8 GB) laesst nn22 nichts anderes daneben laufen -- ein voller
    Parameterraster darauf hat 40 Minuten CPU gebraucht und zwei parallele Laeufe
    OOM-killen lassen.
    """
    pre = reference_od(dataset)
    od = pre.od
    chans = [str(c) for c in od.channel.values]

    gt = ground_truth(dataset, chans, pre.geo3d, target_uM=target_uM)
    A = adot(dataset).sel(channel=chans)
    mask = sensitivity_mask(A)
    xyz = vertex_coords_mm()
    truth = np.asarray(gt["img"].sel(chromo="HbO").pint.dequantify().values, float)
    c_meas = c_meas_of(od)

    print(f"{dataset}: {len(chans)} Kanaele, {int(mask.sum())} von {mask.size} "
          f"Vertices sichtbar (> {SENS_LOG_THRESHOLD} log10)")
    print(f"Wahrheit: Peak {np.abs(truth).max():.3f} µM im Bildraum, "
          f"{target_uM:.2f} µM im Kanalraum\n")
    print(f"{'a_spatial':>10s} {'a_meas':>10s} {'r':>7s} {'peak_hat':>9s} "
          f"{'peak/wahr':>10s} {'Ort [mm]':>9s}")

    rows = []
    for aspat in alpha_spatial_grid:
        for am in alpha_meas_grid:
            if am == "est":
                r_, _ = recon_operator(A, od, alpha_spatial=aspat)
                a_used = r_.alpha_meas
            else:
                a_used = float(am)
                r_ = dot.ImageRecon(A, recon_mode="mua2conc", brain_only=True,
                                    alpha_meas=a_used, alpha_spatial=aspat,
                                    apply_c_meas=True, spatial_basis_functions=None)
            img = r_.reconstruct(gt["chan_od"], c_meas)
            h = np.asarray(img.sel(chromo="HbO").pint.dequantify().values, float)
            hm, tm = h[mask], truth[mask]
            r = (float(np.corrcoef(hm, tm)[0, 1])
                 if hm.std() > 0 and tm.std() > 0 else np.nan)
            idx = np.flatnonzero(mask)
            j = idx[int(np.nanargmax(np.abs(hm)))]
            loc = min(float(np.linalg.norm(xyz[j] - xyz[s]))
                      for s in gt["seeds"].values())
            rows.append(dict(alpha_spatial=aspat, alpha_meas=a_used, r=r,
                             peak=float(np.abs(hm).max()), loc_err_mm=loc))
            print(f"{str(aspat):>10s} {a_used:10.4g} {r:+7.3f} "
                  f"{np.abs(hm).max():9.3f} {np.abs(hm).max() / np.abs(tm).max():10.3f} "
                  f"{loc:9.1f}", flush=True)
            del r_, img
    return rows


def vertex_coords_mm(head_ras=None) -> np.ndarray:
    """Hirn-Vertexkoordinaten in mm, (n_vertex, 3) -- fuer Lokalisationsfehler."""
    head_ras = head_ras if head_ras is not None else head()
    v = head_ras.brain.vertices
    try:
        v = v.pint.to("mm").pint.dequantify()
    except Exception:
        pass
    return np.asarray(v.values, dtype=float)


def localisation_error_mm(img: xr.DataArray, seed: int, head_ras=None,
                          chromo: str = "HbO") -> float:
    """Abstand des rekonstruierten Maximums zum wahren Seed-Vertex [mm].

    Das ist das Guetemass, das der Kanalraum grundsaetzlich nicht liefern kann: ob die
    Aktivierung am RICHTIGEN ORT landet. Der Fehler wird euklidisch gemessen, nicht
    geodaetisch -- er soll die Abweichung im Raum beziffern, nicht einen Weg auf der
    Kortexoberflaeche.
    """
    a = img.sel(chromo=chromo) if "chromo" in img.dims else img
    a = np.asarray(a.pint.dequantify().values if hasattr(a, "pint") else a.values,
                   dtype=float)
    if a.ndim > 1:                                     # ueber Zusatzdimensionen mitteln
        ax = tuple(i for i in range(a.ndim) if i != 0)
        a = np.nanmean(a, axis=ax)
    xyz = vertex_coords_mm(head_ras)
    j = int(np.nanargmax(np.abs(a)))
    return float(np.linalg.norm(xyz[j] - xyz[seed]))


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        recon_check(*sys.argv[2:3])       # Default: die guenstige 28-Kanal-Montage
        sys.exit(0)

    dataset = sys.argv[1] if len(sys.argv) > 1 else "nn22_resting"

    t0 = time.time()
    hd = head()
    A = adot(dataset)
    print(f"Kopfmodell {HEAD_MODEL}: Hirn {hd.brain.nvertices} Vertices, "
          f"Kopfhaut {hd.scalp.nvertices}, CRS {hd.brain.crs}")
    print(f"Adot '{dataset}': {dict(A.sizes)}, "
          f"{int(A.is_brain.values.sum())} Hirnvertices, "
          f"{len(set(map(str, A.parcel.values)))} Parzellen   "
          f"({time.time() - t0:.1f}s)")

    for lab in MOTOR_LANDMARKS:
        print(f"  Seed {lab}: Vertex {seed_vertex(hd, lab)}")

    t = time.time()
    img = spatial_activation(hd)
    print(f"\nBloebe {list(map(str, img.trial_type.values))}: {dict(img.sizes)}  "
          f"({time.time() - t:.1f}s)")

    Ab = brain_adot(A)
    chan = to_channel_space(Ab, img)
    print(f"Kanalraum (OD): {dict(chan.sizes)}")
    for tt in chan.trial_type.values:
        v = np.abs(chan.sel(trial_type=tt).pint.dequantify()).max("wavelength")
        top = v.channel.values[np.argsort(v.values)[-5:]][::-1]
        print(f"  {str(tt):10s} staerkste Kanaele: {', '.join(map(str, top))}")

    # Die sichtbaren Parzellen sind die Kontrolle, dass Montage und Kopfmodell
    # zueinander passen: bei einer Motorkortex-Montage muessen es die sensomotorischen
    # sein, nicht irgendwelche.
    t = time.time()
    parc = sensitive_parcels(A)
    print(f"\nSichtbare Parzellen: {len(parc)} von "
          f"{len(set(map(str, A.parcel.values)))}  ({time.time() - t:.1f}s)")
    print("  " + ", ".join(parc[:8]) + (" ..." if len(parc) > 8 else ""))
