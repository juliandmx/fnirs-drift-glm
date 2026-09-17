"""Bildraum: Kopfmodell, Sensitivitaetsmatrix, Vorwaerts- und Rueckweg.

Einzige Stelle, an der Kopfmodell (ICBM152) und Sensitivitaetsmatrix beschafft werden;
pipeline.py, imageglm.py und multisubject.py greifen hier zu. Zwei Richtungen:
`to_channel_space` traegt einen Blob auf dem Kortex (C3/C4) als Vorwaertsmodell in
Optical Density je Kanal (synthetische Aktivierung), `recon_operator` projiziert
GLM-Ergebnisse regularisiert auf den Kortex zurueck.

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

HEAD_MODEL = "icbm152"      # durchgaengig ICBM152, nicht Colin27

# Regularisierung der Rekonstruktion. alpha_meas wird aus den Daten geschaetzt
# (`dot.estimate_alpha_meas`, s. `recon_operator`). alpha_spatial steuert die
# Tiefenkorrektur R_j = 1/(sum_i A_ij^2 + alpha_spatial * max); kleine Werte blaehen
# schwach gesehene Vertices auf. Rauschfrei auf der 28-Kanal-Montage (recon_check) liefert
# 0.01 (Cedalion-Preset REG_PAPER_MUA_SBF) r = 0.65 und Peak/wahr 0.97, 0.001 dagegen
# r = 0.51 und Peak/wahr 2.0; der Lokalisationsfehler (~14 mm) haengt kaum daran. 0.001
# bleibt der Default und ist in `recon_operator` durchgereicht.
ALPHA_SPATIAL = 0.001

# K in alpha_meas = K / median(C_meas); Cedalion-Default.
K_ALPHA_MEAS = 0.01

# Aktivierungsorte: primaerer Motorkortex links/rechts (10-20-System).
MOTOR_LANDMARKS = ("C3", "C4")

# Logischer Datensatzname -> Schluessel der vorberechneten Sensitivitaetsmatrix. Fuer die
# NIRScout-Montage des Khan-Datensatzes gibt es keine, siehe `coregister.py`.
ADOT_KEYS = {
    "nn22_resting": "nn22_resting",
    "multisubject_fingertapping": "fingertapping",
}


@lru_cache(maxsize=2)
def head(model: str = HEAD_MODEL, ras: bool = True):
    """Kopfmodell (Hirn- und Kopfhautoberflaeche, Landmarken), gecacht.

    `ras=True` transformiert in den RAS/MNI-Raum, in dem die Vertices in mm vorliegen;
    die ijk-Oberflaeche ist einheitenlos, `build_spatial_activation` koennte dort
    `spatial_scale` nicht umrechnen. Die Vertex-Reihenfolge bleibt erhalten, ein auf
    `head(ras=True).brain` gerechneter Blob passt also auf die Brain-Vertices von Adot.
    """
    h = dot.get_standard_headmodel(model)
    return h.apply_transform(h.t_ijk2ras) if ras else h


@lru_cache(maxsize=4)
def adot(dataset: str, model: str = HEAD_MODEL) -> xr.DataArray:
    """Vorberechnete Sensitivitaetsmatrix, dims (channel, vertex, wavelength).

    Traegt `is_brain` (Hirnvertices zuerst, dann Kopfhaut) und `parcel` (Schaefer2018)
    als Vertex-Koordinaten.
    """
    key = ADOT_KEYS.get(dataset, dataset)
    return cedalion.data.get_precomputed_sensitivity(key, model)


def brain_adot(Adot: xr.DataArray, channels=None) -> xr.DataArray:
    """Adot auf die Hirnvertices und optional auf die vorhandenen Kanaele beschneiden.

    Nach dem Pruning enthaelt die Zeitreihe weniger Kanaele als Adot, das ist vorgesehen;
    Kanaele, die Adot nicht kennt, sind ein Fehler.
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
    """Index des Hirn-Vertex unter einer Kopfhaut-Landmarke (KD-Baum, wie Tutorial 7)."""
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

    Die Ausdehnung folgt geodaetischen Abstaenden auf der Hirnoberflaeche (ein Blob laeuft
    nicht ueber einen Sulcus). `spatial_scale` ist die Standardabweichung, `hbr_scale` das
    Verhaeltnis HbR/HbO. `intensity_scale` ist der Peak je Vertex und noch nicht
    physiologisch kalibriert; das passiert in `calibrate_to_channel_peak`.
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

    Das Spektrum muss dasselbe sein wie in `preprocess.to_conc` ("prahl"), sonst sind
    Hin- und Rueckweg inkonsistent.
    """
    return dot.forward_model.image_to_channel_space(Adot_brain, img, spectrum=spectrum)


# Anteil der maximalen Hirnsensitivitaet, ab dem ein Kanal als "sieht den Kortex" gilt.
CORTEX_CHANNEL_FRAC = 0.10


def cortex_channels(Adot_brain: xr.DataArray, frac: float = CORTEX_CHANNEL_FRAC
                    ) -> np.ndarray:
    """Boolesche Maske ueber die Kanaele: welche den Kortex sehen.

    Kriterium ist die aufsummierte Hirnsensitivitaet je Kanal relativ zum besten Kanal,
    nicht der Quell-Detektor-Abstand; das ist montageunabhaengig. Auf der 28-Kanal-Montage
    liegen die langen Kanaele bei 14..83, die acht 7-mm-Kanaele bei 0.08..0.58.
    """
    A = np.abs(np.asarray(Adot_brain.values, dtype=float))
    ax = tuple(i for i, d in enumerate(Adot_brain.dims) if d != "channel")
    s = A.sum(axis=ax)
    return s >= frac * s.max()


def calibrate_to_channel_peak(od_channel: xr.DataArray, geo3d, target_uM: float = 1.0,
                              dpf: float = 6.0, spectrum: str = "prahl",
                              channel_mask: np.ndarray | None = None) -> float:
    """Faktor, der das Kanalraum-Muster auf `target_uM` Peak-Konzentration bringt.

    `spatial_activation` setzt den Peak je Vertex; ein Kanal integriert ueber sein ganzes
    Sensitivitaetsprofil, die Kanal-Konzentration ist also eine andere Zahl (Tutorial 7:
    `rescale_factor`). Erst kalibriert ist die eingemischte Amplitude in µM interpretierbar.

    `channel_mask` schliesst Kanaele von der Peak-Suche aus, sinnvoll ist
    `cortex_channels(Adot_brain)`: die Konzentration eines kurzen Kanals ist keine
    Amplitude, weil od2conc durch Kanalabstand x DPF teilt und dieselbe OD bei 7 mm gegen
    37 mm eine fuenfmal groessere Konzentration ergibt. Ohne Maske laege der Peak auf einem
    kurzen Kanal und alle langen bekaemen nur einen Bruchteil der Zielamplitude.
    """
    ts = od_channel
    if "time" not in ts.dims:                        # od2conc braucht eine Zeitachse
        ts = ts.expand_dims("time").assign_coords(time=[0.0])
        ts.time.attrs["units"] = "second"
    dpf_da = xr.DataArray([dpf] * ts.sizes["wavelength"], dims="wavelength",
                          coords={"wavelength": ts.wavelength})
    conc = cedalion.nirs.cw.od2conc(ts, geo3d, dpf_da, spectrum=spectrum)
    conc = np.abs(conc.pint.to("uM").pint.dequantify())
    # od2conc gibt die Kanaele nicht in der Eingabereihenfolge zurueck; channel_mask liegt
    # in der Reihenfolge von od_channel, deshalb vor dem positionalen isel zurueckordnen.
    conc = conc.sel(channel=od_channel.channel.values)
    if channel_mask is not None:
        conc = conc.isel(channel=np.flatnonzero(np.asarray(channel_mask, bool)))
    peak = float(conc.max())
    if not np.isfinite(peak) or peak == 0.0:
        raise ValueError("Kanalraum-Muster ist leer -- passt die Adot zur Montage?")
    return target_uM / peak


# Memo fuer `ground_truth`: der Vorwaertsweg (compute_stacked_sensitivity, bei nn22 ~450 MB)
# haengt nur von Montage, Blob-Parametern und Kanalmenge ab; im Sweep wird build ~1800
# mal aufgerufen.
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
    """Ground Truth im Bildraum plus ihr Abbild im Kanalraum.

    Rueckgabe (dict):
        `beta_true_map` -- (channel, chromo) in µM: Peak der eingemischten Aktivierung je
            Kanal; wegen der Peak-1-Normierung des HRF-Regressors exakt der erwartete
            GLM-Koeffizient.
        `img`         -- (vertex, chromo[, trial_type]) in µM, die Wahrheit auf dem Kortex.
        `chan_od`     -- (channel, wavelength[, trial_type]): dasselbe Muster in Optical
            Density, die Groesse, die eingemischt wird.
        `seeds`       -- {Landmarke: Vertexindex}.
        `factor`      -- Kalibrierfaktor (s. `calibrate_to_channel_peak`).
        `sees_cortex` -- Maske aus `cortex_channels`.

    `separate_trial_types=False` fasst C3 und C4 zu einem Regressor zusammen (bilaterale
    Aktivierung); `True` haelt die Seiten getrennt (Lateralisierungsfrage).
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
        # Die Bloebe liegen auf verschiedenen Hemisphaeren und ueberlappen nicht; die
        # Summe ist das gemeinsame raeumliche Muster.
        img = img.sum("trial_type")

    Ab = brain_adot(adot(dataset, model), channels=ch)
    chan_od = to_channel_space(Ab, img)
    # Der Vorwaertsweg gibt die Kanaele nicht zwingend in der Eingabereihenfolge zurueck;
    # sees_cortex, Kalibrierung und Rueckgabe rechnen in der Reihenfolge von `channels`.
    chan_od = chan_od.sel(channel=list(ch))

    # Nur Kanaele, die den Kortex sehen, legen die Amplitude fest
    # (s. `calibrate_to_channel_peak`).
    sees = cortex_channels(Ab)
    factor = calibrate_to_channel_peak(chan_od, geo3d, target_uM=target_uM, dpf=dpf,
                                       channel_mask=sees)
    img, chan_od = img * factor, chan_od * factor

    # Kanalraum-Wahrheit in Konzentration: der Peak, den das GLM schaetzen soll.
    ts = chan_od.expand_dims("time").assign_coords(time=[0.0])
    ts.time.attrs["units"] = "second"
    dpf_da = xr.DataArray([dpf] * ts.sizes["wavelength"], dims="wavelength",
                          coords={"wavelength": ts.wavelength})
    beta = cedalion.nirs.cw.od2conc(ts, geo3d, dpf_da, spectrum="prahl")
    beta = beta.isel(time=0, drop=True).pint.to("uM").pint.dequantify()
    # od2conc gibt die Kanaele nicht in der Eingabereihenfolge zurueck; ohne Rueckordnung
    # wuerde jede positionale Verwendung der Karte die Werte auf falsche Kanaele verteilen.
    beta = beta.sel(channel=list(ch))

    out = dict(beta_true_map=beta, img=img.pint.to("uM"), chan_od=chan_od,
               seeds={lab: seed_vertex(hd, lab) for lab in labels}, factor=factor,
               sees_cortex=sees)
    _GT_MEMO[key] = out
    return out


def c_meas_of(od: xr.DataArray) -> xr.DataArray:
    """Messvarianz je Kanal x Wellenlaenge als Rauschproxy fuer die Rekonstruktion.

    Verrauschte Kanaele bekommen in der Pseudoinversen weniger Gewicht. Gesaettigte
    Kanaele waeren durch das Klippen kuenstlich rauscharm und bekaemen maximales Gewicht;
    dagegen stehen die Amplitudengrenzen in `preprocess.quality_masks`.
    """
    return measurement_variance(od.pint.dequantify(), calc_covariance=False)


def recon_operator(Adot: xr.DataArray, od: xr.DataArray, *,
                   alpha_spatial: float | None = ALPHA_SPATIAL,
                   k_alpha_meas: float = K_ALPHA_MEAS, brain_only: bool = True):
    """`ImageRecon` mit den Projekt-Defaults, plus das dazu passende `c_meas`.

    Rueckgabe `(recon, c_meas)`; `c_meas` muss an `recon.reconstruct(y, c_meas)`
    weitergegeben werden. Gegenueber Notebook 50b: `alpha_meas` wird als
    K / median(c_meas) aus den Daten geschaetzt statt fest gesetzt; `alpha_spatial`
    korrigiert zusaetzlich die Tiefenabhaengigkeit; `apply_c_meas=True`, weil `ImageRecon`
    die Kovarianz sonst stillschweigend ignoriert. `brain_only=True` beschraenkt die
    Rekonstruktion auf Hirnvertices (W ist bei nn22 50 000 x 1134, mit Kopfhaut 40 %
    groesser).
    """
    c_meas = c_meas_of(od)
    alpha_meas = float(dot.estimate_alpha_meas(
        np.asarray(c_meas.pint.dequantify().values, dtype=float).ravel(),
        K=k_alpha_meas))
    recon = dot.ImageRecon(
        Adot,
        recon_mode="mua2conc",     # je Wellenlaenge mua, dann Konzentration (wie NB 50b/47)
        brain_only=brain_only,
        alpha_meas=alpha_meas,
        alpha_spatial=alpha_spatial,
        apply_c_meas=True,
        spatial_basis_functions=None,
    )
    return recon, c_meas


def sensitive_parcels(Adot: xr.DataArray, *, dOD_thresh: float = 0.001, min_ch: int = 1,
                      dHbO: float = 10.0, dHbR: float = -3.0) -> list[str]:
    """Parzellen (Schaefer2018), die die Montage sehen kann.

    Sichtbar heisst: eine Konzentrationsaenderung dHbO/dHbR [µM] in der Parzelle erzeugt
    in mindestens `min_ch` Kanaelen eine OD-Aenderung von mindestens `dOD_thresh`.
    """
    _, mask = dot.ForwardModel.parcel_sensitivity(
        Adot, chan_droplist=None, dOD_thresh=dOD_thresh, minCh=min_ch,
        dHbO=dHbO, dHbR=dHbR)
    return mask.where(mask, drop=True)["parcel"].values.tolist()


def to_parcels(img: xr.DataArray, parcels: list[str] | None = None) -> xr.DataArray:
    """Vertex-Bild -> Parzellen-Mittelwerte, optional auf sichtbare Parzellen beschraenkt.

    Reduziert ~25 000 Vertices auf ~600 Parzellen; die Ortsaufloesung einer 3-cm-Montage
    liegt ohnehin bei 2-3 cm.
    """
    out = img.groupby("parcel").mean()
    if parcels is not None:
        out = out.sel(parcel=out.parcel.isin(parcels))
    return out


# Sichtbarkeitsschwelle fuer Vertices: log10 der auf das Maximum normierten
# Gesamtsensitivitaet, -2 = ein Prozent des besten Vertex (Cedalion-Plots: low_th=-3).
SENS_LOG_THRESHOLD = -2.0


def sensitivity_mask(Adot: xr.DataArray, log_threshold: float = SENS_LOG_THRESHOLD
                     ) -> np.ndarray:
    """Boolesche Maske ueber die Hirnvertices, die die Montage sieht.

    Die Rekonstruktion liefert fuer jeden Vertex einen Wert, auch fuer unsichtbare; dort
    ist er reine Regularisierung, und die Tiefenkorrektur blaeht unsichtbare Vertices
    systematisch auf. Korrelation und Lokalisationsfehler werden deshalb nur auf dieser
    Maske berechnet. Notebook 50b macht dasselbe grober ueber `parcel_sensitivity`.
    """
    A = np.abs(np.asarray(brain_adot(Adot).values, dtype=float))
    ax = tuple(i for i, d in enumerate(brain_adot(Adot).dims) if d != "vertex")
    s = A.sum(axis=ax)
    s = s / s.max()
    with np.errstate(divide="ignore"):
        return np.log10(s) > log_threshold


def vertex_coords_mm(head_ras=None) -> np.ndarray:
    """Hirn-Vertexkoordinaten in mm, (n_vertex, 3)."""
    head_ras = head_ras if head_ras is not None else head()
    v = head_ras.brain.vertices
    try:
        v = v.pint.to("mm").pint.dequantify()
    except Exception:
        pass
    return np.asarray(v.values, dtype=float)


if __name__ == "__main__":
    import sys
    import time

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

    # Kontrolle, dass Montage und Kopfmodell zusammenpassen: bei einer Motorkortex-Montage
    # muessen die sichtbaren Parzellen die sensomotorischen sein.
    t = time.time()
    parc = sensitive_parcels(A)
    print(f"\nSichtbare Parzellen: {len(parc)} von "
          f"{len(set(map(str, A.parcel.values)))}  ({time.time() - t:.1f}s)")
    print("  " + ", ".join(parc[:8]) + (" ..." if len(parc) > 8 else ""))
