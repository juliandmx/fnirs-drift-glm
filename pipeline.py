"""Gemeinsame Augmentations-/GLM-Pipeline fuer die Bachelorarbeit.

Laedt Ruhedaten, speist eine HRF mit BEKANNTER Peak-Amplitude ein und baut die
Designmatrix. Die Vorverarbeitung selbst (Motion Correction, Kanalqualitaet, Pruning)
liegt in `preprocess.py` -- sie haengt weder vom Analysefenster noch vom Seed ab und
kann deshalb einmal berechnet und ueber `build(pre=...)` durchgereicht werden. Der HRF-Regressor wird auf Peak = 1 normiert -- dadurch ist die
Ground-Truth-Amplitude direkt die Peak-Konzentrationsaenderung in µM (realistisch
~0.1..1 µM) statt eines uninterpretierbaren Koeffizienten auf einem ungenormten
Regressor. Injektion und Fit nutzen denselben (normierten) Regressor, sodass die
Amplitude exakt der GLM-Koeffizient ist.

Hinweis zur Form: Nur die AMPLITUDE ist physiologisch (~0.1..1 µM), nicht
automatisch die Kurvenform. Die Blockbreite entsteht aus der Faltung der
Gamma-Basis mit der Stimulusdauer (in hrf_regressors). Der Gamma-Parameter T
ist davon ENTKOPPELT (Parameter smooth_T_s, Default 0). Fruehere Versionen
setzten T=stim_dur und falteten die Boxcar dadurch doppelt -> unrealistisch
breite/spaete HRF.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from dataclasses import replace as dc_replace

import numpy as np
import xarray as xr

import cedalion
import cedalion.data
import cedalion.models.glm as glm
import cedalion.sim.synthetic_hrf as synhrf
from cedalion import units

import preprocess as prep


@dataclass
class Pipeline:
    conc: xr.DataArray          # Ruhe-Konzentration (time, channel, chromo) [µM]
    activation: xr.DataArray    # eingespeiste HRF (Ground Truth) [µM]
    conc_syn: xr.DataArray      # conc + activation [µM]
    geo3d: object               # Sonden-/Optodengeometrie (LabeledPoints, fuer scalp_plot)
    dm_hrf: glm.design_matrix.DesignMatrix    # nur HRF (normiert)
    dm_full: glm.design_matrix.DesignMatrix   # HRF + Drift
    stim_df: object
    hrf_names: list[str]
    beta_true: dict[str, float]  # {"HbO": .., "HbR": ..}  PEAK-Amplitude (Blob-Max) [µM]
    beta_true_map: xr.DataArray  # per-Kanal Ground-Truth-Peak (channel, chromo) [µM]
    raw_hrf_peak: dict[str, float]  # Peak jeder HRF-Spalte VOR der Normierung
    chromo: np.ndarray
    aux: object                  # rec.aux_ts (Accelerometer/Gyroscope, fuer Motion-Regr.)
    pre: object = None           # preprocess.Preprocessed (verworfene Kanaele, Masken, ...)


def _normalize_hrf_to_unit_peak(dm_hrf, hrf_names):
    """Skaliere jede HRF-Regressorspalte auf Peak = 1 (ueber Zeit & chromo).

    Gibt zusaetzlich die Peaks VOR der Normierung zurueck (dict pro Regressor),
    damit die eingespeiste Amplitude ohne Normierung reproduzierbar gemeldet
    werden kann.
    """
    raw_peaks = {}
    for name in hrf_names:
        peak = float(np.abs(dm_hrf.common.sel(regressor=name)).max())
        raw_peaks[name] = peak
        dm_hrf.common.loc[dict(regressor=name)] = (
            dm_hrf.common.sel(regressor=name) / peak
        )
    return dm_hrf, raw_peaks


def _spatial_beta(conc, geo3d, peak_hbo, hbr_ratio, sigma_mm):
    """Per-Kanal Ground-Truth-Peak als raeumlicher Gauss-Blob (im Kanal-Raum).

    beta_true(Kanal) = Peak * exp(-d^2 / 2 sigma^2), d = Abstand des Kanal-Mittelpunkts
    (Mittel aus Quell- und Detektorposition) zum Blob-Zentrum (Kanal am naechsten zum
    Zentroid aller Mittelpunkte). sigma_mm=None -> flach (Gewicht 1 ueberall).

    Bewusste Vereinfachung ggue. dem Augmentations-Notebook (build_spatial_activation
    ueber Vertices + Forward-Modell): der Blob wird DIREKT im Kanal-Konzentrationsraum
    definiert -> kein Kopfmodell/Adot noetig, beta_true(Kanal) ist exakt der
    Kanal-Raum-Peak in µM. Rueckgabe: DataArray (channel, chromo).
    """
    chromo = conc.chromo.values
    if sigma_mm is None:
        w = np.ones(conc.sizes["channel"])
    else:
        g = geo3d
        try:
            if g.pint.units is not None:
                g = g.pint.to("mm").pint.dequantify()
        except Exception:
            pass
        src = np.asarray(g.sel(label=conc.source.values).values, dtype=float)
        det = np.asarray(g.sel(label=conc.detector.values).values, dtype=float)
        mid = 0.5 * (src + det)                                  # (channel, 3)
        center = mid[int(np.argmin(np.linalg.norm(mid - mid.mean(0), axis=1)))]
        d = np.linalg.norm(mid - center, axis=1)
        w = np.exp(-(d ** 2) / (2.0 * sigma_mm ** 2))
    scale = {"HbO": peak_hbo, "HbR": peak_hbo * hbr_ratio}
    arr = np.stack([w * scale[str(c)] for c in chromo], axis=1)   # (channel, chromo)
    return xr.DataArray(arr, dims=("channel", "chromo"),
                        coords={"channel": conc.channel.values, "chromo": chromo})


def build(
    *,
    motion_method: str = prep.DEFAULT_MOTION,   # Achsenstufe, Begruendung in preprocess
    snr_threshold: float = 3.0,       # Betreuungsvorgabe (vorher 10)
    amp_range: tuple[float, float] = (1e-3, 0.84),   # dunkel / gesaettigt [V]
    sd_range: tuple[float, float] = (0.0, 4.5),      # Quell-Detektor-Abstand [cm]
    drift_order: int = 3,
    beta_true_hbo: float = 0.6,
    hbr_ratio: float = -0.4,
    blob_sigma_mm: float | None = 30.0,   # raeumlicher HRF-Blob; None = flache Injektion

    stim_dur_s: float = 10.0,
    min_interval_s: float = 25.0,
    max_interval_s: float = 35.0,
    tau_s: float = 0.0,
    sigma_s: float = 3.0,
    smooth_T_s: float = 0.0,   # optionale Gamma-Glaettung; NICHT die Stimulusdauer
    window_s: float | None = None,   # Analysefenster [s]; None = volle Aufnahme
    rec=None,                  # vorgeladenes Recording (spart Neuladen im Sweep)
    stage=None,                # vorberechnete preprocess.ODStage (spart int2od im Sweep)
    dpf: float = 6.0,
    seed: int = 42,
) -> Pipeline:
    # build_stim_df nutzt Pythons random-Modul -> seeden fuer Reproduzierbarkeit.
    # (Eine einzelne Stimulus-Platzierung ist nur eine Stichprobe; fuer stabile
    #  Bias/Varianz-Schaetzung ueber mehrere Seeds mitteln.)
    random.seed(seed)
    np.random.seed(seed)

    # Erste Haelfte der Kette: Rohamplitude -> OD. Sie haengt weder vom Fenster noch vom
    # Seed ab und kann im Sweep als `stage` durchgereicht werden.
    if stage is None:
        if rec is None:
            rec = cedalion.data.get_nn22_resting_state()
        stage = prep.to_od_stage(rec)
    geo3d = stage.geo3d

    # Analysefenster: auf die ersten window_s Sekunden kuerzen. Das passiert VOR der
    # Motion Correction, damit Korrektur und Auswertung dieselbe Zeitreihe sehen.
    if window_s is not None:
        fs = 1.0 / float(np.median(np.diff(stage.od.time.values)))
        n = int(round(window_s * fs))
        stage = dc_replace(stage, od=stage.od.isel(time=slice(0, n)),
                           amp_raw=stage.amp_raw.isel(time=slice(0, n)))

    # Gitter fuer Stimulus/Designmatrix/Blob: Konzentration OHNE Korrektur und OHNE
    # Pruning -- hier zaehlen nur die Koordinaten (Zeit, Kanaele, Chromophore).
    conc_grid = prep.to_conc(stage.od, geo3d, dpf)

    conc = conc_grid
    stim_df = synhrf.build_stim_df(
        max_time=conc.time.values[-1] * units.seconds,
        trial_types=["Stim"],
        min_interval=min_interval_s * units.seconds,
        max_interval=max_interval_s * units.seconds,
        min_stim_dur=stim_dur_s * units.seconds,
        max_stim_dur=stim_dur_s * units.seconds,
        min_stim_value=1.0,
        max_stim_value=1.0,
        order="random",
    )

    # T (smooth_T_s) ist bewusst von der Stimulusdauer entkoppelt: hrf_regressors
    # faltet die Basis bereits mit der Stimulusdauer (=> Blockbreite). Mit
    # T=stim_dur wuerde die Boxcar doppelt gefaltet -> unrealistisch breite/spaete
    # HRF. Default 0; 3.0 reproduziert die Glaettung aus Notebook 62.
    basis = glm.Gamma(tau=tau_s * units.s, sigma=sigma_s * units.s,
                      T=smooth_T_s * units.s)
    dm_hrf = glm.design_matrix.hrf_regressors(conc, stim_df, basis)
    hrf_names = [r for r in dm_hrf.common.regressor.values if str(r).startswith("HRF")]
    dm_hrf, raw_hrf_peak = _normalize_hrf_to_unit_peak(dm_hrf, hrf_names)

    dm_full = dm_hrf & glm.design_matrix.drift_regressors(conc, drift_order=drift_order)

    chromo = conc.chromo.values
    beta_true = {"HbO": beta_true_hbo, "HbR": beta_true_hbo * hbr_ratio}  # Peak (Blob-Max)
    beta_true_map = _spatial_beta(conc, geo3d, beta_true_hbo, hbr_ratio, blob_sigma_mm)

    betas_true = xr.DataArray(
        np.zeros((conc.sizes["channel"], dm_hrf.common.sizes["regressor"],
                  conc.sizes["chromo"])),
        dims=("channel", "regressor", "chromo"),
        coords={"channel": conc.channel.values,
                "regressor": dm_hrf.common.regressor.values, "chromo": chromo},
    )
    for name in hrf_names:
        for c in chromo:
            betas_true.loc[:, name, c] = beta_true_map.sel(chromo=c).values

    # Die INTENDIERTE Aktivierung in Konzentration (µM) -- das ist die Ground Truth.
    activation = glm.predict(conc, betas_true, dm_hrf).transpose(*conc.dims)
    # glm.predict reicht source/detector nicht durch; conc2od braucht sie aber fuer die
    # Kanalabstaende. Vom Konzentrationsgitter uebernehmen.
    activation = activation.assign_coords(
        {k: conc[k] for k in ("source", "detector") if k in conc.coords})

    # ... und jetzt der eigentliche Punkt des Umbaus: die Aktivierung wird in die OD
    # zurueckgerechnet und dort EINGEMISCHT, also VOR der Motion Correction. Dadurch
    # laeuft die Korrektur ueber Signal und Rauschen -- so wie auf echten Daten. In der
    # frueheren Fassung wurde erst nach dem kompletten Preprocessing addiert; die
    # Korrektur konnte die HRF dann per Konstruktion nicht beschaedigen und ein
    # Verfahren wie TDDR sah kuenstlich gut aus.
    act_od = prep.to_od_activation(activation, geo3d, stage.od.wavelength, dpf)

    kw = dict(motion_method=motion_method, snr_threshold=snr_threshold,
              amp_range=amp_range, sd_range=sd_range, dpf=dpf)
    # Reine Ruhedaten durch dieselbe Kette -> liefert zugleich die Kanalmasken.
    pre_clean = prep.finish(stage, **kw)
    # Augmentiert, mit den IDENTISCHEN Masken: die Kanalqualitaet ist eine Eigenschaft
    # der Messung, nicht des eingemischten Signals. Sonst haetten conc und conc_syn
    # unterschiedliche Kanalmengen.
    pre_syn = prep.finish(stage, stage.od + act_od,
                          masks={k: v for k, v in pre_clean.masks.items()
                                 if k != "nonpositive"}, **kw)

    conc, conc_syn = pre_clean.conc, pre_syn.conc
    # Ground Truth und intendierte Aktivierung auf die ueberlebenden Kanaele beschneiden.
    keep = conc_syn.channel
    activation = activation.sel(channel=keep)
    beta_true_map = beta_true_map.sel(channel=keep)

    return Pipeline(conc=conc, activation=activation, conc_syn=conc_syn,
                    geo3d=geo3d, dm_hrf=dm_hrf, dm_full=dm_full, stim_df=stim_df,
                    hrf_names=hrf_names, beta_true=beta_true,
                    beta_true_map=beta_true_map, raw_hrf_peak=raw_hrf_peak,
                    chromo=chromo, aux=pre_syn.aux, pre=pre_syn)
