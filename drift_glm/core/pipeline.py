"""Gemeinsame Augmentations-/GLM-Pipeline fuer die Bachelorarbeit.

Laedt Ruhedaten, speist eine HRF mit BEKANNTER Peak-Amplitude ein und baut die
Designmatrix.

WO DIE GROUND TRUTH LIEGT (Betreuungsvorgabe 2026-08-05). Die Aktivierung wird im
BILDRAUM erzeugt -- als Gauss-Blob auf dem Kortex unter C3 und C4 -- und von dort ueber
das Vorwaertsmodell (Sensitivitaetsmatrix) in den Kanalraum getragen. Erst dort kommen
die Ruhedaten dazu. Damit gelten beide Vorgaben gleichzeitig, die sich zu widersprechen
schienen: erzeugt wird im Bildraum (2026-08-05), eingemischt im Kanalraum (2026-07-11).

Der Unterschied zur alten Fassung ist nicht kosmetisch. Vorher war die Wahrheit ein Blob
ueber den KANAL-MITTELPUNKTEN -- eine Hilfskonstruktion ohne Kopfmodell, bei der die
raeumliche Ausdehnung nichts mit Anatomie zu tun hatte und ein Kanal genau dann aktiv war,
wenn sein Mittelpunkt nah am Zentrum lag. Jetzt ergibt sich die Aktivierung eines Kanals
daraus, wie stark sein Sensitivitaetsprofil den Blob ueberlappt. Das ist zugleich der
einzige Weg, den Fehler AM ORT zu messen statt nur je Kanal. Die alte Variante bleibt
ueber `activation_space="channel"` erreichbar, damit die v4-Zahlen anschlussfaehig
bleiben.

Die Vorverarbeitung selbst liegt vollstaendig in `preprocess.py` -- dieses Modul ruft
sie nur auf und haelt KEINE eigenen Preprocessing-Schritte und keine eigenen
Standardwerte dafuer (die Defaults werden aus `preprocess` referenziert, damit sie nicht
auseinanderlaufen). Die erste Haelfte der Kette (Rohamplitude -> Optical Density) haengt
weder vom Analysefenster noch vom Seed ab und kann ueber `build(stage=...)`
durchgereicht werden; die zweite Haelfte (Motion Correction -> Pruning -> Konzentration)
laeuft je Build, weil die Aktivierung VOR der Korrektur eingemischt wird.

Der HRF-Regressor wird auf Peak = 1 normiert -- dadurch ist die Ground-Truth-Amplitude
direkt die Peak-Konzentrationsaenderung in µM (realistisch ~0.1..1 µM) statt eines
uninterpretierbaren Koeffizienten auf einem ungenormten Regressor. Injektion und Fit
nutzen denselben (normierten) Regressor, sodass die Amplitude exakt der GLM-Koeffizient
ist.

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

import cedalion.data
import cedalion.models.glm as glm
import cedalion.sim.synthetic_hrf as synhrf
from cedalion import units

from drift_glm.core import imagespace as ims
from drift_glm.core import preprocess as prep
from drift_glm.core import shortchannel as sc


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
    activation_space: str = "image"   # "image" (Kortex-Blob) oder "channel" (alte Fassung)
    beta_true_img: xr.DataArray = None   # Ground Truth auf dem Kortex (vertex, chromo) [µM]
    seeds: dict = None           # {Landmarke: Vertexindex} des Blob-Zentrums
    dataset: str = "nn22_resting"        # bestimmt die Sensitivitaetsmatrix


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
    snr_threshold: float = prep.DEFAULT_SNR_THRESHOLD,
    amp_range: tuple[float, float] = prep.DEFAULT_AMP_RANGE,
    sd_range: tuple[float, float] = prep.DEFAULT_SD_RANGE,
    drift_order: int = 3,
    beta_true_hbo: float = 0.6,
    hbr_ratio: float = -0.4,
    activation_space: str = "image",      # "image" = Kortex-Blob (Vorgabe), "channel" = alt
    dataset: str = "nn22_resting",        # Schluessel der Sensitivitaetsmatrix
    act_labels=ims.MOTOR_LANDMARKS,       # Blob-Zentren, Vorgabe: C3 und C4
    spatial_scale_mm: float = 20.0,       # Blob-Streuung auf dem Kortex (geodaetisch)
    separate_trial_types: bool = False,   # C3/C4 getrennt statt bilateral (s. imagespace)
    blob_sigma_mm: float | None = 30.0,   # nur activation_space="channel": Kanalraum-Blob
    inject_long_only: bool | None = None,  # None = automatisch, s.u.
    short_threshold=sc.SHORT_THRESHOLD,   # Grenze lang/kurz (Betreuungsvorgabe 1,8 cm)

    stim_dur_s: float = 10.0,
    min_interval_s: float = 25.0,
    max_interval_s: float = 35.0,
    tau_s: float = 0.0,
    sigma_s: float = 3.0,
    smooth_T_s: float = 0.0,   # optionale Gamma-Glaettung; NICHT die Stimulusdauer
    window_s: float | None = None,   # Analysefenster [s]; None = volle Aufnahme
    rec=None,                  # vorgeladenes Recording (spart Neuladen im Sweep)
    stage=None,                # vorberechnete preprocess.ODStage (spart int2od im Sweep)
    dpf: float = prep.DEFAULT_DPF,
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

    # Die Ground Truth zuerst -- sie legt fest, WIE VIELE HRF-Regressoren es gibt.
    if activation_space == "image":
        gt = ims.ground_truth(
            dataset, conc.channel.values, geo3d, labels=act_labels,
            spatial_scale_mm=spatial_scale_mm, hbr_scale=hbr_ratio,
            target_uM=beta_true_hbo, separate_trial_types=separate_trial_types, dpf=dpf)
        beta_true_map, beta_true_img, seeds = gt["beta_true_map"], gt["img"], gt["seeds"]
        sees_cortex = gt["sees_cortex"]
        trial_types = ([str(t) for t in beta_true_map.trial_type.values]
                       if "trial_type" in beta_true_map.dims else ["Stim"])
    elif activation_space == "channel":
        beta_true_map = _spatial_beta(conc, geo3d, beta_true_hbo, hbr_ratio,
                                      blob_sigma_mm)
        beta_true_img, seeds, trial_types = None, None, ["Stim"]
        sees_cortex = np.ones(conc.sizes["channel"], dtype=bool)
    else:
        raise ValueError(f"activation_space muss 'image' oder 'channel' sein, "
                         f"nicht {activation_space!r}")

    stim_df = synhrf.build_stim_df(
        max_time=conc.time.values[-1] * units.seconds,
        trial_types=trial_types,
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

    # Kurze Kanaele duerfen die HRF nicht enthalten: ihre "Banane" erreicht den Kortex
    # nicht, sie messen nur Kopfhaut. Enthielte der Short-Channel-Regressor die HRF, wuerde
    # er sie aus den langen Kanaelen herausregressieren -- exakt das Artefakt, das in
    # Sweep v1 der Global-Mean-Regressor bei raeumlich flacher Injektion erzeugte.
    #
    # Im Bildraum erledigt das die Physik: der Blob sitzt auf dem Kortex, und die
    # Sensitivitaet eines 8-18-mm-Kanals fuer Hirnvertices ist um Groessenordnungen
    # kleiner als die eines 3-cm-Kanals. Das Nullen von Hand ist dann nicht nur
    # unnoetig, sondern falsch -- es wuerde einen real vorhandenen (kleinen) Anteil
    # unterdruecken und den Short-Channel-Regressor besser aussehen lassen, als er ist.
    # Wie klein der Restanteil tatsaechlich ist, misst `python -m drift_glm.core.pipeline leakage`.
    if inject_long_only is None:
        inject_long_only = (activation_space == "channel")
    if inject_long_only:
        _, ts_short = sc.split(conc, geo3d, short_threshold)
        short_labels = {str(c) for c in ts_short.channel.values}
        is_long = xr.DataArray(
            [str(c) not in short_labels for c in beta_true_map.channel.values],
            dims="channel", coords={"channel": beta_true_map.channel.values})
        beta_true_map = beta_true_map.where(is_long, 0.0)

    # Peak-Referenz: der staerkste Kanal bei HbO, und der HbR-Wert DESSELBEN Kanals.
    # Bei der Kanalraum-Variante ist das per Konstruktion (beta_true_hbo, ratio*peak); im
    # Bildraum wird es gemessen, weil der Weg Bildraum -> Adot -> Beer-Lambert das
    # Verhaeltnis leicht verschiebt (verschiedene Extinktionskoeffizienten und
    # Kanalabstaende).
    _bt = beta_true_map
    if "trial_type" in _bt.dims:
        _bt = _bt.max("trial_type")
    # Nur Kanaele, die den Kortex sehen, duerfen die Peak-Referenz stellen: die
    # "Konzentration" eines kurzen Kanals ist wegen des kurzen Nenners in od2conc keine
    # Amplitude (Begruendung in imagespace.calibrate_to_channel_peak).
    _v = np.abs(np.asarray(_bt.sel(chromo="HbO").values, float))
    _j = int(np.nanargmax(np.where(sees_cortex, _v, -np.inf)))
    beta_true = {str(c): float(_bt.sel(chromo=c).isel(channel=_j)) for c in chromo}

    betas_true = xr.DataArray(
        np.zeros((conc.sizes["channel"], dm_hrf.common.sizes["regressor"],
                  conc.sizes["chromo"])),
        dims=("channel", "regressor", "chromo"),
        coords={"channel": conc.channel.values,
                "regressor": dm_hrf.common.regressor.values, "chromo": chromo},
    )
    for name in hrf_names:
        # Regressorname ist "HRF <trial_type>" -- bei getrennten Seiten bekommt jeder
        # Regressor sein eigenes raeumliches Muster.
        bt = beta_true_map
        if "trial_type" in bt.dims:
            bt = bt.sel(trial_type=str(name).removeprefix("HRF ").strip())
        for c in chromo:
            betas_true.loc[:, name, c] = bt.sel(chromo=c).values

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
                    chromo=chromo, aux=pre_syn.aux, pre=pre_syn,
                    activation_space=activation_space, beta_true_img=beta_true_img,
                    seeds=seeds, dataset=dataset)


def _leakage_report(dataset: str = "nn22_resting", window_s: float = 180.0):
    """Wie viel der eingemischten HRF landet in den kurzen Kanaelen?

    Die Zahl entscheidet, ob das Nullen von Hand (`inject_long_only`) noch gebraucht
    wird. Bei Bildraum-Injektion sollte der Anteil klein sein, weil ein kurzer Kanal fuer
    Hirnvertices kaum sensitiv ist -- die Frage ist, WIE klein.
    """
    P = build(dataset=dataset, window_s=window_s, activation_space="image",
              inject_long_only=False)
    bt = P.beta_true_map
    if "trial_type" in bt.dims:
        bt = bt.max("trial_type")
    _, ts_short = sc.split(P.conc, P.geo3d, sc.SHORT_THRESHOLD)
    short = {str(c) for c in ts_short.channel.values}
    is_short = np.array([str(c) in short for c in bt.channel.values])
    d = sc.distances_mm(P.conc, P.geo3d)

    print(f"Datensatz {dataset}: {bt.sizes['channel']} Kanaele, "
          f"{int(is_short.sum())} davon kurz (< {sc.SHORT_THRESHOLD})")
    print(f"Abstaende: kurz {d[is_short].min():.1f}-{d[is_short].max():.1f} mm, "
          f"lang {d[~is_short].min():.1f}-{d[~is_short].max():.1f} mm")

    # Die scharfe Fassung der Limitation aus shortchannel.py: sehen die "kurzen" Kanaele
    # den Kortex? Auf einer echten Short-Separation-Montage (7-8 mm) ist die Antwort nein,
    # auf nn22 (15,5-18 mm) ja -- und dann ist der Regressor kein reiner Systemik-Proxy.
    Ab = ims.brain_adot(ims.adot(dataset), channels=[str(c) for c in bt.channel.values])
    sees = ims.cortex_channels(Ab)
    print(f"Sehen den Kortex (>= {100 * ims.CORTEX_CHANNEL_FRAC:.0f} % der maximalen "
          f"Hirnsensitivitaet): {int(sees.sum())} Kanaele, davon "
          f"{int((sees & is_short).sum())} von {int(is_short.sum())} kurzen")
    print("\nWie viel der eingemischten HRF steckt in welchem systemischen Regressor?")
    print("(Ein Regressor, der einen Teil des Gesuchten enthaelt, rechnet ihn weg.)")
    for c in ("HbO", "HbR"):
        v = np.asarray(bt.sel(chromo=c).values, float)
        a = np.abs(v)
        pk = a.max()
        print(f"  {c}  Peak {pk:+.4f} µM")
        print(f"      staerkster kurzer Kanal : {a[is_short].max():.4f} µM "
              f"({100 * a[is_short].max() / pk:4.1f} % des Peaks)")
        print(f"      short_avg  (Mittel kurz): {v[is_short].mean():+.4f} µM "
              f"({100 * abs(v[is_short].mean()) / pk:4.1f} %)")
        print(f"      global     (Mittel alle): {v.mean():+.4f} µM "
              f"({100 * abs(v.mean()) / pk:4.1f} %)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "leakage":
        _leakage_report(*(sys.argv[2:3] or ["nn22_resting"]))
    else:
        P = build(window_s=180.0)
        print(f"activation_space : {P.activation_space}  (Datensatz {P.dataset})")
        print(f"Kanaele          : {P.conc.sizes['channel']}")
        print(f"HRF-Regressoren  : {P.hrf_names}")
        print(f"Ground Truth Peak: " + ", ".join(f"{k} {v:+.3f} µM"
                                                 for k, v in P.beta_true.items()))
        print(f"Blob-Zentren     : {P.seeds}")
        if P.beta_true_img is not None:
            im = P.beta_true_img
            print(f"Bildraum-Wahrheit: {dict(im.sizes)}, "
                  f"Peak HbO {float(np.abs(im.sel(chromo='HbO')).max()):.3f} µM")
