"""Preprocessing-Kette: Amplitude -> OD -> Motion Correction -> Masken -> Pruning -> Konzentration.

Reihenfolge wie in der Cedalion-Kette (Tutorial 3_signal_processing). Die Kanalqualitaet
(dunkel/gesaettigt) wird an der Amplitude beurteilt, die Motion Correction laeuft auf OD.
Dafuer wird die Baseline aus int2od aufgehoben: od2int(od, baseline) = baseline * exp(-od)
ist der einzige Weg von der korrigierten OD zurueck zur Amplitude. Danach wird auf OD
geprunt und per od2conc in Konzentration umgerechnet.

Aufruf:  conda run -n cedalion python -m drift_glm.core.preprocess [motion_method]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

import cedalion
import cedalion.nirs
import cedalion.sigproc.motion as motion
import cedalion.sigproc.quality as quality
from cedalion import units


@dataclass
class Preprocessed:
    """Ergebnis der Preprocessing-Kette samt Zwischenstufen fuer die Diagnose."""

    conc: xr.DataArray       # (time, channel, chromo) [µM], dequantifiziert
    od: xr.DataArray         # Optical Density nach Korrektur/Pruning
    amp_raw: xr.DataArray    # Rohamplitude [V], wie eingelesen (quantifiziert)
    amp_corr: xr.DataArray   # Amplitude nach Motion Correction [V], Basis der Masken
    baseline: xr.DataArray   # mittlere Rohamplitude (channel, wavelength) [V]
    geo3d: object            # Optodengeometrie (LabeledPoints)
    aux: object              # rec.aux_ts (Accelerometer/Gyroskop/dark signal)
    dropped: list[str] = field(default_factory=list)   # entfernte Kanaele
    masks: dict = field(default_factory=dict)          # Einzelmasken zur Diagnose
    motion_method: str = "none"                        # angewandte Motion Correction
    od_uncorrected: xr.DataArray | None = None         # OD vor der Korrektur (Diagnose)


def gate_positive(amp: xr.DataArray) -> tuple[xr.DataArray, list[str]]:
    """Verwirft Kanaele mit nicht-positiver Amplitude, Vorbedingung fuer int2od.

    int2od bildet -log(amp/baseline) und bricht bei Werten <= 0 ab. Auf nn22 betrifft das
    6 von 567 Kanaelen, die spaeter ohnehin an der mean_amp-Grenze scheitern.
    """
    ok = (amp > 0).all(dim=[d for d in amp.dims if d != "channel"])
    dropped = [str(c) for c in amp.channel.values[~ok.values]]
    return amp.sel(channel=ok), dropped


def to_od(amp: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Amplitude -> Optical Density, mit Baseline (amp.mean("time")).

    Die Baseline wird fuer den Rueckweg od2int nach der Motion Correction gebraucht.
    Der Parameter von int2od heisst `return_baseline`.
    """
    return cedalion.nirs.cw.int2od(amp, return_baseline=True)


MOTION_METHODS = ("none", "tddr", "wavelet", "tddr+wavelet")

# Default fuer Demos/Tests; im Sweep ist die Motion Correction eine eigene Achse.
# "wavelet" statt "tddr+wavelet": TDDR daempft das Driftband (<0.01 Hz) auf ~56 % und
# entfernt damit einen Teil dessen, was die Driftregressoren modellieren sollen; Wavelet
# laesst das Driftband unveraendert (siehe `band_power_ratio`).
DEFAULT_MOTION = "wavelet"

# Vorverarbeitungs-Parameter an einer Stelle; pipeline.py und realdata.py referenzieren sie.
DEFAULT_SNR_THRESHOLD = 3.0                  # SNR-Schwelle
DEFAULT_AMP_RANGE = (1e-3, 0.84)             # dunkel / gesaettigt [V], NinjaNIRS
DEFAULT_SD_RANGE = (0.0, 4.5)                # Quell-Detektor-Abstand [cm]
DEFAULT_DPF = 6.0                            # differentieller Pfadlaengenfaktor


def motion_correct(
    od: xr.DataArray,
    method: str = "tddr+wavelet",
    *,
    wavelet_iqr: float = 1.5,
    wavelet_name: str = "db2",
    wavelet_level: int = 4,
) -> xr.DataArray:
    """Motion Correction auf Optical Density.

    `tddr` (robuste Regression auf der zeitlichen Ableitung) faengt Baseline-Spruenge,
    `wavelet` (Koeffizienten ausserhalb des IQR-Bandes verwerfen) faengt Spikes; bei
    "tddr+wavelet" zuerst TDDR, wie in Cedalion NB 25. Beide arbeiten auf OD.
    Groesseres `wavelet_iqr` = staerkere Korrektur, `iqr < 0` laesst das Signal unveraendert.
    """
    if method not in MOTION_METHODS:
        raise ValueError(f"Unbekannte Motion-Correction: {method!r} "
                         f"(erlaubt: {MOTION_METHODS})")
    if method == "none":
        return od
    if "tddr" in method:
        od = motion.tddr(od)
    if "wavelet" in method:
        od = motion.wavelet(od, iqr=wavelet_iqr, wavelet=wavelet_name,
                            level=wavelet_level)
    return od


def to_amp(od: xr.DataArray, baseline: xr.DataArray) -> xr.DataArray:
    """Optical Density -> Amplitude mit der Baseline aus `to_od`.

    od2int(od, baseline) = baseline * exp(-od) ist die exakte Umkehrung von int2od. Die
    Baseline stammt aus der unkorrigierten Amplitude; die zurueckgerechnete Amplitude
    traegt also die Motion-Korrektur, behaelt aber das urspruengliche Helligkeitsniveau.
    """
    return cedalion.nirs.cw.od2int(od, baseline)


def quality_masks(
    amp: xr.DataArray,
    geo3d,
    *,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    amp_range: tuple[float, float] = DEFAULT_AMP_RANGE,
    sd_range: tuple[float, float] = DEFAULT_SD_RANGE,
) -> dict[str, xr.DataArray]:
    """Qualitaetsmasken auf der (korrigierten) Amplitude. CLEAN = True.

    `snr`: Mittelwert/Streuung ueber die Zeit. `mean_amp`: mittlere Amplitude innerhalb
    (dunkel, gesaettigt), nach Homer3 hmR_PruneChannels. `sd_dist`: Quell-Detektor-Abstand.
    Gesaettigte Kanaele wirken durch das Klippen rauscharm und werden von varianzbasierten
    Metriken nicht erkannt; in der Image Reconstruction bekaemen sie maximales Gewicht.
    """
    _, snr_mask = quality.snr(amp, snr_threshold)
    _, amp_mask = quality.mean_amp(amp, (amp_range[0] * units.V,
                                         amp_range[1] * units.V))
    _, sd_mask = quality.sd_dist(amp, geo3d, (sd_range[0] * units.cm,
                                              sd_range[1] * units.cm))
    return {"snr": snr_mask, "mean_amp": amp_mask, "sd_dist": sd_mask}


def dark_noise_floor(aux, key: str = "dark signal") -> float | None:
    """Robuster Rauschboden des Detektors aus der Dunkelmessung [V], oder None.

    nn22 fuehrt "dark signal" (Schreibweise mit Leerzeichen) als Aux-Zeitreihe mit 1134
    Spuren. Die Werte streuen um Null, ausgewertet wird deshalb die Streuung (MAD), nicht
    der Mittelwert. Die Zuordnung der Spuren zu (Kanal, Wellenlaenge) ist nicht belegt,
    der Wert wird nur aggregiert verwendet. Auf nn22 liegt er bei ~9.5e-6 V, die
    Untergrenze 1e-3 V also beim ~105-fachen.
    """
    if aux is None or key not in aux:
        return None
    d = aux[key]
    try:
        d = d.pint.dequantify()
    except Exception:
        pass
    v = np.asarray(d.values, dtype=float)
    ax = list(d.dims).index("time")
    mad = 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v, axis=ax, keepdims=True)),
                                axis=ax)
    return float(np.nanmedian(mad))


def prune(ts: xr.DataArray, masks: dict[str, xr.DataArray]) -> tuple[xr.DataArray, list[str]]:
    """Wendet die kombinierten Qualitaetsmasken an und verwirft die Kanaele.

    `prune_ch(ts, masks, "all")` verknuepft die Masken mit `&`; ein Kanal faellt heraus,
    sobald er in einer der Wellenlaengen markiert ist. Verworfen statt auf NaN gesetzt,
    weil NaN sich durch AR-IRLS und die Image Reconstruction fortpflanzen wuerde.
    """
    ts_pruned, dropped = quality.prune_ch(ts, list(masks.values()), "all")
    return ts_pruned, [str(c) for c in np.atleast_1d(dropped)]


DRIFT_BANDS = (("Drift   <0.01 Hz", 0.0, 0.01),
               ("0.01-0.1 Hz     ", 0.01, 0.1),
               ("0.1-0.5 Hz      ", 0.1, 0.5),
               ("Kardial >0.5 Hz ", 0.5, np.inf))


def band_power_ratio(od_before: xr.DataArray, od_after: xr.DataArray) -> dict:
    """Leistung je Frequenzband nach der Korrektur relativ zu vorher (Median ueber Kanaele).

    Diagnose, ob die Motion Correction in das Driftband (<0.01 Hz) eingreift und damit
    den Anteil entfernt, den die Driftregressoren modellieren sollen.
    Rueckgabe: {Bandname: Verhaeltnis}, 1.0 = unveraendert.
    """
    fs = 1.0 / float(np.median(np.diff(od_before.time.values)))
    out = {}
    a0 = np.asarray(od_before.values, float).reshape(-1, od_before.sizes["time"])
    a1 = np.asarray(od_after.values, float).reshape(-1, od_after.sizes["time"])
    a0 = a0 - a0.mean(-1, keepdims=True)
    a1 = a1 - a1.mean(-1, keepdims=True)
    freq = np.fft.rfftfreq(a0.shape[-1], d=1.0 / fs)
    P0 = np.abs(np.fft.rfft(a0, axis=-1)) ** 2
    P1 = np.abs(np.fft.rfft(a1, axis=-1)) ** 2
    for name, lo, hi in DRIFT_BANDS:
        m = (freq >= lo) & (freq < hi)
        out[name] = float(np.median(P1[:, m].sum(-1) / (P0[:, m].sum(-1) + 1e-30)))
    return out


# Amplitudenbereich, der nichts verwirft (Datensaetze ohne dunkle Population).
AMP_RANGE_OFF = (0.0, 1e12)


def amp_range_from_data(rec, min_gap: float = 3.0, max_share: float = 0.1):
    """Amplitudengrenzen (dunkel/gesaettigt) aus den Daten, oder `AMP_RANGE_OFF`.

    Die NinjaNIRS-Grenzen gelten nur fuer nn22. Hier wird geprueft, ob es eine abgetrennte
    dunkle Population gibt: die groesste relative Luecke zwischen benachbarten
    Kanalamplituden im unteren Bereich muss mindestens `min_gap` betragen und hoechstens
    `max_share` der Messungen unter sich lassen; dann wird in die Luecke geschnitten.
    Auf Khan und Multisubject-Fingertapping greift das nicht (keine Luecke), auf nn22
    liegt die Luecke zwischen 50x und 105x Rauschboden.
    """
    key = "amp" if "amp" in rec.timeseries else list(rec.timeseries.keys())[0]
    a = rec[key].pint.dequantify() if hasattr(rec[key], "pint") else rec[key]
    mp = np.asarray(a.mean("time").values, dtype=float).ravel()
    mp = np.sort(mp[np.isfinite(mp) & (mp > 0)])
    if mp.size < 10:
        return AMP_RANGE_OFF
    lower = mp[: max(int(mp.size * max_share), 1) + 1]
    if lower.size < 2:
        return AMP_RANGE_OFF
    ratios = lower[1:] / lower[:-1]
    i = int(np.argmax(ratios))
    if ratios[i] < min_gap:
        return AMP_RANGE_OFF                     # keine abgetrennte dunkle Population
    return float(np.sqrt(lower[i] * lower[i + 1])), 1e12    # Schnitt in die Luecke


def to_conc(od: xr.DataArray, geo3d, dpf: float = DEFAULT_DPF) -> xr.DataArray:
    """Optical Density -> Haemoglobinkonzentration [µM], dequantifiziert."""
    dpf_da = xr.DataArray(
        [dpf] * od.sizes["wavelength"],
        dims="wavelength",
        coords={"wavelength": od.wavelength},
    )
    conc = cedalion.nirs.cw.od2conc(od, geo3d, dpf_da, spectrum="prahl")
    return conc.pint.to("uM").pint.dequantify()


@dataclass
class ODStage:
    """Ruhedaten als Optical Density vor der Motion Correction.

    Hier wird die synthetische Aktivierung eingemischt (`to_od_activation`), damit die
    Motion Correction ueber Signal und Rauschen laeuft wie auf echten Daten. Nach der
    Korrektur eingemischt koennte sie die HRF per Konstruktion nicht beschaedigen.
    """

    od: xr.DataArray         # (channel, wavelength, time), ungeprunt, unkorrigiert
    baseline: xr.DataArray
    amp_raw: xr.DataArray
    geo3d: object
    aux: object
    dropped_nonpositive: list[str]


def to_od_stage(rec) -> ODStage:
    """Rohamplitude -> Optical Density (inkl. Positivitaets-Gate und Baseline)."""
    # nn22 liefert die Amplitude dimensionslos; als Volt quantifizieren, damit die
    # Amplitudengrenzen (dunkel/gesaettigt) eine Einheit haben.
    amp_raw = rec["amp"].pint.dequantify().pint.quantify("V")
    amp, dropped_nonpos = gate_positive(amp_raw)
    od, baseline = to_od(amp)
    return ODStage(od=od, baseline=baseline, amp_raw=amp_raw, geo3d=rec.geo3d,
                   aux=rec.aux_ts, dropped_nonpositive=dropped_nonpos)


def to_od_activation(activation_conc: xr.DataArray, geo3d, wavelength,
                     dpf: float = DEFAULT_DPF) -> xr.DataArray:
    """Konzentrations-Aktivierung [µM] -> Optical Density zum Einmischen.

    conc2od ist die exakte Umkehrung von od2conc; ohne Motion Correction kommt die
    Aktivierung ueber conc -> od -> conc unveraendert zurueck, jede Abweichung ist der
    Eingriff der Korrektur.

    Args:
        activation_conc: (time, channel, chromo) in µM, dequantifiziert.
        wavelength: Wellenlaengen-Koordinate der Ziel-OD.
        dpf: identisch zu `to_conc`.
    """
    dpf_da = xr.DataArray([dpf] * len(wavelength), dims="wavelength",
                          coords={"wavelength": wavelength})
    conc = activation_conc
    if conc.pint.units is None:
        conc = conc.pint.quantify("uM")
    return cedalion.nirs.cw.conc2od(conc, geo3d, dpf_da, spectrum="prahl")


def finish(
    stage: ODStage,
    od_in: xr.DataArray | None = None,
    *,
    motion_method: str = DEFAULT_MOTION,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    amp_range: tuple[float, float] = DEFAULT_AMP_RANGE,
    sd_range: tuple[float, float] = DEFAULT_SD_RANGE,
    dpf: float = DEFAULT_DPF,
    masks: dict[str, xr.DataArray] | None = None,
) -> Preprocessed:
    """Zweite Haelfte der Kette: Motion Correction -> Amplitude -> Masken -> Pruning -> Konzentration.

    Args:
        stage: Ergebnis von `to_od_stage`.
        od_in: zu verarbeitende OD, Default `stage.od`; fuer die Augmentation
            `stage.od + Aktivierung`.
        amp_range: (dunkel, gesaettigt) in Volt.
        sd_range: zulaessiger Quell-Detektor-Abstand in cm.
        masks: vorgegebene Masken statt neu berechneter, damit augmentierte und reine
            Variante dieselben Kanaele behalten (die Kanalqualitaet ist eine Eigenschaft
            der Messung, nicht des eingemischten Signals).
    """
    od_raw = stage.od if od_in is None else od_in
    baseline, dropped_nonpos = stage.baseline, stage.dropped_nonpositive
    amp_raw = stage.amp_raw
    od = motion_correct(od_raw, motion_method)
    # Zurueck zur Amplitude: nur dort sind "dunkel" und "gesaettigt" definiert.
    amp_corr = to_amp(od, baseline)

    if masks is None:
        masks = quality_masks(amp_corr, stage.geo3d, snr_threshold=snr_threshold,
                              amp_range=amp_range, sd_range=sd_range)

    # Pruning auf OD; amp_raw/amp_corr bleiben ungeprunt (die Stufe, auf der die Masken
    # bestimmt wurden).
    od_pruned, dropped_quality = prune(od, masks)
    conc = to_conc(od_pruned, stage.geo3d, dpf)
    # Unkorrigierte OD auf dieselben Kanaele, damit band_power_ratio gleiche Mengen sieht.
    od_raw = od_raw.sel(channel=od_pruned.channel)

    return Preprocessed(
        conc=conc,
        od=od_pruned,
        amp_raw=amp_raw,
        amp_corr=amp_corr,
        baseline=baseline,
        geo3d=stage.geo3d,
        aux=stage.aux,
        dropped=list(dropped_nonpos) + dropped_quality,
        masks={"nonpositive": dropped_nonpos, **masks},
        motion_method=motion_method,
        od_uncorrected=od_raw,
    )


def run(rec, **kwargs) -> Preprocessed:
    """Komplette Kette ohne Augmentation (`to_od_stage` + `finish`)."""
    return finish(to_od_stage(rec), **kwargs)


if __name__ == "__main__":
    import sys
    import time

    import cedalion.data

    method = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MOTION
    t0 = time.time()
    P = run(cedalion.data.get_nn22_resting_state(), motion_method=method)
    bl = np.asarray(P.baseline.pint.dequantify().values, dtype=float)
    print(f"Kanaele roh        : {P.amp_raw.sizes['channel']}")
    print(f"  nicht positiv    : {len(P.masks['nonpositive'])} verworfen "
          f"{P.masks['nonpositive']}")
    print(f"Kanaele            : {P.conc.sizes['channel']}")
    print(f"Zeitpunkte         : {P.conc.sizes['time']}")
    print(f"Baseline [V]       : min {bl.min():.3e}  median {np.median(bl):.3e}  "
          f"max {bl.max():.3e}")
    print(f"OD                 : {dict(P.od.sizes)}")
    print(f"Konzentration [µM] : {dict(P.conc.sizes)}")

    # Rueckweg-Kontrolle: ohne Korrektur muss od2int(int2od(amp)) == amp gelten.
    amp_in = gate_positive(P.amp_raw)[0].pint.dequantify().values
    amp_out = P.amp_corr.pint.dequantify().values
    rel = np.abs(amp_out - amp_in) / np.abs(amp_in)
    hi, med = float(np.nanmax(rel)), float(np.nanmedian(rel))
    if hi < 1e-9:
        print(f"\nRueckweg OD->Amp   : max. rel. Abweichung {hi:.2e} (exakt)")
    else:
        # Mit Korrektur ist die Abweichung die Korrektur selbst. Im Amplitudenraum wirkt
        # sie exponentiell (amp = baseline * exp(-od)), das Maximum stammt daher von
        # einzelnen Spikes in dunklen Kanaelen.
        print(f"\nRueckweg OD->Amp   : median {100 * med:.2f} %, max {hi:.2e} "
              f"(entspricht {np.log(hi + 1):.1f} OD, Anteil der Korrektur)")

    print(f"\nMotion Correction  : {P.motion_method}   ({time.time() - t0:.1f}s gesamt)")
    if P.od_uncorrected is not None and method != "none":
        d1 = np.abs(np.diff(np.asarray(P.od_uncorrected.values, float), axis=-1))
        d2 = np.abs(np.diff(np.asarray(P.od.values, float), axis=-1))
        print(f"  groesster Sprung : {d1.max():.4f} -> {d2.max():.4f} OD "
              f"(Spitzen der zeitlichen Ableitung)")
        print("  Restleistung je Band (100 % = unveraendert):")
        for name, r in band_power_ratio(P.od_uncorrected, P.od).items():
            flag = "  <-- Driftband" if name.startswith("Drift") and r < 0.9 else ""
            print(f"    {name} {100 * r:6.1f} %{flag}")

    nf = dark_noise_floor(P.aux)
    if nf is not None:
        mp = P.amp_raw.mean("time").pint.dequantify().values.ravel()
        print(f"\nDunkelmessung      : Rauschboden {nf:.3e} V (robust, MAD)")
        print(f"  Untergrenze 1e-3 V entspricht dem {1e-3 / nf:.0f}-fachen; "
              f"Mediansignal dem {np.median(mp) / nf:.0f}-fachen")
        counts = {k: int((mp < k * nf).sum()) for k in (10, 20, 50, 100)}
        print("  Messungen unter k x Rauschboden: "
              + ", ".join(f"{k}x:{v}" for k, v in counts.items())
              + f"  (unter 1e-3 V: {int((mp < 1e-3).sum())})")

    print("\nQualitaetsmasken auf der korrigierten Amplitude (CLEAN = True):")
    n_ch = P.amp_corr.sizes["channel"]
    keep_all = None
    for key in ("snr", "mean_amp", "sd_dist"):
        m = P.masks[key]
        # Ein Kanal ueberlebt nur, wenn er in allen uebrigen Dims (beide Wellenlaengen)
        # sauber ist; dieselbe Logik wie in xrutils.apply_mask.
        keep = m.all(dim=[d for d in m.dims if d != "channel"])
        keep_all = keep if keep_all is None else (keep_all & keep)
        print(f"  {key:9s}: {int(keep.sum()):4d} / {n_ch} behalten "
              f"({n_ch - int(keep.sum())} verworfen)")
    print(f"  {'kombiniert':9s}: {int(keep_all.sum()):4d} / {n_ch} behalten "
          f"({n_ch - int(keep_all.sum())} verworfen)")

    print(f"\nErgebnis der Kette : {P.amp_raw.sizes['channel']} roh -> "
          f"{P.conc.sizes['channel']} verwertbar  "
          f"({len(P.dropped)} verworfen: {len(P.masks['nonpositive'])} nicht positiv, "
          f"{len(P.dropped) - len(P.masks['nonpositive'])} Qualitaet)")
    print(f"OD nach Pruning    : {dict(P.od.sizes)}")
    print(f"Konzentration      : {dict(P.conc.sizes)}")
