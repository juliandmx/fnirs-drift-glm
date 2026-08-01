"""Preprocessing-Kette nach Betreuungsvorgabe (Gespraechsnotizen 2026-08-01).

Die Reihenfolge folgt exakt der Vorgabe aus dem Betreuungsgespraech -- und damit auch
der kanonischen Cedalion-Kette (examples/tutorial/3_signal_processing.ipynb:
"quality assessment -> OD conversion -> motion correction -> filtering ->
haemoglobin concentration"):

    Rohamplitude
      -> int2od  (BASELINE merken, sonst ist der Rueckweg nicht moeglich)
      -> Motion Correction auf OD          [Schritt 1.3]
      -> zurueck zur Amplitude via od2int  [Schritt 1.4]
      -> Qualitaetsmasken auf der KORRIGIERTEN Amplitude   [Schritt 1.5]
      -> Pruning, dann weiter auf OD -> od2conc            [Schritt 1.6]

Der entscheidende Punkt der Vorgabe: die Kanalqualitaet (dunkel/gesaettigt) wird an der
AMPLITUDE beurteilt, die Korrektur passiert aber auf OD. Deshalb der Umweg
OD -> Amplitude -> Maske -> zurueck auf OD. Die Baseline ist das, was diesen Rueckweg
ueberhaupt erlaubt: od = -log(amp / baseline), also amp = baseline * exp(-od).

Stand: Schritt 1.2 -- OD-Umrechnung mit Baseline-Rueckgabe. Die weiteren Stufen
kommen schrittweise dazu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

import cedalion
import cedalion.nirs
import cedalion.sigproc.motion as motion


@dataclass
class Preprocessed:
    """Ergebnis der Preprocessing-Kette samt Zwischenstufen fuer die Diagnose."""

    conc: xr.DataArray       # (time, channel, chromo) [µM], dequantifiziert
    od: xr.DataArray         # Optical Density, Stand nach Korrektur/Pruning
    amp_raw: xr.DataArray    # Rohamplitude [V], wie eingelesen (quantifiziert)
    baseline: xr.DataArray   # mittlere Rohamplitude (channel, wavelength) [V]
    geo3d: object            # Optodengeometrie (LabeledPoints)
    aux: object              # rec.aux_ts (Accelerometer/Gyroskop/dark signal)
    dropped: list[str] = field(default_factory=list)   # entfernte Kanaele
    masks: dict = field(default_factory=dict)          # Einzelmasken zur Diagnose
    motion_method: str = "none"                        # angewandte Motion Correction
    od_uncorrected: xr.DataArray | None = None         # OD vor der Korrektur (Diagnose)


def gate_positive(amp: xr.DataArray) -> tuple[xr.DataArray, list[str]]:
    """Verwirft Kanaele mit nicht-positiver Amplitude -- Vorbedingung fuer int2od.

    `int2od` bildet -log(amp/baseline) und bricht bei Werten <= 0 mit einer
    AssertionError ab. Physikalisch ist eine nicht-positive Lichtintensitaet ohnehin
    unmoeglich; solche Samples liegen unter dem Rauschboden des Detektors.

    Das ist bewusst KEINE methodische Vorentscheidung, sondern die minimale technische
    Vorbedingung: auf nn22 betrifft es 40 von 3.75 Mio. Samples (0.001 %) in 6 von 567
    Kanaelen, und alle 6 werden von der spaeteren mean_amp-Grenze (1e-3 V) ebenfalls
    verworfen -- die Vor-Maskierung nimmt also nichts weg, was sonst ueberlebt haette.
    Die eigentliche Qualitaetsbewertung passiert weiterhin erst nach der Motion
    Correction auf der korrigierten Amplitude (Betreuungsvorgabe).
    """
    ok = (amp > 0).all(dim=[d for d in amp.dims if d != "channel"])
    dropped = [str(c) for c in amp.channel.values[~ok.values]]
    return amp.sel(channel=ok), dropped


def to_od(amp: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Amplitude -> Optical Density, mit Baseline.

    Cedalion bietet das direkt an (nirs/cw.py): `int2od(amp, return_baseline=True)`
    liefert `(od, baseline)` mit `baseline = amp.mean("time")`. Der Parameter heisst
    `return_baseline`, NICHT `baseline=`.

    Die Baseline wird gebraucht, um nach der Motion Correction wieder auf die Amplitude
    zu kommen (`od2int(od, baseline)`), denn nur dort sind "dunkel" und "gesaettigt"
    ueberhaupt definierte Begriffe.
    """
    return cedalion.nirs.cw.int2od(amp, return_baseline=True)


MOTION_METHODS = ("none", "tddr", "wavelet", "tddr+wavelet")


def motion_correct(
    od: xr.DataArray,
    method: str = "tddr+wavelet",
    *,
    wavelet_iqr: float = 1.5,
    wavelet_name: str = "db2",
    wavelet_level: int = 4,
) -> xr.DataArray:
    """Motion Correction auf Optical Density.

    Die Betreuungsvorgabe nennt zwei Artefakttypen, die entfernt werden sollen:
    "scharfer Spike oder ruckartige Verschiebung". Genau darauf zielen die beiden
    Verfahren, und daher auch ihre Reihenfolge (identisch zu Cedalion NB 25:
    "apply TDDR first to correct jumps, then apply Wavelet motion artifact correction"):

      * `tddr`    -- Temporal Derivative Distribution Repair: robuste Regression auf der
                     zeitlichen Ableitung; faengt Baseline-Spruenge / ruckartige
                     Verschiebungen. Parameterfrei (`motion.tddr(ts)`).
      * `wavelet` -- verwirft Wavelet-Koeffizienten ausserhalb des IQR-Bandes; faengt
                     scharfe Spikes. Groesseres `iqr` = drastischere Korrektur;
                     `iqr < 0` laesst das Signal unveraendert.

    Beide arbeiten laut Cedalion ausdruecklich auf OD, nicht auf Amplitude oder
    Konzentration ("The correction algorithms operate on optical densities").
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


DRIFT_BANDS = (("Drift   <0.01 Hz", 0.0, 0.01),
               ("0.01-0.1 Hz     ", 0.01, 0.1),
               ("0.1-0.5 Hz      ", 0.1, 0.5),
               ("Kardial >0.5 Hz ", 0.5, np.inf))


def band_power_ratio(od_before: xr.DataArray, od_after: xr.DataArray) -> dict:
    """Leistung je Frequenzband NACH der Korrektur relativ zu VORHER (Median).

    Diagnose fuer die zentrale methodische Frage dieser Arbeit: greift die Motion
    Correction in das Driftband ein? Ein Verfahren, das unterhalb 0.01 Hz Leistung
    entfernt, nimmt genau den Anteil weg, den die Driftregressoren modellieren sollen --
    dann bestimmt die Vorverarbeitung das Ergebnis statt des Driftmodells, und der
    Familienvergleich wird verfaelscht (Betreuungshinweis 2026-07-11).

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


def to_conc(od: xr.DataArray, geo3d, dpf: float = 6.0) -> xr.DataArray:
    """Optical Density -> Haemoglobinkonzentration [µM], dequantifiziert."""
    dpf_da = xr.DataArray(
        [dpf] * od.sizes["wavelength"],
        dims="wavelength",
        coords={"wavelength": od.wavelength},
    )
    conc = cedalion.nirs.cw.od2conc(od, geo3d, dpf_da, spectrum="prahl")
    return conc.pint.to("uM").pint.dequantify()


def run(rec, *, motion_method: str = "tddr+wavelet", dpf: float = 6.0) -> Preprocessed:
    """Fuehrt die Preprocessing-Kette auf einem Recording aus.

    Args:
        rec: Cedalion-Recording (z.B. aus `cedalion.data.get_nn22_resting_state()`).
        motion_method: eines aus `MOTION_METHODS`. `"none"` schaltet die Korrektur ab --
            gebraucht fuer den A/B-Vergleich gegen die Motion-Regressoren in der
            Designmatrix (Betreuungshinweis: koennte sich doppeln).
        dpf: Differentieller Pfadlaengenfaktor fuer die modifizierte Beer-Lambert-Umrechnung.

    Returns:
        `Preprocessed` mit Konzentration, OD, Baseline und Diagnose-Zwischenstufen.
    """
    # nn22 liefert die Amplitude dimensionslos -> als Volt quantifizieren, damit die
    # spaeteren Amplitudengrenzen (dunkel/gesaettigt) eine physikalische Einheit haben.
    amp_raw = rec["amp"].pint.dequantify().pint.quantify("V")

    amp, dropped_nonpos = gate_positive(amp_raw)
    od_raw, baseline = to_od(amp)
    od = motion_correct(od_raw, motion_method)
    conc = to_conc(od, rec.geo3d, dpf)

    return Preprocessed(
        conc=conc,
        od=od,
        amp_raw=amp_raw,
        baseline=baseline,
        geo3d=rec.geo3d,
        aux=rec.aux_ts,
        dropped=list(dropped_nonpos),
        masks={"nonpositive": dropped_nonpos},
        motion_method=motion_method,
        od_uncorrected=od_raw,
    )


if __name__ == "__main__":
    import sys
    import time

    import cedalion.data

    method = sys.argv[1] if len(sys.argv) > 1 else "tddr+wavelet"
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

    print(f"\nMotion Correction  : {P.motion_method}   ({time.time() - t0:.1f}s gesamt)")
    if P.od_uncorrected is not None and method != "none":
        d1 = np.abs(np.diff(np.asarray(P.od_uncorrected.values, float), axis=-1))
        d2 = np.abs(np.diff(np.asarray(P.od.values, float), axis=-1))
        print(f"  groesster Sprung : {d1.max():.4f} -> {d2.max():.4f} OD "
              f"(Spitzen der zeitlichen Ableitung)")
        print("  Restleistung je Band (100 % = unveraendert):")
        for name, r in band_power_ratio(P.od_uncorrected, P.od).items():
            flag = "  <-- Driftband!" if name.startswith("Drift") and r < 0.9 else ""
            print(f"    {name} {100 * r:6.1f} %{flag}")
