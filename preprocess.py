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
import cedalion.sigproc.quality as quality
from cedalion import units


@dataclass
class Preprocessed:
    """Ergebnis der Preprocessing-Kette samt Zwischenstufen fuer die Diagnose."""

    conc: xr.DataArray       # (time, channel, chromo) [µM], dequantifiziert
    od: xr.DataArray         # Optical Density, Stand nach Korrektur/Pruning
    amp_raw: xr.DataArray    # Rohamplitude [V], wie eingelesen (quantifiziert)
    amp_corr: xr.DataArray   # Amplitude NACH Motion Correction [V] -- Basis der Masken
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


def to_amp(od: xr.DataArray, baseline: xr.DataArray) -> xr.DataArray:
    """Optical Density -> Amplitude, mit der beim Hinweg gemerkten Baseline.

    Das ist der Kern der Betreuungsvorgabe "erst motion correction, dann zu amplitude
    umwandeln und dann schlechte channels markieren": korrigiert wird auf OD, bewertet
    wird auf der Amplitude. `od2int(od, baseline)` = `baseline * exp(-od)` ist die exakte
    Umkehrung von `int2od` -- ohne die Baseline waere der Rueckweg nicht eindeutig, weil
    OD nur relative Aenderungen gegenueber dem eigenen Mittel kodiert.

    Wichtig: die Baseline stammt aus der UNKORRIGIERTEN Amplitude. Die zurueckgerechnete
    Amplitude traegt also die Motion-Korrektur, behaelt aber das urspruengliche
    Helligkeitsniveau -- genau das, worauf "dunkel" und "gesaettigt" sich beziehen.
    """
    return cedalion.nirs.cw.od2int(od, baseline)


def quality_masks(
    amp: xr.DataArray,
    geo3d,
    *,
    snr_threshold: float = 3.0,
    amp_range: tuple[float, float] = (1e-3, 0.84),
    sd_range: tuple[float, float] = (0.0, 4.5),
) -> dict[str, xr.DataArray]:
    """Qualitaetsmasken auf der (korrigierten) Amplitude. CLEAN = True.

    Die drei Kriterien adressieren verschiedene Defekte und ersetzen einander nicht:

      * `snr`      -- Verhaeltnis Mittelwert/Streuung ueber die Zeit. Faengt verrauschte
                      Kanaele. Betreuungsvorgabe: Schwelle 3 (bisher 10). Der neue Wert
                      ist PERMISSIVER; die eigentliche Arbeit macht jetzt `mean_amp`.
      * `mean_amp` -- mittlere Amplitude innerhalb eines Fensters. Faengt DUNKLE (zu wenig
                      Licht, Rauschen dominiert) und GESAETTIGTE Kanaele (Detektor am
                      Anschlag, Signal geklippt). Vorgabe: NinjaNIRS-Grenzen
                      1e-3 .. 0.84 V. Basiert auf Homer3 `hmR_PruneChannels.m`.
      * `sd_dist`  -- Quell-Detektor-Abstand innerhalb eines Bereichs.

    Gesaettigte Kanaele sind besonders heimtueckisch: durch das Klippen wirken sie
    RAUSCHARM, weshalb varianzbasierte Metriken sie nicht erkennen (Cedalion NB 24:
    "the metric cannot account for saturation"). Bei der Image Reconstruction bekommen
    sie deshalb maximales Gewicht in der Pseudoinversen und schmieren ihren Fehler ueber
    ihr gesamtes Sensitivitaetsprofil -- daher die Betreuungsvorgabe, sie spaetestens
    dort zwingend zu entfernen.
    """
    _, snr_mask = quality.snr(amp, snr_threshold)
    _, amp_mask = quality.mean_amp(amp, (amp_range[0] * units.V,
                                         amp_range[1] * units.V))
    _, sd_mask = quality.sd_dist(amp, geo3d, (sd_range[0] * units.cm,
                                              sd_range[1] * units.cm))
    return {"snr": snr_mask, "mean_amp": amp_mask, "sd_dist": sd_mask}


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


def run(
    rec,
    *,
    motion_method: str = "tddr+wavelet",
    snr_threshold: float = 3.0,
    amp_range: tuple[float, float] = (1e-3, 0.84),
    sd_range: tuple[float, float] = (0.0, 4.5),
    dpf: float = 6.0,
) -> Preprocessed:
    """Fuehrt die Preprocessing-Kette auf einem Recording aus.

    Args:
        rec: Cedalion-Recording (z.B. aus `cedalion.data.get_nn22_resting_state()`).
        motion_method: eines aus `MOTION_METHODS`. `"none"` schaltet die Korrektur ab --
            gebraucht fuer den A/B-Vergleich gegen die Motion-Regressoren in der
            Designmatrix (Betreuungshinweis: koennte sich doppeln).
        snr_threshold: SNR-Schwelle (Betreuungsvorgabe: 3).
        amp_range: (dunkel, gesaettigt) in Volt (NinjaNIRS-Vorgabe: 1e-3 .. 0.84).
        sd_range: zulaessiger Quell-Detektor-Abstand in cm.
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
    # Zurueck zur Amplitude: dort -- und nur dort -- sind "dunkel" und "gesaettigt"
    # definiert. Die Qualitaetsmasken (Schritt 1.5) setzen auf amp_corr auf.
    amp_corr = to_amp(od, baseline)

    masks = quality_masks(amp_corr, rec.geo3d, snr_threshold=snr_threshold,
                          amp_range=amp_range, sd_range=sd_range)
    conc = to_conc(od, rec.geo3d, dpf)

    return Preprocessed(
        conc=conc,
        od=od,
        amp_raw=amp_raw,
        amp_corr=amp_corr,
        baseline=baseline,
        geo3d=rec.geo3d,
        aux=rec.aux_ts,
        dropped=list(dropped_nonpos),
        masks={"nonpositive": dropped_nonpos, **masks},
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

    # Rueckweg-Kontrolle: ohne Korrektur muss od2int(int2od(amp)) == amp gelten.
    # Damit ist belegt, dass die Baseline den Rueckweg exakt traegt.
    amp_in = gate_positive(P.amp_raw)[0].pint.dequantify().values
    amp_out = P.amp_corr.pint.dequantify().values
    rel = np.abs(amp_out - amp_in) / np.abs(amp_in)
    hi, med = float(np.nanmax(rel)), float(np.nanmedian(rel))
    if hi < 1e-9:
        print(f"\nRueckweg OD->Amp   : max. rel. Abweichung {hi:.2e} -- exakt "
              f"(Baseline traegt den Rueckweg verlustfrei)")
    else:
        # Erwartet, sobald korrigiert wurde: die Abweichung IST die Korrektur.
        # Im Amplitudenraum wirkt sie exponentiell (amp = baseline * exp(-od)),
        # eine OD-Aenderung von d entspricht dem Faktor exp(d) -- das Maximum wird
        # daher von einzelnen Spikes in dunklen Kanaelen dominiert.
        print(f"\nRueckweg OD->Amp   : median {100 * med:.2f} %, max {hi:.2e} "
              f"(entspricht {np.log(hi + 1):.1f} OD) -- das ist die Korrektur selbst")

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

    print("\nQualitaetsmasken auf der korrigierten Amplitude (CLEAN = True):")
    n_ch = P.amp_corr.sizes["channel"]
    keep_all = None
    for key in ("snr", "mean_amp", "sd_dist"):
        m = P.masks[key]
        # ein Kanal ueberlebt nur, wenn er in ALLEN uebrigen Dims (z.B. beide
        # Wellenlaengen) sauber ist -- dieselbe Logik wie in xrutils.apply_mask.
        keep = m.all(dim=[d for d in m.dims if d != "channel"])
        keep_all = keep if keep_all is None else (keep_all & keep)
        print(f"  {key:9s}: {int(keep.sum()):4d} / {n_ch} behalten "
              f"({n_ch - int(keep.sum())} verworfen)")
    print(f"  {'kombiniert':9s}: {int(keep_all.sum()):4d} / {n_ch} behalten "
          f"({n_ch - int(keep_all.sum())} verworfen)")
