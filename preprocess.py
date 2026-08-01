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


def to_conc(od: xr.DataArray, geo3d, dpf: float = 6.0) -> xr.DataArray:
    """Optical Density -> Haemoglobinkonzentration [µM], dequantifiziert."""
    dpf_da = xr.DataArray(
        [dpf] * od.sizes["wavelength"],
        dims="wavelength",
        coords={"wavelength": od.wavelength},
    )
    conc = cedalion.nirs.cw.od2conc(od, geo3d, dpf_da, spectrum="prahl")
    return conc.pint.to("uM").pint.dequantify()


def run(rec, *, dpf: float = 6.0) -> Preprocessed:
    """Fuehrt die Preprocessing-Kette auf einem Recording aus.

    Args:
        rec: Cedalion-Recording (z.B. aus `cedalion.data.get_nn22_resting_state()`).
        dpf: Differentieller Pfadlaengenfaktor fuer die modifizierte Beer-Lambert-Umrechnung.

    Returns:
        `Preprocessed` mit Konzentration, OD, Baseline und Diagnose-Zwischenstufen.
    """
    # nn22 liefert die Amplitude dimensionslos -> als Volt quantifizieren, damit die
    # spaeteren Amplitudengrenzen (dunkel/gesaettigt) eine physikalische Einheit haben.
    amp_raw = rec["amp"].pint.dequantify().pint.quantify("V")

    amp, dropped_nonpos = gate_positive(amp_raw)
    od, baseline = to_od(amp)
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
    )


if __name__ == "__main__":
    import cedalion.data

    P = run(cedalion.data.get_nn22_resting_state())
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
