"""Short-Channel-Regression: Long/Short-Split und Regressor-Varianten.

Ein Kanal mit kleinem Quell-Detektor-Abstand misst nur Kopfhaut und Schaedel, also die
systemische Physiologie; als Regressor in der Designmatrix nimmt er diesen Anteil aus den
langen Kanaelen heraus. nn22 hat keine echten Short-Separation-Kanaele (<10 mm), der
kuerzeste Abstand ist 15.55 mm; die Schwelle 1.8 cm folgt der Konstruktion aus
Cedalion-Workshop-Notebook 36 (22.5 mm auf fingertappingDOT). Kanaele mit 15.5-18 mm
sehen noch etwas Kortex, der Regressor kann also auch Hirnsignal entfernen.

Aufruf:  conda run -n cedalion python -m drift_glm.core.shortchannel
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import cedalion
import cedalion.nirs
from cedalion import units

SHORT_THRESHOLD = 1.8 * units.cm     # Grenze lang/kurz; kuerzester Abstand in nn22: 15.55 mm


def distances_mm(ts: xr.DataArray, geo3d) -> np.ndarray:
    """Quell-Detektor-Abstaende je Kanal in mm (dequantifiziert)."""
    d = cedalion.nirs.channel_distances(ts, geo3d).pint.to("mm")
    return np.asarray(d.pint.dequantify().values, dtype=float)


def split(ts: xr.DataArray, geo3d, threshold=SHORT_THRESHOLD):
    """Zerlegt eine Zeitreihe in lange und kurze Kanaele; Rueckgabe (long, short).

    `threshold` muss eine pint-Laenge sein.
    """
    return cedalion.nirs.split_long_short_channels(ts, geo3d, distance_threshold=threshold)


def report(ts: xr.DataArray, geo3d, thresholds_cm=(1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2)) -> dict:
    """Kennzahlen zur Schwellenwahl: Anzahl kurzer Kanaele je Schwelle."""
    d = distances_mm(ts, geo3d)
    return {
        "n_channels": int(d.size),
        "min_mm": float(d.min()),
        "median_mm": float(np.median(d)),
        "max_mm": float(d.max()),
        "counts": {c: int((d < c * 10.0).sum()) for c in thresholds_cm},
        "distances_mm": d,
    }


VARIANTS = ("short_avg", "short_maxcorr", "short_closest")


def short_dm(variant: str, ts_long: xr.DataArray, ts_short: xr.DataArray, geo3d):
    """Short-Channel-Regressor als DesignMatrix.

    `short_avg`: Mittel ueber alle kurzen Kanaele, ein gemeinsamer `common`-Regressor.
    `short_maxcorr`: je langem Kanal der am staerksten korrelierende kurze Kanal (Maximum
    ueber die Chromophore, damit HbO und HbR denselben bekommen). `short_closest`: je
    langem Kanal der raeumlich naechste kurze Kanal (Cedalion NB 32/34/35). Die letzten
    beiden liefern kanalweise Designmatrizen (`channel_wise`) mit dem Regressornamen
    "short"; `glm.fit` gruppiert dabei intern nach `comp_group`.
    """
    import cedalion.models.glm as glm

    dmx = glm.design_matrix
    if variant == "short_avg":
        return dmx.average_short_channel_regressor(ts_short)
    if variant == "short_maxcorr":
        return dmx.max_corr_short_channel_regressor(ts_long, ts_short)
    if variant == "short_closest":
        return dmx.closest_short_channel_regressor(ts_long, ts_short, geo3d)
    raise ValueError(f"Unbekannte Short-Channel-Variante: {variant!r} "
                     f"(erlaubt: {VARIANTS})")


GLOBAL_COMP_MODES = ("none", "dm", "subtract")


def subtract_global_component(ts_long, ts_short, stim_df, basis, *,
                              variant: str = "short_avg", geo3d=None,
                              noise_model: str = "ols"):
    """Den vom Short-Regressor erklaerten Anteil abziehen statt ihn im Modell zu lassen.

    Variante aus Notebook 50b. Beim Regressor in der Designmatrix (`dm`) wird die HRF
    gemeinsam mit dem Short-Regressor geschaetzt, also gegen ihn orthogonalisiert. Hier
    wird der Anteil auf Daten geschaetzt, die die HRF enthalten; ist die Systemik
    aufgabengekoppelt, nimmt der Abzug einen Teil der Antwort mit, den ein spaeterer Fit
    nicht zurueckholen kann. Der Vorteil ist praktisch: die bereinigte Zeitreihe laesst
    sich ohne GLM weiterverarbeiten (Blockmittel, Epochen, Bildraum).

    Rueckgabe: `ts_long` minus dem erklaerten Anteil.
    """
    import cedalion.models.glm as glm

    dm = (glm.design_matrix.hrf_regressors(ts_long, stim_df, basis)
          & short_dm(variant, ts_long, ts_short, geo3d))
    res = glm.fit(ts_long, dm, noise_model=noise_model)
    comp = glm.predict(ts_long, res.sm.params.sel(regressor=["short"]), dm)
    comp = comp.transpose(*ts_long.dims)
    # Einheiten angleichen: to_conc liefert dequantifiziert, 50b arbeitet quantifiziert.
    u = getattr(ts_long, "pint", None)
    if u is not None and ts_long.pint.units is not None and comp.pint.units is None:
        comp = comp.pint.quantify(ts_long.pint.units)
    return ts_long - comp


if __name__ == "__main__":
    import cedalion.data

    from drift_glm.core import preprocess as prep

    P = prep.run(cedalion.data.get_nn22_resting_state())
    r = report(P.conc, P.geo3d)

    print(f"Kanaele nach Preprocessing : {r['n_channels']}")
    print(f"Abstaende [mm]             : min {r['min_mm']:.2f}  "
          f"median {r['median_mm']:.2f}  max {r['max_mm']:.2f}")
    print("\nKurze Kanaele je Schwelle:")
    for cm, n in r["counts"].items():
        mark = "   <-- Default" if abs(cm - 1.8) < 1e-9 else ""
        print(f"  < {cm:.1f} cm : {n:4d} kurz / {r['n_channels'] - n:4d} lang{mark}")

    ts_long, ts_short = split(P.conc, P.geo3d)
    print(f"\nSplit bei {SHORT_THRESHOLD}: {ts_short.sizes['channel']} kurz, "
          f"{ts_long.sizes['channel']} lang")

    d = r["distances_mm"]
    short = np.sort(d[d < 18.0])
    print(f"Abstaende der kurzen Kanaele: {short.min():.2f} .. {short.max():.2f} mm "
          f"(Median {np.median(short):.2f})")
    print("\nKeine echten Short-Separation-Kanaele (<10 mm); sie sehen noch etwas Kortex.")
