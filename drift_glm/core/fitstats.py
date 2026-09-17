"""Guete des Modellfits: variance explained (R^2) und Residual-RMS je Kanal x Chromophor.

Residuum r(t) = y(t) - X @ beta_hat im Datenraum (nicht prewhitened, in µM);
resid_rms = sqrt(mean_t r^2); r2 = 1 - mean_t r^2 / var_t y;
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p) mit p = Spaltenzahl inkl. Offset, weil
Familien mit mehr Spalten (dct:0.02 bei langen Fenstern) R^2 mechanisch erhoehen.
Fuer Filter-Familien bezieht sich R^2 auf die gefilterte Zeitreihe und ist nicht direkt
mit den Regressor-Familien vergleichbar. Beide Groessen existieren auch auf realen Daten
ohne Ground Truth; der Sweep korreliert sie mit dem GT-Fehler (Spalte `resid_err_corr`).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import cedalion.models.glm as glm


def dequantify(ts: xr.DataArray) -> xr.DataArray:
    """Zeitreihe ohne pint-Einheit (Werte in µM); vertraegt beide Zustaende."""
    acc = getattr(ts, "pint", None)
    if acc is not None and acc.units is not None:
        return ts.pint.to("micromolar").pint.dequantify()
    return ts


def n_regressors(dm: glm.design_matrix.DesignMatrix) -> int:
    """Spaltenzahl der Designmatrix je Kanal (inkl. Offset).

    Kanalweise Regressoren (short_maxcorr/short_closest) zaehlen einmal pro Kanal.
    """
    p = int(dm.common.sizes["regressor"]) if dm.common is not None else 0
    for cw in dm.channel_wise:
        p += int(cw.sizes["regressor"])
    return p


def residuals(ts: xr.DataArray, betas: xr.DataArray,
              dm: glm.design_matrix.DesignMatrix) -> xr.DataArray:
    """Residuum y - X@beta_hat im Datenraum, Dims wie `ts` (unitless, µM-Skala).

    Ueber `glm.predict`, das auch kanalweise Designmatrizen verarbeitet.
    """
    y = dequantify(ts)
    pred = glm.predict(ts, betas, dm)
    return (y - pred).transpose(*y.dims)


def fit_metrics(ts: xr.DataArray, betas: xr.DataArray,
                dm: glm.design_matrix.DesignMatrix) -> xr.Dataset:
    """R^2, adjustiertes R^2 und Residual-RMS je (channel, chromo).

    Rueckgabe: Dataset mit `r2`, `r2_adj`, `resid_rms`; `n_samples` und `n_regressors`
    stehen in den attrs.
    """
    y = dequantify(ts)
    r = residuals(ts, betas, dm)
    ms_res = (r ** 2).mean("time")
    var_y = y.var("time")
    r2 = 1.0 - ms_res / var_y
    n = int(y.sizes["time"])
    p = n_regressors(dm)
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - p, 1)
    out = xr.Dataset(dict(r2=r2, r2_adj=r2_adj, resid_rms=np.sqrt(ms_res)))
    out.attrs["n_samples"] = n
    out.attrs["n_regressors"] = p
    return out
