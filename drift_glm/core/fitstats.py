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

import logging

import numpy as np
import xarray as xr

import cedalion.models.glm as glm

LOG = logging.getLogger(__name__)


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


def design_ranks(ts, dm, assert_full=True):
    """Numerical rank per channel/chromophore after column normalisation.

    Normalisation prevents units and polynomial magnitude from determining rank.
    Channel-wise regressors are checked in the actual computational groups used
    by Cedalion, rather than checking only the shared part of the design.
    """
    dim3 = next(d for d in dm.common.dims if d not in ("time", "regressor"))
    ranks = xr.DataArray(
        np.zeros((ts.sizes["channel"], ts.sizes[dim3]), dtype=int),
        dims=("channel", dim3),
        coords={"channel": ts.channel, dim3: ts[dim3]},
    )
    p = n_regressors(dm)
    for chromo, channels, design in dm.iter_computational_groups(ts):
        arr = np.asarray(design.pint.dequantify().transpose("time", "regressor"),
                         dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"Nonfinite design matrix for {chromo}, {channels}")
        scale = np.max(np.abs(arr), axis=0)
        scaled = np.divide(arr, scale, out=np.zeros_like(arr), where=scale != 0)
        norm = np.linalg.norm(scaled, axis=0)
        scaled = np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm != 0)
        rank = int(np.linalg.matrix_rank(scaled))
        ranks.loc[{"channel": channels, dim3: chromo}] = rank
        if assert_full and rank != p:
            raise ValueError(f"Design rank {rank} < {p} columns for {chromo}, "
                             f"channels {list(channels)}; regressors={dm.regressors}")
    LOG.info("Design rank %d..%d; columns=%d", int(ranks.min()), int(ranks.max()), p)
    return ranks


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

    Rueckgabe: Dataset mit `r2`, `r2_adj`, `resid_rms`, `design_rank`;
    `n_samples` und `n_regressors` stehen in den attrs. Raw-space R² and its
    column-count adjustment are descriptive under AR-IRLS, not whitened-space
    likelihood measures or an estimate of robust effective degrees of freedom.
    """
    ranks = design_ranks(ts, dm, assert_full=True)
    y = dequantify(ts)
    r = residuals(ts, betas, dm)
    ms_res = (r ** 2).mean("time")
    var_y = y.var("time")
    r2 = 1.0 - ms_res / var_y
    n = int(y.sizes["time"])
    p = n_regressors(dm)
    if n <= p:
        raise ValueError(f"Adjusted R² requires n > p; received n={n}, p={p}")
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p)
    out = xr.Dataset(dict(r2=r2, r2_adj=r2_adj, resid_rms=np.sqrt(ms_res),
                          design_rank=ranks))
    out.attrs["n_samples"] = n
    out.attrs["n_regressors"] = p
    out.attrs["design_rank_min"] = int(ranks.min())
    out.attrs["design_rank_max"] = int(ranks.max())
    out.attrs["r2_interpretation"] = "descriptive raw-space fit; not AR-IRLS effective df"
    return out
