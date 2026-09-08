"""Guete des Modellfits: variance explained (R^2) und Residual-Masse.

Beide Groessen beantworten den Punkt aus dem Betreuungsgespraech (Notizen 2026-09-08):
"Residuals vom Modelfit -> was konnte das Modell nicht fitten?" -- als Mass fuer den
Modellfit, das AUCH auf realen Daten ohne Ground Truth existiert. Auf simulierten Daten
kommt die pruefbare Erwartung dazu: je kleiner die Residuen, desto kleiner sollte die
Abweichung von der Ground Truth sein. Genau diese Korrelation berechnet der Sweep
(`sweep._aggregate_and_export`, Spalte `resid_err_corr`).

Definitionen (je Kanal x Chromophor, ueber die Zeit):

  Residuum    r(t) = y(t) - X @ beta_hat  -- im DATENraum, nicht prewhitened. AR-IRLS
              arbeitet intern mit gewichteten Residuen; fuer die Frage "was blieb
              unerklaert?" zaehlt aber die interpretierbare Groesse in µM.
  resid_rms   sqrt( mean_t r(t)^2 )                     [µM]
  r2          1 - mean_t r(t)^2 / var_t y(t)            (variance explained)
  r2_adj      1 - (1 - r2) * (n - 1) / (n - p)          (p = Spaltenzahl der
              Designmatrix INKL. Offset). Relevant, weil die Driftfamilien
              verschieden viele Spalten haben: dct:0.02 hat bei langen Fenstern ein
              Vielfaches von poly:1, und mehr Spalten erhoehen R^2 mechanisch.

Vergleichbarkeits-Hinweis: R^2 bezieht sich auf die Zeitreihe, die der Fit tatsaechlich
gesehen hat. Fuer die Filter-Familien (butter/lowpass/bandpass) ist das die GEFILTERTE
Zeitreihe -- der Filter hat einen Teil der Varianz bereits entfernt, ihr R^2 ist also
nicht direkt mit dem der Regressor-Familien vergleichbar. In Abbildungen ausweisen.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import cedalion.models.glm as glm


def dequantify(ts: xr.DataArray) -> xr.DataArray:
    """Zeitreihe ohne pint-Einheit (Werte in µM). Vertraegt beide Zustaende."""
    acc = getattr(ts, "pint", None)
    if acc is not None and acc.units is not None:
        return ts.pint.to("micromolar").pint.dequantify()
    return ts


def n_regressors(dm: glm.design_matrix.DesignMatrix) -> int:
    """Spaltenzahl der Designmatrix je Kanal (inkl. Offset).

    Kanalweise Regressoren (short_maxcorr/short_closest) zaehlen einmal pro Kanal --
    jeder Kanal sieht genau eine Spalte je channel_wise-Matrix.
    """
    p = int(dm.common.sizes["regressor"]) if dm.common is not None else 0
    for cw in dm.channel_wise:
        p += int(cw.sizes["regressor"])
    return p


def residuals(ts: xr.DataArray, betas: xr.DataArray,
              dm: glm.design_matrix.DesignMatrix) -> xr.DataArray:
    """Residuum y - X@beta_hat im Datenraum, Dims wie `ts` (unitless, µM-Skala).

    Schaetzerunabhaengig ueber `glm.predict` (verarbeitet auch kanalweise
    Designmatrizen), statt auf statsmodels-Interna der Ergebnisobjekte zu bauen.
    """
    y = dequantify(ts)
    pred = glm.predict(ts, betas, dm)
    return (y - pred).transpose(*y.dims)


def fit_metrics(ts: xr.DataArray, betas: xr.DataArray,
                dm: glm.design_matrix.DesignMatrix) -> xr.Dataset:
    """R^2, adjustiertes R^2 und Residual-RMS je (channel, chromo).

    Rueckgabe: Dataset mit `r2`, `r2_adj`, `resid_rms` (float64). `n` (Samples) und
    `p` (Regressorspalten) stehen in den attrs.
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
