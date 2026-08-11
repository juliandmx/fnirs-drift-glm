"""Demo: Augmentations- und GLM-Pipeline mit bekannter Ground Truth.

Schritt 2 der Bachelorarbeit -- die komplette Verarbeitungskette einmal
end-to-end validieren. Die HRF wird mit realistischer Peak-Amplitude (~µM) in
echte Ruhedaten eingespeist; geschaetztes beta wird gegen die Ground Truth
verglichen, getrennt fuer HbO und HbR.

Wichtig: Die ausgegebenen Kennzahlen sind ueber KANAELE einer EINZELNEN
Realisierung aggregiert -- das ist noch KEINE Monte-Carlo-Bias/Varianz-
Schaetzung. Letztere folgt im eigentlichen Sweep (Schleife ueber mehrere Seeds).

Siehe pipeline.py fuer die gemeinsame Pipeline.

Aufruf:
    conda run -n cedalion python -m drift_glm.reports.demo_recovery
"""

from __future__ import annotations

import numpy as np

import cedalion.models.glm as glm
from drift_glm.core import pipeline as pl


def main() -> None:
    print("[1-6] Baue augmentierte Pipeline (Ruhedaten + bekannte HRF) ...")
    P = pl.build()
    print(f"      conc: {dict(P.conc.sizes)}  (µM)")
    print(f"      Impulse: {len(P.stim_df)} | Regressoren: "
          f"{list(P.dm_full.common.regressor.values)}")
    print(f"      Ground-Truth-Peak: HbO={P.beta_true['HbO']:+.3f} µM, "
          f"HbR={P.beta_true['HbR']:+.3f} µM")
    print(f"      eingespeister HbO-Peak real: "
          f"{float(P.activation.sel(chromo='HbO').max()):.3f} µM "
          f"(vor Normierung: "
          f"{P.beta_true['HbO'] * P.raw_hrf_peak[P.hrf_names[0]]:.2f} µM)")

    # raeumlicher Blob: nur die Kern-Kanaele tragen Aktivierung (GT variiert je Kanal)
    active = np.abs(P.beta_true_map.sel(chromo="HbO")) > 0.1 * abs(P.beta_true["HbO"])
    print(f"      raeumlicher Blob: {int(active.sum())} aktive Kanaele "
          f"(|GT_HbO| > 10% Peak) von {P.conc.sizes['channel']}")

    main_hrf = P.hrf_names[0]
    print("\n[7] Schaetze GLM (OLS & AR-IRLS) ...")
    results = {}
    for nm in ("ols", "ar_irls"):
        # ar_order ist der einzige AR-Ordnungs-Parameter von glm.fit und wird
        # intern als pmax=ar_order an ar_irls_GLM weitergereicht. glm.fit hat
        # KEIN pmax-Argument; die fs-basierte Auto-Wahl (pmax=None) ist hier
        # nicht erreichbar -- ar_irls laeuft mit fester AR-Ordnung (hier 30).
        results[nm] = glm.fit(P.conc_syn, P.dm_full, noise_model=nm,
                              ar_order=30, max_jobs=-1).sm.params

    print(f"\n[8] Kennzahlen ueber die {int(active.sum())} AKTIVEN Kanaele (Blob-Kern) "
          f"fuer Regressor '{main_hrf}' (EINE Realisierung -- keine MC-Schaetzung):")
    for c in P.chromo:
        bt = P.beta_true_map.sel(chromo=c).where(active, drop=True)  # per-Kanal-GT
        print(f"\n  --- {c}  (GT-Peak = {P.beta_true[str(c)]:+.3f} µM, "
              f"median GT aktiv = {float(bt.median()):+.3f}) ---")
        for nm in ("ols", "ar_irls"):
            est = results[nm].sel(regressor=main_hrf, chromo=c).where(active, drop=True)
            err = est - bt
            print(f"    {nm:>8}:  mean_ch(beta_hat) = {float(est.mean()):+.4f} µM"
                  f"  | mean_ch(err) = {float(err.mean()):+.4f}"
                  f"  | RMSE_ch = {float(np.sqrt((err ** 2).mean())):.4f}"
                  f"  | std_ch = {float(est.std()):.4f}")

    print("\n[OK] PoC lief end-to-end. Naechster Schritt: dieselbe Schleife ueber "
          "Driftfamilien (poly / cosine / legendre), Fensterlaengen UND mehrere "
          "Seeds iterieren (echte Monte-Carlo-Bias/Varianz).")


if __name__ == "__main__":
    main()
