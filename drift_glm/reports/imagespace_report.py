"""Abbildungen zum Bildraum und zum Multisubject-Datensatz (Abb. 19-23).

Getrennt von `sweep_report.py` und `realglm_report.py`, weil hier eine andere Datenquelle
zugrunde liegt: `results/imageglm_summary.csv` und `results/msglm_summary.csv`.

Aufruf:
    conda run -n cedalion python -m drift_glm.reports.imagespace_report            # alles, was da ist
    conda run -n cedalion python -m drift_glm.reports.imagespace_report hrf        # nur Abb. 19
    conda run -n cedalion python -m drift_glm.reports.imagespace_report cortex     # nur Abb. 20
    conda run -n cedalion python -m drift_glm.reports.imagespace_report tables     # nur Abb. 21-23
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

from drift_glm import paths       # noqa: E402

RESULTS = paths.RESULTS
FIGURES = paths.FIGURES

HBO_COLOR, HBR_COLOR = "#c44e52", "#4c72b0"
PROJ_LABEL = {"hrf": "geschätzte HRF", "residual": "Residuum",
              "cleaned": "Residuum + HRF"}
SYS_LABEL = {"none": "ohne", "global_dm": "global (DM)", "short_avg_dm": "short avg (DM)",
             "short_avg_sub": "short avg (abgezogen)",
             "short_maxcorr_dm": "short maxcorr (DM)",
             "short_closest_dm": "short closest (DM)"}


# ---------------------------------------------------------------- Abb. 19: HRF je Kanal

def fig_hrf_per_channel(window_s: float = 368.0, family: str = "dct:0.02",
                        noise_model: str = "ar_irls", n_channels: int = 12,
                        seed: int = 0):
    """Abb. 19 -- je Kanal die geschaetzte gegen die eingemischte HRF.

    Betreuungsnotiz "pro channel die hrf betrachten". Gezeigt werden die staerksten Kanaele
    und zum Kontrast drei ohne Aktivierung: dort MUSS die Schaetzung flach sein, und wenn
    sie es nicht ist, ist das ein Falsch-Positiv-Beleg, den kein Mittelwert sichtbar macht.
    """
    from drift_glm.analysis import imageglm as ig
    from drift_glm.core import pipeline as pl
    from drift_glm.core import shortchannel as sc
    import cedalion.models.glm as glm
    from cedalion import units
    from drift_glm.analysis.sweep import drift_dm

    P = pl.build(window_s=window_s, seed=seed, activation_space="image")
    ts = P.conc_syn
    dm_drift, _ = drift_dm(family, ts)
    dm_hrf = glm.design_matrix.hrf_regressors(
        ts, P.stim_df, glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=0 * units.s))
    names = [r for r in dm_hrf.common.regressor.values if str(r).startswith("HRF")]
    dm_hrf, _ = pl._normalize_hrf_to_unit_peak(dm_hrf, names)
    res = glm.fit(ts, dm_hrf & dm_drift, noise_model=noise_model)
    beta = res.sm.params.sel(regressor=names)
    hrf_hat = ig.with_time_coords(
        glm.predict(ts, beta, dm_hrf).transpose(*ts.dims), ts)

    tts = [str(t) for t in pd.unique(P.stim_df.trial_type)]

    def blocks(x):
        ep = x.cd.to_epochs(P.stim_df, tts, before=ig.EPOCH_BEFORE,
                            after=ig.EPOCH_AFTER)
        ep = ep - ep.sel(reltime=(ep.reltime < 0)).mean("reltime")
        ba = ep.groupby("trial_type").mean("epoch")
        return ba.mean("trial_type") if "trial_type" in ba.dims else ba

    ba_hat = blocks(hrf_hat)
    ba_true = blocks(ig.with_time_coords(P.activation, ts))

    bt = P.beta_true_map
    if "trial_type" in bt.dims:
        bt = bt.max("trial_type")
    strength = np.abs(np.asarray(bt.sel(chromo="HbO").values, float))
    order = np.argsort(strength)[::-1]
    pick = list(order[: n_channels - 3]) + list(order[-3:])

    n = len(pick)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.2 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for k, j in enumerate(pick):
        ax = axes[k]
        ch = str(bt.channel.values[j])
        for chromo, col in (("HbO", HBO_COLOR), ("HbR", HBR_COLOR)):
            ax.plot(ba_true.reltime, ba_true.sel(channel=ch, chromo=chromo),
                    color=col, lw=2.4, alpha=0.45)
            ax.plot(ba_hat.reltime, ba_hat.sel(channel=ch, chromo=chromo),
                    color=col, lw=1.2, ls="--")
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_title(f"{ch}   wahr {strength[j]:.3f} µM",
                     fontsize=8, color="k" if strength[j] > 0.05 else "#888888")
        ax.grid(alpha=0.25)
    for k in range(n, len(axes)):
        axes[k].set_axis_off()
    axes[0].set_ylabel(r"$\Delta c$ / µM")
    fig.suptitle("Abb. 19 – HRF je Kanal: eingemischt (dick, blass) gegen geschätzt "
                 "(dünn, gestrichelt)\n"
                 f"rot HbO, blau HbR · {family}, {noise_model}, {window_s:.0f} s · "
                 "die letzten drei Kanäle haben keine Aktivierung (Kontrolle)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGURES / "19_hrf_per_channel.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"-> {out}")


# ------------------------------------------------------- Abb. 20: Kortex, wahr vs. rekon

def fig_cortex(window_s: float = 368.0, family: str = "dct:0.02",
               noise_model: str = "ar_irls", seed: int = 0):
    """Abb. 20 -- eingemischte und rekonstruierte Aktivierung auf dem Kortex.

    Die Abbildung, die den Bildraum ueberhaupt rechtfertigt: sie zeigt den ORT. Links die
    Wahrheit (Bloebe unter C3 und C4), rechts, was aus dem GLM-Ergebnis zurueckkommt --
    je Projektion eine Spalte, je Hemisphaere eine Zeile.
    """
    import cedalion.vis.anatomy as vis
    from drift_glm.analysis import imageglm as ig
    from drift_glm.core import imagespace as ims
    from drift_glm.core import pipeline as pl

    P = pl.build(window_s=window_s, seed=seed, activation_space="image")
    head = ims.head()
    A = ims.adot(P.dataset).sel(channel=[str(c) for c in P.pre.od.channel.values])
    recon, c_meas = ims.recon_operator(A, P.pre.od)
    sens = ims.sensitivity_mask(A)

    truth = P.beta_true_img
    if "trial_type" in truth.dims:
        truth = truth.sum("trial_type")

    pr = ig.fit_projections(P, family, "baseline", noise_model)
    imgs = {"Wahrheit": truth}
    for proj in ig.PROJECTIONS:
        od = ig.conc_map_to_od(pr[proj], P.geo3d, P.pre.od.wavelength, like=P.conc_syn)
        imgs[PROJ_LABEL[proj]] = recon.reconstruct(od, c_meas.sel(channel=od.channel))

    # Unsichtbare Vertices auf NaN setzen statt mitzufaerben -- dort ist der Wert reine
    # Regularisierung (siehe imagespace.sensitivity_mask); plot_brain_in_axes stellt NaN
    # ueber `bad_color` grau dar. Bewusst als DataArray OHNE Einheit: die Funktion ruft
    # `metric.pint.dequantify()` auf, und eine Mischung aus quantifizierten µM und einem
    # nackten numpy-Array wuerde an der Einheitenpruefung von pint scheitern.
    def masked(img):
        v = img.sel(chromo="HbO")
        if getattr(v, "pint", None) is not None and v.pint.units is not None:
            v = v.pint.to("uM").pint.dequantify()
        a = np.asarray(v.values, dtype=float).copy()
        a[~sens] = np.nan
        return v.copy(data=a)

    cols = list(imgs)
    fig, axes = plt.subplots(2, len(cols), figsize=(3.6 * len(cols), 7.2))
    for i_col, name in enumerate(cols):
        m = masked(imgs[name])
        lim = float(np.nanpercentile(np.abs(m.values), 99.5))
        if not np.isfinite(lim) or lim == 0.0:
            lim = 1.0
        for i_row, cam in enumerate(("C3", "C4")):
            vis.plot_brain_in_axes(
                P.pre.od, head.landmarks, m, head.brain, axes[i_row, i_col],
                camera_pos=cam, cmap="RdBu_r", vmin=-lim, vmax=+lim,
                cb_label=r"$\Delta$ HbO / µM", title=None)
            axes[i_row, i_col].set_title(f"{name} · Blick von {cam}", fontsize=9)
    fig.suptitle("Abb. 20 – Aktivierung auf dem Kortex: eingemischt und aus dem "
                 f"GLM-Ergebnis rekonstruiert ({family}, {noise_model})\n"
                 "graue Bereiche sieht die Montage nicht – dort ist jeder Wert reine "
                 "Regularisierung", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGURES / "20_cortex_truth_vs_recon.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"-> {out}")


# ------------------------------------------------- Abb. 21: Driftfamilien im Bildraum

def fig_image_families(csv: str = "imageglm_summary.csv"):
    """Abb. 21 -- Driftfamilien im Bildraum: Form, Ort, Trefferanteil.

    Die rauschfreie Obergrenze (`truth_ref`) wird als waagerechte Linie eingezeichnet, nicht
    als weiterer Balken: sie ist keine Schaetzung, sondern das Beste, was die Rekonstruktion
    ueberhaupt liefern kann. Ohne diese Linie ist kein Balken interpretierbar.
    """
    path = RESULTS / csv
    if not path.exists():
        print(f"(uebersprungen: {path} fehlt -- 'python -m drift_glm.analysis.imageglm' laufen lassen)")
        return
    df = pd.read_csv(path)
    hbo = df[df.chromo == "HbO"]
    ref = hbo[hbo.projection == "truth_ref"]
    d = hbo[hbo.projection != "truth_ref"]
    if d.empty:
        print(f"(uebersprungen: {path} enthaelt nur die Obergrenze)")
        return

    metrics = [("img_r", "Korrelation mit der Wahrheit", "höher = besser"),
               ("img_loc_err_mm", "Lokalisationsfehler [mm]", "niedriger = besser"),
               ("img_hit_frac", "Anteil am richtigen Ort", "höher = besser")]
    projs = [p for p in ("hrf", "residual", "cleaned") if p in set(d.projection)]
    cons = list(dict.fromkeys(d.constellation))
    fams = list(dict.fromkeys(d.family))
    x = np.arange(len(fams))
    w = 0.8 / max(len(projs), 1)

    fig, axes = plt.subplots(len(metrics), len(cons), sharex=True, squeeze=False,
                             figsize=(max(8, 0.7 * len(fams)) * len(cons), 9))
    for i_con, con in enumerate(cons):
        dc = d[d.constellation == con]
        for i_m, (col, label, hint) in enumerate(metrics):
            ax = axes[i_m][i_con]
            for k, proj in enumerate(projs):
                g = dc[dc.projection == proj].groupby("family")[col].mean().reindex(fams)
                ax.bar(x + (k - (len(projs) - 1) / 2) * w, g.values, width=w * 0.92,
                       label=PROJ_LABEL.get(proj, proj))
            if not ref.empty and np.isfinite(ref[col].mean()):
                ax.axhline(ref[col].mean(), color="k", ls="--", lw=1.2,
                           label="Obergrenze (rauschfrei)")
            ax.grid(axis="y", alpha=0.3)
            ax.axhline(0, color="k", lw=0.6)
            if i_con == 0:
                ax.set_ylabel(f"{label}\n({hint})", fontsize=9)
            if i_m == 0:
                ax.set_title(f"Konstellation: {con}", fontsize=10)
        axes[-1][i_con].set_xticks(x)
        axes[-1][i_con].set_xticklabels(fams, rotation=30, ha="right", fontsize=8)
    axes[0][0].legend(ncol=max(len(projs), 1) + 1, fontsize=8, loc="upper left",
                      framealpha=0.9)
    fig.suptitle("Abb. 21 – Driftfamilien im Bildraum (HbO). Der Lokalisationsfehler ist "
                 "die Kennzahl,\ndie es im Kanalraum nicht gibt: sitzt die Aktivierung "
                 "am richtigen Ort?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIGURES / "21_image_families.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"-> {out}")


# --------------------------------- Abb. 22/23: Multisubject -- Lateralisierung, Systemik

def fig_multisubject(csv: str = "msglm_summary.csv"):
    """Abb. 22 -- kontralaterale Kontrolle; Abb. 23 -- Systemik-Achse."""
    path = RESULTS / csv
    if not path.exists():
        print(f"(uebersprungen: {path} fehlt -- 'python -m drift_glm.analysis.msglm' laufen lassen)")
        return
    df = pd.read_csv(path)
    d = df[df.chromo == "HbO"]
    sysl = [s for s in SYS_LABEL if s in set(d.systemic)]

    # --- Abb. 22: Lateralisierung + Ort im Bildraum
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    x = np.arange(len(sysl))
    for k, (col, lab) in enumerate((("lat_right", "Tapping rechts → links erwartet"),
                                    ("lat_left", "Tapping links → rechts erwartet"))):
        if col not in d:
            continue
        g = d.groupby("systemic")[col].mean().reindex(sysl)
        ax.bar(x + (k - 0.5) * 0.38, g.values, width=0.36, label=lab)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([SYS_LABEL[s] for s in sysl], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("β kontralateral − ipsilateral [µM]")
    ax.set_title("Kanalraum: ist die kontralaterale Seite stärker?\n"
                 "positiv = Erwartung erfüllt")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    if "loc_err_mm" in d:
        for k, (col, lab) in enumerate((("loc_err_mm", "zur erwarteten Landmarke"),
                                        ("loc_err_wrong_mm", "zur Gegenseite"))):
            g = d.groupby("systemic")[col].mean().reindex(sysl)
            ax.bar(x + (k - 0.5) * 0.38, g.values, width=0.36, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([SYS_LABEL[s] for s in sysl], rotation=25, ha="right",
                           fontsize=8)
        ax.set_ylabel("Abstand des Maximums [mm]")
        ax.set_title("Bildraum: liegt das Maximum näher an der\nerwarteten Landmarke "
                     "als an der Gegenseite?")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Abb. 22 – Kontralaterale Kontrolle auf dem Multisubject-Datensatz "
                 "(5 Probanden, HbO)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIGURES / "22_ms_lateralisation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"-> {out}")

    # --- Abb. 23: Designmatrix gegen Abzug
    pairs = [(s, s.replace("_dm", "_sub")) for s in sysl if s.endswith("_dm")
             and s.replace("_dm", "_sub") in set(d.systemic)]
    if not pairs:
        print("(Abb. 23 uebersprungen: keine _dm/_sub-Paare im Lauf)")
        return
    fams = list(dict.fromkeys(d.family))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (col, lab) in zip(axes, (("reliability_r", "Reproduzierbarkeit (Halbierung)"),
                                     ("n_significant", "signifikante Kanäle"),
                                     ("beta_max", "größtes β [µM]"))):
        xx = np.arange(len(fams))
        for k, s in enumerate(sum(map(list, pairs), [])):
            g = d[d.systemic == s].groupby("family")[col].mean().reindex(fams)
            ax.plot(xx, g.values, marker="o", ms=4,
                    ls="-" if s.endswith("_dm") else "--", label=SYS_LABEL[s])
        ax.set_xticks(xx)
        ax.set_xticklabels(fams, rotation=45, ha="right", fontsize=7)
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Abb. 23 – Global-Component-Subtraktion: derselbe Regressor in der "
                 "Designmatrix (durchgezogen) gegen vorher abgezogen (gestrichelt)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = FIGURES / "23_global_component.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"-> {out}")


def _try(label, fn, *a, **kw):
    """Eine Abbildung erzeugen, Fehler melden statt den Lauf abzubrechen.

    Das Skript laeuft am Ende eines mehrstuendigen Nachtlaufs unbeaufsichtigt. Wuerde eine
    fehlgeschlagene Abbildung die uebrigen mitnehmen, waere am Morgen nichts da -- und die
    billigen Tabellen-Abbildungen haengen an den teuren (Abb. 19/20 brauchen je einen
    vollen Build plus AR-IRLS-Fit).
    """
    import traceback
    try:
        fn(*a, **kw)
        return True
    except Exception:                                          # noqa: BLE001
        print(f"!! {label} fehlgeschlagen:\n{traceback.format_exc()}", flush=True)
        return False


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    paths.ensure()
    ok = True
    if what in ("all", "tables"):        # zuerst: billig und haengt nur an den CSVs
        ok &= _try("Abb. 21", fig_image_families)
        ok &= _try("Abb. 22/23", fig_multisubject)
    if what in ("all", "hrf"):
        ok &= _try("Abb. 19", fig_hrf_per_channel)
    if what in ("all", "cortex"):
        ok &= _try("Abb. 20", fig_cortex)
    sys.exit(0 if ok else 1)
