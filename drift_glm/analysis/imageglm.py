"""GLM-Ergebnis zurueck in den Bildraum -- geschaetzte HRF und Residuum.

Betreuungsvorgabe 2026-08-05: "vorgehen fuer alle datasets: ich mache glm, dann
geschaetzte hrf (in alternativlauf auch residuals nutzen) in den image space bringen".

WAS HIER GEMESSEN WIRD, UND WARUM DAS ETWAS ANDERES IST ALS BISHER. Im Kanalraum lautet
die Frage "wie genau ist die Amplitude je Kanal?". Im Bildraum kommt eine zweite Frage
dazu, die der Kanalraum grundsaetzlich nicht beantworten kann: **landet die Aktivierung am
richtigen Ort?** Genau das ist der Grund fuer die Vorgabe -- eine Driftfamilie kann die
Amplitude gut treffen und die Aktivierung trotzdem an die falsche Stelle des Kortex
schmieren, weil die Rekonstruktion die Fehler ueber ganze Sensitivitaetsprofile verteilt.

DREI PROJEKTIONEN, und sie beantworten verschiedene Fragen:

  `hrf`       Die beta-Karte des HRF-Regressors. Das ist die parametrische Schaetzung:
              was das Modell fuer die Aktivierung HAELT. Der Hauptlauf.

  `residual`  Das Residuum allein, an den Stimuluszeiten blockgemittelt. Wenn das
              Driftmodell passt, ist hier zur HRF-Zeit nichts Systematisches -- Rauschen
              mittelt sich weg. Bleibt dort HRF-Struktur uebrig, hat das Modell
              Aktivierung NICHT erklaert. Fuer eine Arbeit ueber Driftregressoren ist das
              das direkteste Mass fuer "Modell zu schwach".

  `cleaned`   Residuum + HRF-Anteil, also die Zeitreihe ohne Drift und ohne Stoerregressoren,
              blockgemittelt. Die nichtparametrische Gegenprobe: sie braucht die
              HRF-Kurvenform nicht und zeigt, wieviel von der Uebereinstimmung der
              parametrischen Annahme geschuldet ist.

Alle drei werden auf DIESELBE Weise zu einer Amplitudenkarte je Kanal verdichtet und
DIESELBE Verdichtung wird auf die Ground Truth angewandt -- dadurch sind sie ohne
Umrechnungsannahme miteinander und mit der Wahrheit vergleichbar.

Aufruf:
    conda run -n cedalion python -m drift_glm.analysis.imageglm test    # 2 Familien, kurzes Fenster
    conda run -n cedalion python -m drift_glm.analysis.imageglm         # alle Familien (Nachtlauf)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import cedalion.models.glm as glm
from cedalion import units

from drift_glm.core import imagespace as ims
from drift_glm.core import pipeline as pl
from drift_glm.core import preprocess as prep
from drift_glm.core import shortchannel as sc
from drift_glm.analysis.sweep import drift_dm
from drift_glm import paths

RESULTS = paths.RESULTS

PROJECTIONS = ("hrf", "residual", "cleaned")

#: Zusaetzliche "Projektion", die keine Schaetzung ist: die WAHRE Kanalkarte durch
#: denselben Rueckweg. Sie ist die Obergrenze, die die Rekonstruktion ueberhaupt erreichen
#: kann, und ohne sie ist keine der anderen Zahlen interpretierbar -- ein
#: Lokalisationsfehler von 12 mm kann hervorragend oder schlecht sein, je nachdem, was
#: rauschfrei herauskaeme. Auf nn22 sind es rauschfrei r = 0.80 und 11.9 mm
#: (alpha_spatial = 0.001) bzw. r = 0.87 und 3.0 mm (alpha_spatial = 0.01).
TRUTH_REF = "truth_ref"

#: Fenster nach Stimulusbeginn, ueber das die Blockantwort gemittelt wird [s].
#: Bei 10 s Blockdauer und einer Gamma-Basis mit sigma = 3 s liegt das Plateau hier.
#: Die genaue Wahl ist unkritisch, weil Schaetzung UND Wahrheit dasselbe Fenster benutzen.
AMP_WINDOW = (4.0, 10.0)
EPOCH_BEFORE = 5.0 * units.s
EPOCH_AFTER = 20.0 * units.s

#: Radius um das wahre Blob-Zentrum, innerhalb dessen eine Rekonstruktion als "am
#: richtigen Ort" gilt [mm]. 30 mm ist etwa die Ortsauflaesung, die von einer
#: Einzelabstandsmontage bei 3 cm ueberhaupt erwartbar ist.
HIT_RADIUS_MM = 30.0


def with_time_coords(ts: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
    """Stellt die `samples`-Koordinate und die Zeiteinheit wieder her.

    `glm.predict` reicht sie nicht durch, `to_epochs` verlangt sie aber (es rechnet die
    Fensterlaengen in Samples um). Ohne diesen Schritt scheitert jede Blockmittelung auf
    einer vorhergesagten Zeitreihe.
    """
    if "samples" in ts.coords or "samples" not in like.coords:
        return ts
    out = ts.assign_coords(samples=("time", like.samples.values))
    if not out.time.attrs.get("units"):
        out.time.attrs["units"] = like.time.attrs.get("units", "second")
    return out


def block_amplitude(ts: xr.DataArray, stim_df, trial_types=None,
                    like: xr.DataArray | None = None) -> xr.DataArray:
    """Zeitreihe -> Amplitudenkarte je Kanal (channel, chromo), ueber `AMP_WINDOW`.

    Blockmittelung mit Baseline-Abzug vor dem Reiz, danach Mittel ueber das
    Plateaufenster. Der Baseline-Abzug ist nicht Kosmetik: ohne ihn traegt jeder Epoch
    seinen Drift-Offset mit in den Mittelwert, und genau der ist hier die Stoergroesse.

    Bei mehreren Trial-Typen wird ueber sie gemittelt -- die Karten sind dann direkt mit
    der bilateralen Ground Truth vergleichbar.
    """
    tt = trial_types
    if tt is None:
        tt = [str(t) for t in pd.unique(stim_df.trial_type)]
    if like is not None:
        ts = with_time_coords(ts, like)
    ep = ts.cd.to_epochs(stim_df, tt, before=EPOCH_BEFORE, after=EPOCH_AFTER)
    ep = ep - ep.sel(reltime=(ep.reltime < 0)).mean("reltime")
    ba = ep.groupby("trial_type").mean("epoch")
    amp = ba.sel(reltime=slice(*AMP_WINDOW)).mean("reltime")
    return amp.mean("trial_type") if "trial_type" in amp.dims else amp


def conc_map_to_od(m: xr.DataArray, geo3d, wavelength, dpf: float = prep.DEFAULT_DPF,
                   like: xr.DataArray | None = None) -> xr.DataArray:
    """Amplitudenkarte in Konzentration [µM] -> Optical Density, fuer die Rekonstruktion.

    `ImageRecon` erwartet OD. Die Schaetzung liegt aber in Konzentration vor, weil das GLM
    im Konzentrationsraum laeuft. `conc2od` ist die exakte Umkehrung des `od2conc` aus der
    Vorverarbeitung -- Hin- und Rueckweg sind also dieselbe Transformation, und ohne
    Stoerung ist der Kreis Bildraum -> Kanal -> Bildraum geschlossen. Nur deshalb ist eine
    gemessene Abweichung im Bildraum ein Effekt der Methode und kein Umrechnungsartefakt.
    """
    x = m
    if like is not None:                     # source/detector fuer die Kanalabstaende
        x = x.assign_coords({k: like[k] for k in ("source", "detector")
                             if k in like.coords})
    x = x.expand_dims("time").assign_coords(time=[0.0])
    x.time.attrs["units"] = "second"
    od = prep.to_od_activation(x, geo3d, wavelength, dpf)
    return od.isel(time=0, drop=True)


def fit_projections(P: pl.Pipeline, family: str, constellation: str = "baseline",
                    noise_model: str = "ar_irls", *, ar_order: int = 30,
                    max_jobs: int = -1) -> dict:
    """GLM fitten und die drei Amplitudenkarten (Kanalraum) zurueckgeben.

    Rueckgabe: {"hrf": .., "residual": .., "cleaned": .., "truth": ..} je (channel, chromo)
    in µM, plus `"fit_s"` als Laufzeit.

    Der Fit laeuft -- wie im gesamten Projekt -- auf den LANGEN Kanaelen, sobald ein
    Short-Channel-Regressor im Spiel ist; ausgewertet wird ohnehin nur dort.
    """
    ts_all = P.conc_syn
    ts_long, ts_short = sc.split(ts_all, P.geo3d, sc.SHORT_THRESHOLD)
    use_short = constellation in sc.VARIANTS
    ts = ts_long if use_short else ts_all

    dm_drift, filt = drift_dm(family, ts)
    if filt is not None:
        # Filter-Alternative statt Driftregressoren, im Konzentrationsraum
        ts = ts.cd.freq_filter(filt[0] * units.Hz, filt[1] * units.Hz, 4)

    dm_hrf = glm.design_matrix.hrf_regressors(
        ts, P.stim_df, glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=0 * units.s))
    hrf_names = [r for r in dm_hrf.common.regressor.values if str(r).startswith("HRF")]
    dm_hrf, _ = pl._normalize_hrf_to_unit_peak(dm_hrf, hrf_names)

    dm = dm_hrf & dm_drift
    if constellation == "global":
        dm = dm & glm.design_matrix.global_mean_regressor(ts)
    elif use_short:
        dm = dm & sc.short_dm(constellation, ts, ts_short, P.geo3d)

    t0 = time.time()
    res = glm.fit(ts, dm, noise_model=noise_model, ar_order=ar_order, max_jobs=max_jobs)
    fit_s = time.time() - t0
    params = res.sm.params

    # parametrisch: beta des HRF-Regressors (Regressor ist auf Peak 1 normiert)
    beta = params.sel(regressor=hrf_names)
    beta_hrf = beta.sum("regressor") if len(hrf_names) > 1 else beta.isel(regressor=0)

    # Residuum und "gereinigte" Zeitreihe
    full = glm.predict(ts, params, dm).transpose(*ts.dims)
    resid = ts - full
    hrf_part = glm.predict(ts, beta, dm_hrf).transpose(*ts.dims)
    cleaned = resid + hrf_part

    out = {
        "hrf": beta_hrf,
        "residual": block_amplitude(resid, P.stim_df, like=ts),
        "cleaned": block_amplitude(cleaned, P.stim_df, like=ts),
        "truth": block_amplitude(P.activation.sel(channel=ts.channel), P.stim_df,
                                 like=ts),
        "beta_truth": (P.beta_true_map.sel(channel=ts.channel).max("trial_type")
                       if "trial_type" in P.beta_true_map.dims
                       else P.beta_true_map.sel(channel=ts.channel)),
        "fit_s": fit_s,
        "channels": ts.channel,
    }
    return out


def _values(a, chromo):
    x = a.sel(chromo=chromo)
    x = x.pint.dequantify() if getattr(x, "pint", None) is not None \
        and x.pint.units is not None else x
    return np.asarray(x.values, dtype=float)


def channel_metrics(hat: xr.DataArray, truth: xr.DataArray, chromo: str,
                    active_frac: float = 0.1) -> dict:
    """Bias/RMSE/Korrelation im Kanalraum, ueber die aktiven Kanaele.

    "Aktiv" heisst: die Wahrheit liegt ueber `active_frac` des Peaks. Ohne diese
    Einschraenkung dominieren die vielen Kanaele ohne Aktivierung jede Kennzahl, und der
    relative Fehler ist am Blob-Rand per Konstruktion riesig (siehe Abb. 5).
    """
    h, t = _values(hat, chromo), _values(truth, chromo)
    m = np.isfinite(h) & np.isfinite(t) & (np.abs(t) > active_frac * np.nanmax(np.abs(t)))
    if m.sum() < 3:
        return dict(bias=np.nan, rmse=np.nan, r=np.nan, n_active=int(m.sum()))
    d = h[m] - t[m]
    r = (float(np.corrcoef(h[m], t[m])[0, 1])
         if h[m].std() > 0 and t[m].std() > 0 else np.nan)
    return dict(bias=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())), r=r,
                n_active=int(m.sum()))


def image_metrics(img_hat: xr.DataArray, img_true: xr.DataArray, seeds: dict,
                  chromo: str, head_ras=None, mask: np.ndarray | None = None) -> dict:
    """Guetemasse im Bildraum. `loc_err_mm` und `hit_frac` gibt es nur hier.

      `r`          Korrelation ueber die sichtbaren Vertices -- die Form des Bildes.
      `peak_ratio` Verhaeltnis der Spitzenwerte -- die Amplitudentreue. Werte weit unter 1
                   sind normal: jede Regularisierung verschmiert und daempft.
      `rmse_rel`   RMSE, bezogen auf den Peak der Wahrheit.
      `loc_err_mm` Abstand des rekonstruierten Maximums zum naechsten wahren Blob-Zentrum.
                   Das ist die Frage, um die es der Betreuung geht.
      `hit_frac`   Anteil der rekonstruierten Masse innerhalb von `HIT_RADIUS_MM` um ein
                   wahres Zentrum. Ergaenzt `loc_err_mm`: ein Maximum kann zufaellig
                   richtig sitzen, waehrend der Rest des Bildes ueber den Kortex schmiert.

    `mask` MUSS die Sichtbarkeitsmaske sein (`imagespace.sensitivity_mask`). Ohne sie
    laufen alle Kennzahlen ins Leere, und zwar nicht subtil: die Rekonstruktion liefert
    auch fuer Vertices Werte, zu denen kein Photon gelangt ist, und die Tiefenkorrektur
    (`alpha_spatial`) blaest genau diese am staerksten auf. Gemessen ohne Maske lag das
    rekonstruierte Maximum 65-110 mm vom wahren Zentrum entfernt und die Korrelation bei
    0.00 -- gemessen wurde dabei das Verhalten des Regularisierers, nicht die Schaetzung.
    """
    h, t = _values(img_hat, chromo), _values(img_true, chromo)
    if h.ndim > 1:
        h = np.nanmean(h, axis=tuple(range(1, h.ndim)))
    if t.ndim > 1:
        t = np.nanmean(t, axis=tuple(range(1, t.ndim)))
    xyz = ims.vertex_coords_mm(head_ras)
    keep = np.ones(h.shape, dtype=bool) if mask is None else np.asarray(mask, bool)
    idx = np.flatnonzero(keep)

    hv, tv = h[keep], t[keep]
    pk_t, pk_h = float(np.nanmax(np.abs(tv))), float(np.nanmax(np.abs(hv)))
    m = np.isfinite(hv) & np.isfinite(tv)
    r = (float(np.corrcoef(hv[m], tv[m])[0, 1])
         if m.sum() > 3 and hv[m].std() > 0 and tv[m].std() > 0 else np.nan)

    j = idx[int(np.nanargmax(np.abs(hv)))]
    dmin = min(float(np.linalg.norm(xyz[j] - xyz[s])) for s in seeds.values())
    d_seed = np.min(np.stack([np.linalg.norm(xyz[keep] - xyz[s], axis=1)
                              for s in seeds.values()]), axis=0)
    mass = np.abs(hv)
    hit = float(mass[d_seed <= HIT_RADIUS_MM].sum() / mass.sum()) if mass.sum() > 0 \
        else np.nan

    return dict(r=r, peak_ratio=pk_h / pk_t if pk_t else np.nan,
                rmse_rel=float(np.sqrt(np.nanmean((hv - tv) ** 2)) / pk_t) if pk_t else
                np.nan,
                loc_err_mm=dmin, hit_frac=hit, n_vertices=int(keep.sum()))


#: Seeds fuer den Volllauf -- gemessen EINER, und das ist eine Kostenentscheidung.
#:
#: Ein AR-IRLS-Fit ueber die 519 Kanaele bei 368 s kostet gemessen **420 s**. Bei 12
#: Familien x 2 Konstellationen sind das je Seed 24 Zellen, also ~2,8 h, plus rund 20 min
#: fuer den `pipeline.build` selbst (die Wavelet-Korrektur laeuft zweimal: reine Ruhedaten
#: und augmentiert). Mit zwei Seeds waere der Lauf bei ~10 h -- und danach soll noch
#: `msglm.py` laufen. Ein Seed haelt die Kette in einer Nacht.
#:
#: Der Verlust ist vertretbar: die Varianz ueber Stimulus-Platzierungen ist die Frage, die
#: der Kanalraum-Sweep mit 4 Seeds x 3 Fenstern beantwortet. Hier geht es um den Ort, und
#: den beantwortet ein Seed gegen die mitlaufende rauschfreie Obergrenze.
DEFAULT_SEEDS = (0,)


def run(mode: str = "full", *, dataset: str = "nn22_resting", window_s: float = 368.0,
        families=None, constellations=("baseline", "global"), noise_model="ar_irls",
        seeds=DEFAULT_SEEDS, out: str | None = None) -> pd.DataFrame:
    """Driftfamilien im Bildraum vergleichen. Schreibt eine CSV nach `results/`."""
    paths.ensure()
    if families is None:
        families = ["poly:3", "dct:0.02"] if mode == "test" else list(FAMILIES)
    if mode == "test":
        window_s, constellations, noise_model = 180.0, ("baseline",), "ols"

    t00 = time.time()
    head_ras = ims.head()
    recs, total = [], len(families) * len(constellations) * len(seeds)
    done = 0
    path = RESULTS / (out or ("imageglm_summary_test.csv" if mode == "test"
                              else "imageglm_summary.csv"))

    for seed in seeds:
        P = pl.build(dataset=dataset, window_s=window_s, seed=seed,
                     activation_space="image")
        if P.beta_true_img is None:
            raise ValueError("run() braucht die Bildraum-Wahrheit "
                             "(activation_space='image')")
        img_true = P.beta_true_img
        if "trial_type" in img_true.dims:
            img_true = img_true.sum("trial_type")

        # EINE Rekonstruktion je Build: W haengt nur an Montage, Kanalmenge und
        # Messvarianz, nicht an der Driftfamilie. Neu bauen je Familie waere der
        # Kostentreiber (und bei ~450 MB je Instanz auch das Speicherproblem).
        A = ims.adot(dataset).sel(channel=[str(c) for c in P.pre.od.channel.values])
        recon, c_meas = ims.recon_operator(A, P.pre.od)
        sens = ims.sensitivity_mask(A)

        # Die Obergrenze: die WAHRE Kanalkarte durch denselben Rueckweg. Ohne diese Zeile
        # ist keine der folgenden interpretierbar -- 12 mm Lokalisationsfehler koennen
        # ausgezeichnet oder schlecht sein, je nachdem, was rauschfrei herauskommt. Haengt
        # nicht von Familie oder Konstellation ab, also einmal je Build.
        bt_ref = P.beta_true_map
        if "trial_type" in bt_ref.dims:
            bt_ref = bt_ref.max("trial_type")
        od_ref = conc_map_to_od(bt_ref, P.geo3d, P.pre.od.wavelength, like=P.conc_syn)
        img_ref = recon.reconstruct(od_ref, c_meas.sel(channel=od_ref.channel))
        ref_metrics = {ch: image_metrics(img_ref, img_true, P.seeds, ch, head_ras, sens)
                       for ch in ("HbO", "HbR")}
        print(f"  Obergrenze (rauschfrei, seed={seed}): "
              + "  ".join(f"{ch}: r={m['r']:+.3f} Ort={m['loc_err_mm']:.1f} mm"
                          for ch, m in ref_metrics.items()), flush=True)
        for ch, m in ref_metrics.items():
            row = dict(dataset=dataset, seed=seed, family="-", constellation="-",
                       noise_model="-", projection=TRUTH_REF, chromo=ch,
                       window_s=window_s, n_channels=int(od_ref.sizes["channel"]),
                       alpha_meas=recon.alpha_meas, alpha_spatial=recon.alpha_spatial,
                       fit_s=0.0)
            row.update({f"img_{k}": v for k, v in m.items()})
            recs.append(row)
        del od_ref, img_ref

        for fam in families:
            for con in constellations:
                pr = fit_projections(P, fam, con, noise_model)
                for proj in PROJECTIONS:
                    od = conc_map_to_od(pr[proj], P.geo3d, P.pre.od.wavelength,
                                        like=P.conc_syn)
                    img = recon.reconstruct(od, c_meas.sel(channel=od.channel))
                    ref = pr["beta_truth"] if proj == "hrf" else pr["truth"]
                    for ch in ("HbO", "HbR"):
                        row = dict(dataset=dataset, seed=seed, family=fam,
                                   constellation=con, noise_model=noise_model,
                                   projection=proj, chromo=ch, window_s=window_s,
                                   n_channels=int(od.sizes["channel"]),
                                   alpha_meas=recon.alpha_meas,
                                   alpha_spatial=recon.alpha_spatial,
                                   fit_s=round(pr["fit_s"], 1))
                        row.update({f"ch_{k}": v for k, v in
                                    channel_metrics(pr[proj], ref, ch).items()})
                        row.update({f"img_{k}": v for k, v in
                                    image_metrics(img, img_true, P.seeds, ch,
                                                  head_ras, sens).items()})
                        recs.append(row)
                done += 1
                d = recs[-2]
                print(f"  [{done}/{total}] {fam:16s} {con:8s} seed={seed}  "
                      f"HbO: r={d['img_r']:+.3f} Ort={d['img_loc_err_mm']:5.1f} mm "
                      f"Treffer={100 * d['img_hit_frac']:4.1f} %  "
                      f"({pr['fit_s']:.0f}s Fit)", flush=True)
                (paths.LOGS / "imageglm_progress.txt").write_text(
                    f"imageglm: {done}/{total}\n"
                    f"verstrichen: {(time.time() - t00) / 60:.1f} min\n"
                    f"ETA: {(time.time() - t00) / done * (total - done) / 60:.1f} min\n"
                    f"letzte Zelle: {fam} {con} seed={seed}\n")
                # Nach jeder Zelle schreiben -- der Lauf dauert Stunden.
                pd.DataFrame(recs).to_csv(path, index=False)

    df = pd.DataFrame(recs)
    df.to_csv(path, index=False)
    print(f"\n[OK] {len(df)} Zeilen in {(time.time() - t00) / 60:.1f} min -> {path}")
    return df


#: Familien wie im Kanalraum-Sweep, damit die Ranglisten vergleichbar sind.
FAMILIES = ("none", "poly:1", "poly:3", "poly:5", "dct:0.005", "dct:0.01", "dct:0.02",
            "legendre:1", "legendre:3", "bspline:5", "bspline:8", "butter:0.01")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    df = run(mode)
    d = df[(df.chromo == "HbO")]
    print("\nHbO im Bildraum (Korrelation / Lokalisationsfehler / Trefferanteil):")
    print(d.pivot_table(index="family", columns="projection",
                        values=["img_r", "img_loc_err_mm", "img_hit_frac"])
           .to_string(float_format=lambda x: f"{x:.3f}"))
