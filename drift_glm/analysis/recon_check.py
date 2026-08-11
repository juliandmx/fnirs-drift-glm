"""Rauschfreie Kontrolle des Bildraum-Kreises und der Regularisierung.

Warum das hier liegt und nicht bei `core.imagespace`: Diese Diagnose braucht einen
DATENSATZ (eine echte OD-Zeitreihe fuer die Messvarianz), und `core` darf die
Datenschicht nicht kennen -- sonst laeuft die Abhaengigkeit nach oben und die Schichtung
ist nur noch Dekoration. `core.imagespace` bleibt damit frei von Datensatzwissen; die
Kombination aus Bildraum und Datensatz passiert in `analysis`.

Aufruf:
    conda run -n cedalion python -m drift_glm.analysis.recon_check
    conda run -n cedalion python -m drift_glm.analysis.recon_check nn22_resting
"""

from __future__ import annotations

import sys

import numpy as np

import cedalion.dot as dot

from drift_glm.core import imagespace as ims
from drift_glm.core import preprocess as prep


def reference_od(dataset: str):
    """Eine vorverarbeitete OD-Zeitreihe des Datensatzes -- fuer `c_meas` und Diagnosen.

    Ohne Motion Correction, weil hier nur die Rauschgroesse je Kanal gebraucht wird und
    die Korrektur sie veraendern wuerde.
    """
    if dataset == "nn22_resting":
        import cedalion.data as cdata
        return prep.finish(prep.to_od_stage(cdata.get_nn22_resting_state()),
                           motion_method="none")
    if dataset == "multisubject_fingertapping":
        from drift_glm.data import multisubject as ms
        return ms.preprocess_recording(ms.load(ms.paths()[0]),
                                       motion_method="none")[0]
    raise ValueError(f"reference_od kennt den Datensatz {dataset!r} nicht")


def recon_check(dataset: str = "multisubject_fingertapping", *,
                alpha_meas_grid=("est", 0.001, 1.0),
                alpha_spatial_grid=(None, 0.001, 0.01),
                target_uM: float = 0.6) -> list[dict]:
    """Rauschfreie Kontrolle des Kreises Bildraum -> Kanalraum -> Bildraum.

    Das eingemischte Muster ist bekannt, also laesst sich die Rekonstruktion OHNE
    Ruhedaten und OHNE GLM pruefen: was kommt zurueck, wenn man exakt das Wahre
    hineingibt? Was hier nicht funktioniert, kann spaeter nicht an der Driftfamilie
    liegen. Zugleich belegt der Lauf die Wahl der Regularisierungsparameter empirisch,
    statt sie nur zu uebernehmen.

    Gemessen wird ausschliesslich auf den sichtbaren Vertices
    (`imagespace.sensitivity_mask`) -- ohne diese Beschraenkung misst man das Verhalten der
    Tiefenkorrektur statt der Rekonstruktion.

    Default ist die 28-Kanal-Montage, NICHT nn22: dort ist eine `ImageRecon`-Instanz
    ~16 MB statt ~300 MB, und die Aussage ueber die Regularisierung ist dieselbe. Auf
    dieser Maschine (7,8 GB) laesst nn22 nichts anderes daneben laufen -- ein volles
    Parameterraster darauf hat 40 Minuten CPU gebraucht und zwei parallele Laeufe
    OOM-killen lassen.
    """
    pre = reference_od(dataset)
    od = pre.od
    chans = [str(c) for c in od.channel.values]

    gt = ims.ground_truth(dataset, chans, pre.geo3d, target_uM=target_uM)
    A = ims.adot(dataset).sel(channel=chans)
    mask = ims.sensitivity_mask(A)
    xyz = ims.vertex_coords_mm()
    truth = np.asarray(gt["img"].sel(chromo="HbO").pint.dequantify().values, float)
    c_meas = ims.c_meas_of(od)

    print(f"{dataset}: {len(chans)} Kanaele, {int(mask.sum())} von {mask.size} "
          f"Vertices sichtbar (> {ims.SENS_LOG_THRESHOLD} log10)")
    print(f"Wahrheit: Peak {np.abs(truth).max():.3f} µM im Bildraum, "
          f"{target_uM:.2f} µM im Kanalraum\n")
    print(f"{'a_spatial':>10s} {'a_meas':>10s} {'r':>7s} {'peak_hat':>9s} "
          f"{'peak/wahr':>10s} {'Ort [mm]':>9s}")

    rows = []
    for aspat in alpha_spatial_grid:
        for am in alpha_meas_grid:
            if am == "est":
                r_, _ = ims.recon_operator(A, od, alpha_spatial=aspat)
                a_used = r_.alpha_meas
            else:
                a_used = float(am)
                r_ = dot.ImageRecon(A, recon_mode="mua2conc", brain_only=True,
                                    alpha_meas=a_used, alpha_spatial=aspat,
                                    apply_c_meas=True, spatial_basis_functions=None)
            img = r_.reconstruct(gt["chan_od"], c_meas)
            h = np.asarray(img.sel(chromo="HbO").pint.dequantify().values, float)
            hm, tm = h[mask], truth[mask]
            r = (float(np.corrcoef(hm, tm)[0, 1])
                 if hm.std() > 0 and tm.std() > 0 else np.nan)
            idx = np.flatnonzero(mask)
            j = idx[int(np.nanargmax(np.abs(hm)))]
            loc = min(float(np.linalg.norm(xyz[j] - xyz[s]))
                      for s in gt["seeds"].values())
            rows.append(dict(alpha_spatial=aspat, alpha_meas=a_used, r=r,
                             peak=float(np.abs(hm).max()), loc_err_mm=loc))
            print(f"{str(aspat):>10s} {a_used:10.4g} {r:+7.3f} "
                  f"{np.abs(hm).max():9.3f} {np.abs(hm).max() / np.abs(tm).max():10.3f} "
                  f"{loc:9.1f}", flush=True)
            del r_, img
    return rows


if __name__ == "__main__":
    recon_check(*sys.argv[1:2])       # Default: die guenstige 28-Kanal-Montage
