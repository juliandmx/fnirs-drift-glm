"""Koregistrierung der NIRScout-Montage (Khan-Datensatz) auf das ICBM152-Kopfmodell.

Die `geo3d` des Datensatzes enthaelt nur 16 Quellen und 16 Detektoren, keine Landmarken;
die Cedalion-Registrierungen (`register_*`, `simple_scalp_projection`) brauchen aber
Landmarkenpaare. Hier deshalb eine landmarkenfreie 7-Parameter-Anpassung (Verschiebung,
Drehung, gemeinsame Skalierung), die die Optoden auf die ICBM152-Kopfhaut legt und die
gemessenen Quell-Detektor-Abstaende erhaelt. Weil eine Kopfhaut annaehernd
links-rechts-symmetrisch ist, laeuft der Fit zusaetzlich von einem gespiegelten Start;
sind beide Restfehler vergleichbar, ist die Seite nicht bestimmt.

Aus den Positionen wird eine Adot erst ueber eine Photonensimulation (`compute_fluence_mcx`
braucht CUDA, `compute_fluence_nirfaster` das Plugin `nirfasteruff`). Dieses Modul liefert
deshalb die registrierten Positionen und ein Skript dafuer (`FLUENCE_SCRIPT`), nicht die
Matrix.

Aufruf:  conda run -n cedalion python -m drift_glm.data.coregister
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree

import cedalion.nirs

from drift_glm.core import imagespace as ims
from drift_glm.data import realdata as rd


def _rot(a: np.ndarray) -> np.ndarray:
    """Rotationsmatrix aus drei Winkeln (x-y-z, Radiant)."""
    cx, cy, cz = np.cos(a)
    sx, sy, sz = np.sin(a)
    return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))


def optode_positions_mm(rec) -> tuple[np.ndarray, list[str]]:
    """Optodenkoordinaten (n, 3) in mm und ihre Labels."""
    g = rec.geo3d
    g = g.pint.to("mm").pint.dequantify() if g.pint.units is not None else g
    return np.asarray(g.values, dtype=float), [str(x) for x in g.label.values]


def sphere_fit(p: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Kugel durch die Punkte: (Zentrum, Radius, Reststreuung in mm).

    Plausibilitaetspruefung des Rohkoordinatensystems: Optoden auf einer kopfgrossen
    Kugelschale (Radius ~85-95 mm) sind Oberflaechenkoordinaten in mm.
    """
    A = np.hstack([2 * p, np.ones((len(p), 1))])
    b = (p ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(sol[3] + c @ c))
    d = np.linalg.norm(p - c, axis=1)
    return c, r, float(d.std())


def _fit(p0: np.ndarray, tree: KDTree, nominal_mm: np.ndarray, pairs,
         dist_weight: float = 5.0, mirror: bool = False):
    """7-Parameter-Anpassung: Optoden auf die Kopfhaut, Kanalabstaende erhalten.

    Zielfunktion: mittlerer quadrierter Abstand zur Kopfhaut plus gewichteter Strafterm
    fuer veraenderte Quell-Detektor-Abstaende; ohne den Strafterm wuerde die Skalierung
    die Montage an die Kopfform anpassen.
    """
    centre = p0.mean(0)
    q0 = (p0 - centre)
    if mirror:
        q0 = q0 * np.array([-1.0, 1.0, 1.0])       # Spiegelung an der Sagittalebene

    target = tree.data.mean(0)

    def place(x):
        s = np.exp(x[6])                            # positive Skalierung
        return (s * (q0 @ _rot(x[:3]).T)) + target + x[3:6]

    def cost(x):
        q = place(x)
        d, _ = tree.query(q)
        dist = np.linalg.norm(q[pairs[:, 0]] - q[pairs[:, 1]], axis=1)
        return float((d ** 2).mean()
                     + dist_weight * ((dist - nominal_mm) ** 2).mean())

    best = None
    rng = np.random.default_rng(0)
    # Mehrere Startwerte, die Zielfunktion ist wegen der Drehungen nicht konvex.
    for k in range(24):
        x0 = np.zeros(7)
        if k:
            x0[:3] = rng.uniform(-np.pi, np.pi, 3)
            x0[3:6] = rng.normal(0, 10, 3)
        r = minimize(cost, x0, method="Powell",
                     options=dict(maxiter=20000, xtol=1e-3, ftol=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    q = place(best.x)
    d, _ = tree.query(q)
    dist = np.linalg.norm(q[pairs[:, 0]] - q[pairs[:, 1]], axis=1)
    # Kostenanteile getrennt ausweisen: nur der Oberflaechenanteil sagt etwas ueber links/rechts.
    surf_cost = float((d ** 2).mean())
    dist_cost = float(dist_weight * ((dist - nominal_mm) ** 2).mean())
    return dict(positions=q, cost=float(best.fun), scale=float(np.exp(best.x[6])),
                surf_cost=surf_cost, dist_cost=dist_cost,
                surf_mm=(float(np.median(d)), float(d.max())),
                dist_err_mm=(float(np.median(np.abs(dist - nominal_mm))),
                             float(np.abs(dist - nominal_mm).max())),
                mirror=mirror)


FLUENCE_SCRIPT = '''\
# Adot fuer die NIRScout-Montage berechnen; braucht eine CUDA-GPU.
# Laufzeit-Groessenordnung: 32 Optoden x 1e8 Photonen, mehrere GPU-Stunden.
import cedalion.dot as dot
from drift_glm.data import coregister, realdata as rd

rec = rd.load(rd.find_files()[0])
geo3d = coregister.registered_geo3d(rec)          # Positionen aus diesem Modul
head = dot.get_standard_headmodel("icbm152")      # Kopfmodell wie im uebrigen Projekt

fm = dot.ForwardModel(head, geo3d, rec._measurement_lists["amp"])
fm.compute_fluence_mcx("fluence_khan_icbm152.h5", nphoton=1e8)
Adot = fm.compute_sensitivity("fluence_khan_icbm152.h5")
Adot.to_netcdf("sensitivity_khan_icbm152.nc")
'''


def registered_geo3d(rec, mirror: bool = False):
    """`geo3d` mit den registrierten Positionen, im CRS des ICBM152-Kopfmodells.

    Rueckgabe hat dieselben Labels und dieselbe Reihenfolge wie das Original -- nur die
    Koordinaten sind ersetzt. Damit ist sie als Eingabe fuer `dot.ForwardModel` brauchbar.
    """
    from cedalion import units

    head = ims.head()
    p, labels = optode_positions_mm(rec)
    tree = KDTree(np.asarray(head.scalp.vertices.pint.dequantify().values
                             if head.scalp.vertices.pint.units is not None
                             else head.scalp.vertices.values, dtype=float))
    nominal, pairs = _nominal(rec, labels)
    res = _fit(p, tree, nominal, pairs, mirror=mirror)
    # Auf die Kopfhaut projizieren (naechster Scalp-Vertex).
    _, idx = tree.query(res["positions"])
    out = rec.geo3d.copy()
    out.values = tree.data[idx] * (1.0 if out.pint.units is None else 1.0)
    return out.pint.dequantify().pint.quantify(units.mm) \
        if out.pint.units is None else out


def _nominal(rec, labels):
    """Nominelle Kanalabstaende [mm] und die Indexpaare der beteiligten Optoden."""
    ts = rec["amp"]
    d = np.asarray(cedalion.nirs.channel_distances(ts, rec.geo3d)
                   .pint.to("mm").pint.dequantify().values, dtype=float)
    li = {lab: i for i, lab in enumerate(labels)}
    pairs = np.array([[li[str(s)], li[str(t)]]
                      for s, t in zip(ts.source.values, ts.detector.values)])
    return d, pairs


def report():
    """Registrierung fuer die erste Khan-Datei durchrechnen und ausgeben."""
    files = rd.find_files()
    if not files:
        print(f"Keine Daten in {rd.DATA_DIR}")
        return
    rec = rd.load(files[0])
    p, labels = optode_positions_mm(rec)
    nominal, pairs = _nominal(rec, labels)

    print(f"Montage: {len(labels)} Optoden, {len(pairs)} Kanaele")
    print(f"Kanalabstaende [mm]: min {nominal.min():.1f}  median "
          f"{np.median(nominal):.1f}  max {nominal.max():.1f}")
    print(f"Landmarken in geo3d: "
          f"{[l for l in labels if not l[0] in 'SD' or not l[1:].isdigit()] or 'keine'}")

    c, r, sd = sphere_fit(p)
    print(f"\nKugelanpassung an die Rohkoordinaten:")
    print(f"  Zentrum {np.round(c, 1)}  Radius {r:.1f} mm  Reststreuung {sd:.1f} mm")
    print("  -> " + ("kopfgrosse Oberflaechenkoordinaten in mm, plausibel"
                     if 70 < r < 110 and sd < 15 else
                     "kein kopfgrosses Koordinatensystem, Registrierung fragwuerdig"))

    head = ims.head()
    sv = head.scalp.vertices
    sv = np.asarray(sv.pint.dequantify().values if sv.pint.units is not None
                    else sv.values, dtype=float)
    tree = KDTree(sv)

    d0, _ = tree.query(p)
    print(f"\nRohkoordinaten ohne Transformation:")
    print(f"  Abstand zur ICBM152-Kopfhaut: median {np.median(d0):.1f} mm, "
          f"max {d0.max():.1f} mm")
    print("  -> " + ("bereits registriert, keine Transformation noetig"
                     if np.median(d0) < 8 else
                     "nicht im Kopfmodell-CRS, Registrierung noetig"))

    print("\nLandmarkenfreie Anpassung (7 Parameter, Kanalabstaende als Nebenbedingung):")
    print(f"  {'Start':>10s} {'Skalierung':>11s} {'Kopfhaut med/max [mm]':>23s} "
          f"{'Abstandsfehler med/max':>24s}")
    fits = {}
    for mirror in (False, True):
        f = _fit(p, tree, nominal, pairs, mirror=mirror)
        fits["gespiegelt" if mirror else "normal"] = f
        print(f"  {'gespiegelt' if mirror else 'normal':>10s} {f['scale']:11.3f} "
              f"{f['surf_mm'][0]:11.1f} /{f['surf_mm'][1]:9.1f} "
              f"{f['dist_err_mm'][0]:12.1f} /{f['dist_err_mm'][1]:10.1f}")

    print("\n  Kostenanteile getrennt (nur der Oberflaechenanteil sagt etwas ueber die Seite):")
    for name, f in fits.items():
        print(f"    {name:>10s}: Oberflaeche {f['surf_cost']:8.3f}   "
              f"Abstands-Strafterm {f['dist_cost']:8.3f}")
    sa, sb = fits["normal"]["surf_cost"], fits["gespiegelt"]["surf_cost"]
    rel = abs(sa - sb) / max(sa, sb)
    print(f"    Unterschied im Oberflaechenanteil: {100 * rel:.1f} %")
    if rel < 0.25:
        print("  -> links/rechts nicht bestimmt: beide Orientierungen liegen praktisch")
        print("     gleich gut auf dem Kopf. Der Bildraum zeigt fuer diesen Datensatz die")
        print("     Struktur der Aktivierung, nicht die Seite (wie im Kanalraum, Abb. 17).")
    else:
        better = "normal" if sa < sb else "gespiegelt"
        print(f"  -> Orientierung '{better}' liegt deutlich besser auf der Kopfhaut.")
        print("     Ohne Landmarken bleibt die Seite trotzdem eine Annahme.")

    print("\nAdot aus den Positionen braucht eine Photonensimulation (CUDA-GPU).")
    print("Skript fuer eine Maschine mit GPU:\n")
    print("    " + FLUENCE_SCRIPT.replace("\n", "\n    "))
    return fits


if __name__ == "__main__":
    report()
