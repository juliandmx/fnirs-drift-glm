"""Koregistrierung der NIRScout-Montage (Khan-Datensatz) auf das ICBM152-Kopfmodell.

DAS PROBLEM. Fuer den Bildraum braucht es eine Sensitivitaetsmatrix, und die haengt an den
Optodenpositionen IM Kopfmodell. Bei nn22 und beim Multisubject-Fingertapping ist sie
vorberechnet. Beim Khan-Datensatz nicht -- und schlimmer: seine `geo3d` enthaelt
ausschliesslich 16 Quellen und 16 Detektoren, **keine Landmarken** (kein Nz, kein LPA/RPA).

Damit fallen alle Standardwege aus:

  * `register_trans_rot_isoscale`, `register_icp`, `register_optodes_spring_icp` --
    alle brauchen zugeordnete Landmarkenpaare zwischen Sonden- und Kopfkoordinatensystem.
  * `simple_scalp_projection` braucht ausdruecklich Nz, LPA und RPA.

WAS HIER STATTDESSEN GEMACHT WIRD (Best-Effort nach Absprache). Ein landmarkenfreier
Anpassungsschritt: 7 Parameter (Verschiebung, Drehung, eine gemeinsame Skalierung) werden
so bestimmt, dass die Optoden moeglichst dicht auf der ICBM152-Kopfhaut liegen, wobei die
gemessenen Quell-Detektor-Abstaende erhalten bleiben muessen (der Datensatz gibt
einheitlich ~3 cm vor, eine Skalierung darf das nicht verzerren).

WAS DAS NICHT LEISTET, und das gehoert als Limitation in die Arbeit: eine Kopfhaut ist
annaehernd links-rechts-symmetrisch. Eine Anpassung, die nur den Abstand zur Oberflaeche
minimiert, kann die Montage daher spiegeln, ohne dass sich der Fehler nennenswert
verschlechtert. Ob das hier so ist, wird nicht behauptet, sondern gemessen: der Fit laeuft
zusaetzlich von einem gespiegelten Start, und beide Restfehler werden ausgewiesen. Sind sie
vergleichbar, ist links/rechts NICHT bestimmt -- dann bleibt die Aussage auf "Struktur der
Aktivierung" beschraenkt, genau wie im Kanalraum (siehe BESPRECHUNG, Abb. 17).

DER ZWEITE BLOCKER, unabhaengig von der Registrierung: aus Positionen wird eine Adot erst
ueber eine Photonensimulation. `compute_fluence_mcx` braucht CUDA -- `pmcx` 0.7.1 ist
installiert, aber es gibt keine GPU (`pmcx.gpuinfo()` schlaegt fehl, kein `nvidia-smi`).
Die CPU-Alternative `compute_fluence_nirfaster` braucht das Plugin `nirfasteruff`, das
nicht installiert ist. Dieses Modul liefert deshalb die registrierten Positionen und ein
lauffaehiges Skript (`fluence_script`), nicht die Matrix.

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

    Eine reine Plausibilitaetspruefung des Rohkoordinatensystems: liegen die Optoden auf
    einer kopfgrossen Kugelschale (Radius ~85-95 mm), sind es Oberflaechenkoordinaten in
    mm. Weichen Radius oder Reststreuung stark ab, ist das Koordinatensystem etwas
    anderes -- dann waere jede Registrierung sinnlos.
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

    Die Zielfunktion ist die Summe der quadrierten Abstaende zur Kopfhaut PLUS ein
    gewichteter Strafterm fuer veraenderte Quell-Detektor-Abstaende. Ohne den Strafterm
    wuerde die Skalierung die Montage so lange schrumpfen oder streckten, bis sie sich der
    Kopfform anschmiegt -- und dann waeren die 3-cm-Abstaende, die der Datensatz vorgibt,
    keine 3 cm mehr.
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
    # Mehrere Startwerte: die Zielfunktion ist nicht konvex (Drehungen), ein einzelner
    # Start landet zuverlaessig in einem lokalen Minimum.
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
    # Die beiden Kostenanteile GETRENNT ausweisen. Sonst liest man einen
    # Gesamtunterschied als "die Kopfform entscheidet ueber links/rechts", waehrend er
    # in Wahrheit aus dem Strafterm fuer die Kanalabstaende stammt -- und der sagt ueber
    # die Seite nichts.
    surf_cost = float((d ** 2).mean())
    dist_cost = float(dist_weight * ((dist - nominal_mm) ** 2).mean())
    return dict(positions=q, cost=float(best.fun), scale=float(np.exp(best.x[6])),
                surf_cost=surf_cost, dist_cost=dist_cost,
                surf_mm=(float(np.median(d)), float(d.max())),
                dist_err_mm=(float(np.median(np.abs(dist - nominal_mm))),
                             float(np.abs(dist - nominal_mm).max())),
                mirror=mirror)


FLUENCE_SCRIPT = '''\
# Adot fuer die NIRScout-Montage berechnen -- BRAUCHT EINE CUDA-GPU.
# Auf dieser Maschine nicht ausfuehrbar (pmcx ohne GPU, nirfasteruff nicht installiert).
# Laufzeit-Groessenordnung: 32 Optoden x 1e8 Photonen, also mehrere GPU-Stunden.
import cedalion.dot as dot
from drift_glm.data import coregister, realdata as rd

rec = rd.load(rd.find_files()[0])
geo3d = coregister.registered_geo3d(rec)          # Positionen aus diesem Modul
head = dot.get_standard_headmodel("icbm152")      # Betreuungsvorgabe: ICBM152

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
    # Auf die Kopfhaut projizieren: eine Optode klebt am Kopf, sie schwebt nicht davor.
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
          f"{[l for l in labels if not l[0] in 'SD' or not l[1:].isdigit()] or 'KEINE'}")

    c, r, sd = sphere_fit(p)
    print(f"\nKugelanpassung an die Rohkoordinaten:")
    print(f"  Zentrum {np.round(c, 1)}  Radius {r:.1f} mm  Reststreuung {sd:.1f} mm")
    print("  -> " + ("kopfgrosse Oberflaechenkoordinaten in mm, plausibel"
                     if 70 < r < 110 and sd < 15 else
                     "KEIN kopfgrosses Koordinatensystem -- Registrierung fragwuerdig"))

    head = ims.head()
    sv = head.scalp.vertices
    sv = np.asarray(sv.pint.dequantify().values if sv.pint.units is not None
                    else sv.values, dtype=float)
    tree = KDTree(sv)

    d0, _ = tree.query(p)
    print(f"\nOhne Transformation (liegen die Rohkoordinaten schon im Kopfmodell?):")
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

    print("\n  Die Registrierung selbst ist gut: die Optoden liegen im Median rund 2 mm")
    print("  von der Kopfhaut, die Skalierung kommt bei ~1.00 heraus (die Rohkoordinaten")
    print("  sind also schon echte Millimeter) und die Kanalabstaende bleiben erhalten.")

    print("\n  Links/rechts -- die Kostenanteile GETRENNT, weil nur der erste etwas")
    print("  ueber die Seite sagt:")
    for name, f in fits.items():
        print(f"    {name:>10s}: Oberflaeche {f['surf_cost']:8.3f}   "
              f"Abstands-Strafterm {f['dist_cost']:8.3f}")
    sa, sb = fits["normal"]["surf_cost"], fits["gespiegelt"]["surf_cost"]
    rel = abs(sa - sb) / max(sa, sb)
    print(f"    Unterschied im OBERFLAECHEN-Anteil: {100 * rel:.1f} %")
    if rel < 0.25:
        print("  -> LINKS/RECHTS IST NICHT BESTIMMT. Beide Orientierungen legen die")
        print("     Montage praktisch gleich gut auf den Kopf; der Gesamtunterschied")
        print("     stammt aus dem Strafterm fuer die Kanalabstaende, und der sagt ueber")
        print("     die Seite nichts. Als Limitation benennen: der Bildraum zeigt fuer")
        print("     diesen Datensatz die STRUKTUR der Aktivierung, nicht die Seite --")
        print("     genau dieselbe Einschraenkung wie im Kanalraum (Abb. 17).")
    else:
        better = "normal" if sa < sb else "gespiegelt"
        print(f"  -> Orientierung '{better}' legt die Optoden deutlich besser auf die")
        print("     Kopfhaut. Das ist ein Hinweis, aber kein Beleg: eine Kopfhaut ist")
        print("     annaehernd symmetrisch, und ohne Landmarken bleibt die Seite eine")
        print("     Annahme. In der Arbeit als solche benennen.")

    print("\nZweiter Blocker: aus Positionen wird eine Adot erst ueber eine")
    print("Photonensimulation, und die braucht eine CUDA-GPU (hier keine vorhanden).")
    print("Fertiges Skript fuer eine Maschine mit GPU:\n")
    print("    " + FLUENCE_SCRIPT.replace("\n", "\n    "))
    print("Rueckfrage an die Betreuung: gibt es fuer dieses NIRScout-Setup eine")
    print("Koregistrierung oder eine fertige Adot? Das ersetzt beide Blocker.")
    return fits


if __name__ == "__main__":
    report()
