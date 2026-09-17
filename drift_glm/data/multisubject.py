"""Datensatz 3: Multisubject-Fingertapping aus Cedalion, mit echten Short Channels.

Quelle: BIDS-NIRS-Tapping (Rob Luke) ueber `cedalion.data`; Datensatz, Event-Benennung und
Global-Component-Subtraktion folgen Notebook 50b
(`examples/machine_learning/50b_advanced_finger_tapping_lda_classification.ipynb`).
Eckdaten: 5 Probanden, 28 Kanaele (8 kurze bei 7-8 mm, 20 lange bei 33-41 mm), 7,8125 Hz,
2974,5 s je Aufnahme, 760/850 nm, je 30 Trials control / Tapping/Left / Tapping/Right
(Blockdauer 5 s), kein Aux. Landmarken Nz/LPA/RPA sind vorhanden, die Sensitivitaetsmatrix
fuer die Montage ist in Cedalion vorberechnet. Tapping links/rechts liefert eine anatomische
Erwartung (kontralateral: Left -> C4, Right -> C3), die im Bildraum pruefbar ist.

Aufruf:
    conda run -n cedalion python -m drift_glm.data.multisubject            # Inventar
    conda run -n cedalion python -m drift_glm.data.multisubject sub-01     # Inventar + Vorverarbeitung
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import cedalion
import cedalion.data
import cedalion.geometry.landmarks
import cedalion.io
import cedalion.nirs
from cedalion import units

from drift_glm.core import preprocess as prep

#: Trigger-Werte -> Namen wie in Notebook 50b; "15.0" ist ein Sentinel-Event am Aufnahmeende.
EVENT_MAP = {"1.0": "control", "2.0": "Tapping/Left", "3.0": "Tapping/Right",
             "15.0": "sentinel"}

#: Modellierte Bedingungen. Die Ruhe dazwischen ist die implizite Baseline; als eigener
#: Regressor waere sie kollinear mit dem Offset (wie in `realdata.stim_df`).
CONDITIONS = ("control", "Tapping/Left", "Tapping/Right")
TAPPING = ("Tapping/Left", "Tapping/Right")

#: Long/Short-Grenze wie in Notebook 50b; die Abstaende haben eine Luecke zwischen 8 und 33 mm.
SHORT_THRESHOLD = 15.0 * units.mm

#: Kontralaterale Erwartung: Landmarke, unter der die Aktivierung je Hand liegen soll.
EXPECTED_SIDE = {"Tapping/Left": "C4", "Tapping/Right": "C3"}

#: Schluessel der Sensitivitaetsmatrix (siehe `imagespace.ADOT_KEYS`).
DATASET = "multisubject_fingertapping"


def paths() -> list[str]:
    """Pfade der 5 SNIRF-Dateien (laedt beim ersten Aufruf ~42 MB herunter)."""
    return [str(p) for p in cedalion.data.get_multisubject_fingertapping_snirf_paths()]


def subject_of(path) -> str:
    """'.../sub-03/nirs/sub-03_task-tapping_nirs.snirf' -> 'sub-03'."""
    import re
    m = re.search(r"sub-\d+", str(path))
    return m.group(0) if m else str(path)


def load(path):
    """Recording mit sprechenden Events, normierten Landmarken, sortierten Stimuli.

    `normalize_landmarks_labels` bringt "NASION" auf die kanonische Schreibweise "Nz" --
    ohne das findet die Registrierung die Landmarke nicht.
    """
    rec = cedalion.io.read_snirf(str(path))[0]
    rec.stim.cd.rename_events(EVENT_MAP)
    rec.stim = (rec.stim[rec.stim.trial_type.isin(list(CONDITIONS))]
                .sort_values("onset").reset_index(drop=True))
    rec.geo3d = cedalion.geometry.landmarks.normalize_landmarks_labels(rec.geo3d)
    return rec


def stim_df(rec, mode: str = "hands") -> pd.DataFrame:
    """Stimulus-Tabelle.

    `"hands"` haelt beide Haende getrennt (Lateralisierung, Bildraum); `"tapping"` fasst
    sie zu einem Regressor zusammen (Driftfamilien-Vergleich, wie `realdata.stim_df`).
    """
    df = rec.stim.copy()
    if mode == "tapping":
        df.loc[df.trial_type.isin(TAPPING), "trial_type"] = "Tapping"
    elif mode != "hands":
        raise ValueError(f"mode muss 'hands' oder 'tapping' sein, nicht {mode!r}")
    return df


def inventory(rec, name: str = "") -> dict:
    """Kennzahlen einer Aufnahme -- prueft die Annahmen gegen die Datei."""
    ts = rec["amp"]
    t = ts.time.values
    d = np.asarray(cedalion.nirs.channel_distances(ts, rec.geo3d)
                   .pint.to("mm").pint.dequantify().values, dtype=float)
    srcdet = set(map(str, ts.source.values)) | set(map(str, ts.detector.values))
    lm = [str(x) for x in rec.geo3d.label.values
          if str(x) not in srcdet and not str(x).isdigit()]
    return {
        "name": name or "?",
        "n_channels": int(ts.sizes["channel"]),
        "fs_hz": round(1.0 / float(np.median(np.diff(t))), 4),
        "duration_s": round(float(t[-1] - t[0]), 1),
        "dist_min_mm": round(float(d.min()), 1),
        "dist_max_mm": round(float(d.max()), 1),
        "n_short": int((d < SHORT_THRESHOLD.to("mm").magnitude).sum()),
        "n_events": int(len(rec.stim)),
        "trial_types": sorted(set(map(str, rec.stim.trial_type))),
        "landmarks": sorted(lm),
    }


def inventory_report():
    """Inventar aller 5 Probanden + Konsistenzpruefung."""
    rows = [inventory(load(p), subject_of(p)) for p in paths()]
    df = pd.DataFrame(rows)
    cols = ["name", "n_channels", "fs_hz", "duration_s", "dist_min_mm", "dist_max_mm",
            "n_short", "n_events"]
    print(df[cols].to_string(index=False))

    print("\n--- Pruefung der Annahmen ---")

    def check(label, got, want, ok):
        print(f"  {'OK ' if ok else '!! '}{label:26s} {got}  (erwartet: {want})")

    ch = sorted(df.n_channels.unique())
    check("Kanaele", ch, "28", ch == [28])
    ns = sorted(df.n_short.unique())
    check("echte Short Channels", ns, "8", ns == [8])
    check("kuerzester Abstand [mm]", sorted(df.dist_min_mm.unique()), "< 10",
          bool((df.dist_min_mm < 10).all()))
    check("Landmarken", rows[0]["landmarks"], "Nz, LPA, RPA",
          all(x in rows[0]["landmarks"] for x in ("Nz", "LPA", "RPA")))
    check("Bedingungen", rows[0]["trial_types"], list(CONDITIONS),
          set(rows[0]["trial_types"]) == set(CONDITIONS))
    check("Trials je Bedingung", int(df.n_events.min() / 3), "30",
          bool((df.n_events >= 84).all()))

    print("\nAnwendbar auf diesem Datensatz:")
    print("  * short_avg / short_maxcorr / short_closest mit echten kurzen Kanaelen")
    print("  * Global-Component-Subtraktion nach Notebook 50b")
    print("  * Bildraum (vorberechnete Adot fuer ICBM152)")
    print("  * Lateralisierung: Tapping/Left -> C4, Tapping/Right -> C3")
    return df


def preprocess_recording(rec, *, motion_method: str = prep.DEFAULT_MOTION,
                         snr_threshold: float = prep.DEFAULT_SNR_THRESHOLD,
                         amp_range=None, dpf: float = prep.DEFAULT_DPF):
    """Preprocessing-Kette (`preprocess.py`) auf eine Aufnahme. Rueckgabe (Pre, amp_range).

    Gleiche Reihenfolge wie bei Simulation und Khan-Datensatz. Die Amplituden sind hier
    dimensionslos skaliert und ohne dunkle Population, die datengetriebene Amplitudengrenze
    schaltet sich daher ab; verworfen wird ueber SNR und Kanalabstand.
    """
    if amp_range is None:
        amp_range = prep.amp_range_from_data(rec)
    return prep.run(rec, motion_method=motion_method, snr_threshold=snr_threshold,
                    amp_range=amp_range, sd_range=prep.DEFAULT_SD_RANGE, dpf=dpf), \
        amp_range


if __name__ == "__main__":
    if len(sys.argv) > 1:
        want = sys.argv[1]
        path = next(p for p in paths() if want in p)
        rec = load(path)
        for k, v in inventory(rec, subject_of(path)).items():
            print(f"  {k:14s}: {v}")
        print("\nVorverarbeitung ...")
        P, ar = preprocess_recording(rec)
        print(f"  Amplitudengrenzen: {ar[0]:.3e} .. {ar[1]:.3e}"
              + ("  (abgeschaltet -- keine dunkle Population)"
                 if ar == prep.AMP_RANGE_OFF else ""))
        print(f"  {rec['amp'].sizes['channel']} roh -> {P.conc.sizes['channel']} "
              f"verwertbar ({len(P.dropped)} verworfen: {P.dropped})")
        from drift_glm.core import shortchannel as sc
        long, short = cedalion.nirs.split_long_short_channels(
            P.conc, P.geo3d, distance_threshold=SHORT_THRESHOLD)
        print(f"  Split bei {SHORT_THRESHOLD}: {short.sizes['channel']} kurz, "
              f"{long.sizes['channel']} lang")
        print(f"  kurze Kanaele: {', '.join(map(str, short.channel.values))}")
    else:
        inventory_report()
