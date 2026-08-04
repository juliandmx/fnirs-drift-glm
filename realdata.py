"""Stufe 2: reale Daten -- Einlesen, Inventar, Vorverarbeitung.

Datensatz (Stand 2026-08-04): Khan, Nazeer & Mirtaheri (2026), "Open access
individual finger movement dataset with fNIRS", Front. Hum. Neurosci.,
doi:10.3389/fnhum.2026.1747655. Daten auf figshare, Format SNIRF.

Eckdaten aus dem Paper (vor dem Einlesen zu pruefen -- `inventory()` verifiziert sie):
  * Aufgabe   : Einzelfinger-Tapping rechte Hand (Daumen..kleiner Finger), 10 s pro
                Block, Ruhe dazwischen 10 s (Proband S25: 15 s), Start-Ruhe 20 s,
                drei Wiederholungen je Durchgang, 350 s Gesamtdauer
  * Geraet    : NIRScout, 16 Quellen x 16 Detektoren = 48 Kanaele, 760/850 nm,
                3.9063 Hz (S25: 10.1725 Hz), Optodenabstand einheitlich 3 cm
  * Probanden : 25 (19 m, 6 w), rechtshaendig
  * Aux       : nur Trigger -- KEIN Accelerometer, KEIN dark signal

WAS DAS FUER DIE STUDIE BEDEUTET (in der Arbeit zu benennen):
  * KEINE Short-Separation-Kanaele. Alle Kanaele liegen bei 3 cm, es gibt nichts
    Kuerzeres. Die Konstellationen `short_avg`/`short_maxcorr` sind hier nicht
    anwendbar -- sie bleiben ein Befund der Simulation. Als systemisches Surrogat
    bleibt der `global`-Regressor.
  * KEINE Bewegungs-Aux. Die Konstellation `motion` entfaellt ebenfalls; die
    Motion-Correction-Achse (wavelet / tddr+wavelet) bleibt dagegen anwendbar.
  * DAFUER 25 Probanden statt einem -> erstmals Gruppenstatistik statt Einzelfall.

WICHTIG -- welche Dateien verwenden: Der Datensatz liegt in zwei Fassungen vor,
`*_TRIM` (unverarbeitet) und `*_TRIM_CC_filtered` (von den Autoren mit Satori
vorverarbeitet). Die gefilterte Fassung hat einen **Butterworth-Hochpass bei 0.01 Hz**
bereits angewandt -- also genau den Drift entfernt, dessen Modellierung diese Arbeit
untersucht. Fuer den Familienvergleich ist ausschliesslich die UNVERARBEITETE Fassung
brauchbar. Die gefilterte ist trotzdem wertvoll: sie ist eine fertige Referenz fuer den
Filter-Arm (Betreuungsvorgabe "high (0,01) und lowpass (0,5) ... entweder Driftregressor
oder Highpassfilter").

Aufruf:
    conda run -n cedalion python realdata.py            # Inventar aller gefundenen Dateien
    conda run -n cedalion python realdata.py <datei>    # Inventar + Vorverarbeitung einer Datei
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cedalion
import cedalion.io
import cedalion.nirs
from cedalion import units

import preprocess as prep

# Ablage der realen Daten, ausserhalb des Code-Repos (nicht mitversionieren).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Dateinamen-Konvention des Datensatzes: SXYRZ_TRIM(_CC_filtered).snirf
FILTERED_MARKER = "filtered"


def find_files(data_dir: Path | str = DATA_DIR, include_filtered: bool = False):
    """Alle SNIRF-Dateien, standardmaessig OHNE die vorgefilterte Fassung.

    Die gefilterte Fassung enthaelt bereits einen Hochpass bei 0.01 Hz und ist fuer den
    Driftregressor-Vergleich unbrauchbar (der Filter hat den Drift schon entfernt).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    files = sorted(data_dir.rglob("*.snirf"))
    if not include_filtered:
        files = [f for f in files if FILTERED_MARKER not in f.name.lower()]
    return files


def load(path: Path | str):
    """Liest eine SNIRF-Datei. Gibt das erste NIRS-Element als Recording zurueck."""
    recs = cedalion.io.read_snirf(Path(path))
    if not recs:
        raise ValueError(f"Keine NIRS-Elemente in {path}")
    return recs[0]


def inventory(rec, name: str = "") -> dict:
    """Kennzahlen eines Recordings -- prueft die Paper-Angaben gegen die Datei."""
    key = "amp" if "amp" in rec.timeseries else list(rec.timeseries.keys())[0]
    ts = rec[key]
    t = ts.time.values
    fs = 1.0 / float(np.median(np.diff(t)))

    d = cedalion.nirs.channel_distances(ts, rec.geo3d).pint.to("mm")
    d = np.asarray(d.pint.dequantify().values, dtype=float)

    stim = getattr(rec, "stim", None)
    trial_types, n_events = [], 0
    if stim is not None and len(stim) > 0:
        trial_types = sorted(str(x) for x in stim.trial_type.unique())
        n_events = int(len(stim))

    return {
        "name": name or "?",
        "key": key,
        "n_channels": int(ts.sizes["channel"]),
        "n_wavelengths": int(ts.sizes.get("wavelength", 0)),
        "wavelengths": [float(w) for w in ts.wavelength.values]
        if "wavelength" in ts.coords else [],
        "fs_hz": round(fs, 4),
        "duration_s": round(float(t[-1] - t[0]), 1),
        "dist_min_mm": round(float(d.min()), 2),
        "dist_med_mm": round(float(np.median(d)), 2),
        "dist_max_mm": round(float(d.max()), 2),
        "n_short_18mm": int((d < 18.0).sum()),
        "n_events": n_events,
        "trial_types": trial_types,
        "aux": sorted(rec.aux_ts.keys()) if rec.aux_ts else [],
    }


def amp_range_from_data(rec, lo_pct: float = 0.5, hi_headroom: float = 0.98):
    """Geraeteabhaengige Amplitudengrenzen (dunkel/gesaettigt) aus den Daten schaetzen.

    Die NinjaNIRS-Grenzen 1e-3..0.84 V aus der Betreuungsvorgabe gelten fuer nn22 und
    sind NICHT uebertragbar -- ein NIRScout hat einen anderen Detektor und eine andere
    Aussteuerung. Ohne Dunkelmessung (dieser Datensatz hat keine) laesst sich der
    Rauschboden nicht direkt bestimmen; deshalb hier eine datengetriebene Naeherung:

      * untere Grenze: `lo_pct`-Perzentil der mittleren Kanalamplitude
      * obere Grenze:  `hi_headroom` x Maximum (faengt an den Anschlag gelaufene Kanaele)

    Rueckgabe (lo, hi) in der Einheit der Amplitude. IMMER zusammen mit dem Histogramm
    aus `inventory_report` beurteilen und im Zweifel von Hand setzen -- diese Heuristik
    ist ein Startpunkt, keine Messung.
    """
    key = "amp" if "amp" in rec.timeseries else list(rec.timeseries.keys())[0]
    a = rec[key].pint.dequantify() if hasattr(rec[key], "pint") else rec[key]
    mp = np.asarray(a.mean("time").values, dtype=float).ravel()
    mp = mp[np.isfinite(mp) & (mp > 0)]
    return float(np.percentile(mp, lo_pct)), float(mp.max() * hi_headroom)


def inventory_report(files=None):
    """Inventar aller Dateien als Tabelle + Konsistenzpruefung gegen das Paper."""
    files = list(files if files is not None else find_files())
    if not files:
        print(f"Keine SNIRF-Dateien in {DATA_DIR}.")
        print("Daten von https://figshare.com/s/6298861b1dc6be936d73 herunterladen")
        print(f"und nach {DATA_DIR} entpacken.")
        return None

    rows = []
    for f in files:
        try:
            rows.append(inventory(load(f), f.name))
        except Exception as e:                                    # noqa: BLE001
            print(f"  ! {f.name}: {type(e).__name__}: {e}")
    if not rows:
        return None

    df = pd.DataFrame(rows)
    cols = ["name", "n_channels", "fs_hz", "duration_s",
            "dist_min_mm", "dist_med_mm", "dist_max_mm", "n_short_18mm", "n_events"]
    print(df[cols].to_string(index=False))

    print("\n--- Abgleich mit den Paper-Angaben ---")
    def check(label, got, want, ok):
        print(f"  {'OK ' if ok else '!! '}{label:22s} {got}  (Paper: {want})")

    ch = sorted(df.n_channels.unique())
    check("Kanaele", ch, "48", ch == [48])
    fsv = sorted(df.fs_hz.unique())
    check("Abtastrate [Hz]", fsv, "3.9063 (S25: 10.1725)",
          all(abs(v - 3.9063) < 0.01 or abs(v - 10.1725) < 0.01 for v in fsv))
    check("Dauer [s]", sorted(df.duration_s.unique())[:3], "~350", True)
    check("min. Abstand [mm]", sorted(df.dist_min_mm.unique())[:3], "30 (einheitlich)",
          bool((df.dist_min_mm > 25).all()))
    n_short = int(df.n_short_18mm.max())
    check("Short-Channels <18mm", n_short, "0 erwartet", n_short == 0)
    if n_short > 0:
        print("     -> unerwartet: es GIBT kurze Kanaele. Dann ist die "
              "Short-Channel-Regression hier doch anwendbar (shortchannel.py).")

    tt = sorted({t for r in rows for t in r["trial_types"]})
    print(f"\nTrial-Typen ({len(tt)}): {tt}")
    aux = sorted({a for r in rows for a in r["aux"]})
    print(f"Aux-Kanaele: {aux if aux else '(keine)'}")
    if not any("accel" in a.lower() or "gyro" in a.lower() for a in aux):
        print("  -> keine Bewegungsdaten: Konstellation 'motion' entfaellt.")
    if not any("dark" in a.lower() for a in aux):
        print("  -> keine Dunkelmessung: Amplitudengrenzen datengetrieben schaetzen "
              "(amp_range_from_data).")
    return df


def preprocess_recording(rec, *, motion_method=prep.DEFAULT_MOTION,
                         snr_threshold: float = 3.0, amp_range=None, dpf: float = 6.0):
    """Preprocessing-Kette (preprocess.py) auf ein reales Recording.

    Identische Reihenfolge wie in der Simulation -- nur die Amplitudengrenzen sind
    geraeteabhaengig und werden, falls nicht vorgegeben, aus den Daten geschaetzt.
    """
    if amp_range is None:
        amp_range = amp_range_from_data(rec)
    return prep.run(rec, motion_method=motion_method, snr_threshold=snr_threshold,
                    amp_range=amp_range, sd_range=(0.0, 4.5), dpf=dpf), amp_range


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        rec = load(path)
        inv = inventory(rec, path.name)
        for k, v in inv.items():
            print(f"  {k:16s}: {v}")
        print("\nVorverarbeitung ...")
        P, amp_range = preprocess_recording(rec)
        print(f"  Amplitudengrenzen (geschaetzt): {amp_range[0]:.3e} .. {amp_range[1]:.3e}")
        print(f"  {inv['n_channels']} roh -> {P.conc.sizes['channel']} verwertbar "
              f"({len(P.dropped)} verworfen)")
    else:
        inventory_report()
