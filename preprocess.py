"""Preprocessing-Kette nach Betreuungsvorgabe (Gespraechsnotizen 2026-08-01).

Die Reihenfolge folgt exakt der Vorgabe aus dem Betreuungsgespraech -- und damit auch
der kanonischen Cedalion-Kette (examples/tutorial/3_signal_processing.ipynb:
"quality assessment -> OD conversion -> motion correction -> filtering ->
haemoglobin concentration"):

    Rohamplitude
      -> int2od  (BASELINE merken, sonst ist der Rueckweg nicht moeglich)
      -> Motion Correction auf OD          [Schritt 1.3]
      -> zurueck zur Amplitude via od2int  [Schritt 1.4]
      -> Qualitaetsmasken auf der KORRIGIERTEN Amplitude   [Schritt 1.5]
      -> Pruning, dann weiter auf OD -> od2conc            [Schritt 1.6]

Der entscheidende Punkt der Vorgabe: die Kanalqualitaet (dunkel/gesaettigt) wird an der
AMPLITUDE beurteilt, die Korrektur passiert aber auf OD. Deshalb der Umweg
OD -> Amplitude -> Maske -> zurueck auf OD. Die Baseline ist das, was diesen Rueckweg
ueberhaupt erlaubt: od = -log(amp / baseline), also amp = baseline * exp(-od).

Stand: Schritt 1.2 -- OD-Umrechnung mit Baseline-Rueckgabe. Die weiteren Stufen
kommen schrittweise dazu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

import cedalion
import cedalion.nirs
import cedalion.sigproc.motion as motion
import cedalion.sigproc.quality as quality
from cedalion import units


@dataclass
class Preprocessed:
    """Ergebnis der Preprocessing-Kette samt Zwischenstufen fuer die Diagnose."""

    conc: xr.DataArray       # (time, channel, chromo) [µM], dequantifiziert
    od: xr.DataArray         # Optical Density, Stand nach Korrektur/Pruning
    amp_raw: xr.DataArray    # Rohamplitude [V], wie eingelesen (quantifiziert)
    amp_corr: xr.DataArray   # Amplitude NACH Motion Correction [V] -- Basis der Masken
    baseline: xr.DataArray   # mittlere Rohamplitude (channel, wavelength) [V]
    geo3d: object            # Optodengeometrie (LabeledPoints)
    aux: object              # rec.aux_ts (Accelerometer/Gyroskop/dark signal)
    dropped: list[str] = field(default_factory=list)   # entfernte Kanaele
    masks: dict = field(default_factory=dict)          # Einzelmasken zur Diagnose
    motion_method: str = "none"                        # angewandte Motion Correction
    od_uncorrected: xr.DataArray | None = None         # OD vor der Korrektur (Diagnose)


def gate_positive(amp: xr.DataArray) -> tuple[xr.DataArray, list[str]]:
    """Verwirft Kanaele mit nicht-positiver Amplitude -- Vorbedingung fuer int2od.

    `int2od` bildet -log(amp/baseline) und bricht bei Werten <= 0 mit einer
    AssertionError ab. Physikalisch ist eine nicht-positive Lichtintensitaet ohnehin
    unmoeglich; solche Samples liegen unter dem Rauschboden des Detektors.

    Das ist bewusst KEINE methodische Vorentscheidung, sondern die minimale technische
    Vorbedingung: auf nn22 betrifft es 40 von 3.75 Mio. Samples (0.001 %) in 6 von 567
    Kanaelen, und alle 6 werden von der spaeteren mean_amp-Grenze (1e-3 V) ebenfalls
    verworfen -- die Vor-Maskierung nimmt also nichts weg, was sonst ueberlebt haette.
    Die eigentliche Qualitaetsbewertung passiert weiterhin erst nach der Motion
    Correction auf der korrigierten Amplitude (Betreuungsvorgabe).
    """
    ok = (amp > 0).all(dim=[d for d in amp.dims if d != "channel"])
    dropped = [str(c) for c in amp.channel.values[~ok.values]]
    return amp.sel(channel=ok), dropped


def to_od(amp: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Amplitude -> Optical Density, mit Baseline.

    Cedalion bietet das direkt an (nirs/cw.py): `int2od(amp, return_baseline=True)`
    liefert `(od, baseline)` mit `baseline = amp.mean("time")`. Der Parameter heisst
    `return_baseline`, NICHT `baseline=`.

    Die Baseline wird gebraucht, um nach der Motion Correction wieder auf die Amplitude
    zu kommen (`od2int(od, baseline)`), denn nur dort sind "dunkel" und "gesaettigt"
    ueberhaupt definierte Begriffe.
    """
    return cedalion.nirs.cw.int2od(amp, return_baseline=True)


MOTION_METHODS = ("none", "tddr", "wavelet", "tddr+wavelet")

# Default fuer Demos/Tests. Im Sweep ist die Motion Correction eine eigene ACHSE
# ({"wavelet", "tddr+wavelet"}), der Default entscheidet dort also nichts.
#
# Warum "wavelet" und nicht die woertliche Vorgabe "tddr+wavelet": TDDR daempft
# gemessen das Driftband (<0.01 Hz) auf 55.6 % -- es entfernt fast die Haelfte dessen,
# was die Driftregressoren modellieren sollen, und kollidiert damit mit dem
# Betreuungshinweis vom 2026-07-11 (bei Drift-Modellierung nicht hochpassfiltern).
# Wavelet ist im Driftband neutral (100.0 %) und entfernt trotzdem die Spikes.
# Nachpruefbar mit `band_power_ratio` bzw. `python preprocess.py tddr+wavelet`.
DEFAULT_MOTION = "wavelet"

# Die uebrigen Vorverarbeitungs-Parameter -- an EINER Stelle, damit sie nicht
# auseinanderlaufen. Aufrufer (pipeline.py, realdata.py) verweisen hierauf, statt die
# Zahlen zu wiederholen: sonst behaelt ein Aufrufer beim Aendern still den alten Wert,
# und weil er ihn explizit durchreicht, gewinnt der alte.
DEFAULT_SNR_THRESHOLD = 3.0                  # Betreuungsvorgabe (vorher 10)
DEFAULT_AMP_RANGE = (1e-3, 0.84)             # dunkel / gesaettigt [V], NinjaNIRS-Vorgabe
DEFAULT_SD_RANGE = (0.0, 4.5)                # Quell-Detektor-Abstand [cm]
DEFAULT_DPF = 6.0                            # differentieller Pfadlaengenfaktor


def motion_correct(
    od: xr.DataArray,
    method: str = "tddr+wavelet",
    *,
    wavelet_iqr: float = 1.5,
    wavelet_name: str = "db2",
    wavelet_level: int = 4,
) -> xr.DataArray:
    """Motion Correction auf Optical Density.

    Die Betreuungsvorgabe nennt zwei Artefakttypen, die entfernt werden sollen:
    "scharfer Spike oder ruckartige Verschiebung". Genau darauf zielen die beiden
    Verfahren, und daher auch ihre Reihenfolge (identisch zu Cedalion NB 25:
    "apply TDDR first to correct jumps, then apply Wavelet motion artifact correction"):

      * `tddr`    -- Temporal Derivative Distribution Repair: robuste Regression auf der
                     zeitlichen Ableitung; faengt Baseline-Spruenge / ruckartige
                     Verschiebungen. Parameterfrei (`motion.tddr(ts)`).
      * `wavelet` -- verwirft Wavelet-Koeffizienten ausserhalb des IQR-Bandes; faengt
                     scharfe Spikes. Groesseres `iqr` = drastischere Korrektur;
                     `iqr < 0` laesst das Signal unveraendert.

    Beide arbeiten laut Cedalion ausdruecklich auf OD, nicht auf Amplitude oder
    Konzentration ("The correction algorithms operate on optical densities").
    """
    if method not in MOTION_METHODS:
        raise ValueError(f"Unbekannte Motion-Correction: {method!r} "
                         f"(erlaubt: {MOTION_METHODS})")
    if method == "none":
        return od
    if "tddr" in method:
        od = motion.tddr(od)
    if "wavelet" in method:
        od = motion.wavelet(od, iqr=wavelet_iqr, wavelet=wavelet_name,
                            level=wavelet_level)
    return od


def to_amp(od: xr.DataArray, baseline: xr.DataArray) -> xr.DataArray:
    """Optical Density -> Amplitude, mit der beim Hinweg gemerkten Baseline.

    Das ist der Kern der Betreuungsvorgabe "erst motion correction, dann zu amplitude
    umwandeln und dann schlechte channels markieren": korrigiert wird auf OD, bewertet
    wird auf der Amplitude. `od2int(od, baseline)` = `baseline * exp(-od)` ist die exakte
    Umkehrung von `int2od` -- ohne die Baseline waere der Rueckweg nicht eindeutig, weil
    OD nur relative Aenderungen gegenueber dem eigenen Mittel kodiert.

    Wichtig: die Baseline stammt aus der UNKORRIGIERTEN Amplitude. Die zurueckgerechnete
    Amplitude traegt also die Motion-Korrektur, behaelt aber das urspruengliche
    Helligkeitsniveau -- genau das, worauf "dunkel" und "gesaettigt" sich beziehen.
    """
    return cedalion.nirs.cw.od2int(od, baseline)


def quality_masks(
    amp: xr.DataArray,
    geo3d,
    *,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    amp_range: tuple[float, float] = DEFAULT_AMP_RANGE,
    sd_range: tuple[float, float] = DEFAULT_SD_RANGE,
) -> dict[str, xr.DataArray]:
    """Qualitaetsmasken auf der (korrigierten) Amplitude. CLEAN = True.

    Die drei Kriterien adressieren verschiedene Defekte und ersetzen einander nicht:

      * `snr`      -- Verhaeltnis Mittelwert/Streuung ueber die Zeit. Faengt verrauschte
                      Kanaele. Betreuungsvorgabe: Schwelle 3 (bisher 10). Der neue Wert
                      ist PERMISSIVER; die eigentliche Arbeit macht jetzt `mean_amp`.
      * `mean_amp` -- mittlere Amplitude innerhalb eines Fensters. Faengt DUNKLE (zu wenig
                      Licht, Rauschen dominiert) und GESAETTIGTE Kanaele (Detektor am
                      Anschlag, Signal geklippt). Vorgabe: NinjaNIRS-Grenzen
                      1e-3 .. 0.84 V. Basiert auf Homer3 `hmR_PruneChannels.m`.
      * `sd_dist`  -- Quell-Detektor-Abstand innerhalb eines Bereichs.

    Gesaettigte Kanaele sind besonders heimtueckisch: durch das Klippen wirken sie
    RAUSCHARM, weshalb varianzbasierte Metriken sie nicht erkennen (Cedalion NB 24:
    "the metric cannot account for saturation"). Bei der Image Reconstruction bekommen
    sie deshalb maximales Gewicht in der Pseudoinversen und schmieren ihren Fehler ueber
    ihr gesamtes Sensitivitaetsprofil -- daher die Betreuungsvorgabe, sie spaetestens
    dort zwingend zu entfernen.
    """
    _, snr_mask = quality.snr(amp, snr_threshold)
    _, amp_mask = quality.mean_amp(amp, (amp_range[0] * units.V,
                                         amp_range[1] * units.V))
    _, sd_mask = quality.sd_dist(amp, geo3d, (sd_range[0] * units.cm,
                                              sd_range[1] * units.cm))
    return {"snr": snr_mask, "mean_amp": amp_mask, "sd_dist": sd_mask}


def dark_noise_floor(aux, key: str = "dark signal") -> float | None:
    """Robuster Rauschboden des Detektors aus der Dunkelmessung [V], oder None.

    nn22 fuehrt eine Dunkelmessung als Aux-Zeitreihe mit (Schreibweise mit LEERZEICHEN:
    "dark signal", nicht "dark_signal"), mit 1134 Spuren = 567 Kanaele x 2 Wellenlaengen.
    Die Werte streuen um Null -- es ist eine RAUSCH-Referenz, kein Pegel. Deshalb wird
    die Streuung ausgewertet und nicht der Mittelwert; robust ueber MAD, damit einzelne
    defekte Detektoren den Wert nicht anheben.

    Nutzen: die Untergrenze fuer "dunkel" (Vorgabe 1e-3 V) laesst sich damit datengetrieben
    einordnen statt als Faustwert. Auf nn22 ergibt sich ein Rauschboden von ~9.5e-06 V,
    die Vorgabe entspricht also dem ~105-fachen davon -- und liegt in einer Luecke: von
    50x bis 105x Rauschboden faellt kein einziger weiterer Kanal heraus. Die Schwelle
    trennt somit zwei klar getrennte Populationen und ist unempfindlich gegen ihre genaue
    Lage.

    Die Zuordnung der Aux-Spuren zu (Kanal, Wellenlaenge) ist NICHT belegt -- es gibt
    keine aux_channel-Koordinate. Der Wert wird daher nur aggregiert verwendet.
    """
    if aux is None or key not in aux:
        return None
    d = aux[key]
    try:
        d = d.pint.dequantify()
    except Exception:
        pass
    v = np.asarray(d.values, dtype=float)
    ax = list(d.dims).index("time")
    mad = 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v, axis=ax, keepdims=True)),
                                axis=ax)
    return float(np.nanmedian(mad))


def prune(ts: xr.DataArray, masks: dict[str, xr.DataArray]) -> tuple[xr.DataArray, list[str]]:
    """Wendet die kombinierten Qualitaetsmasken an und VERWIRFT die Kanaele.

    Cedalions `prune_ch(ts, masks, "all")` verknuepft die Masken mit `&` und ruft
    intern `apply_mask(..., "drop", dim_collapse="channel")` auf: ein Kanal faellt
    heraus, sobald er in IRGENDEINER uebrigen Dimension (hier: einer der beiden
    Wellenlaengen) als TAINTED markiert ist.

    Verworfen wird bewusst, nicht auf NaN gesetzt: NaN wuerde sich durch AR-IRLS und
    spaeter durch die Image Reconstruction fortpflanzen, und deren Kanalauswahl erwartet
    ohnehin eine Teilmenge ("y may contain less channels then W due to pruning").

    Angewandt wird auf OD -- entsprechend der Vorgabe "dann wieder mit od weiterarbeiten".
    """
    ts_pruned, dropped = quality.prune_ch(ts, list(masks.values()), "all")
    return ts_pruned, [str(c) for c in np.atleast_1d(dropped)]


DRIFT_BANDS = (("Drift   <0.01 Hz", 0.0, 0.01),
               ("0.01-0.1 Hz     ", 0.01, 0.1),
               ("0.1-0.5 Hz      ", 0.1, 0.5),
               ("Kardial >0.5 Hz ", 0.5, np.inf))


def band_power_ratio(od_before: xr.DataArray, od_after: xr.DataArray) -> dict:
    """Leistung je Frequenzband NACH der Korrektur relativ zu VORHER (Median).

    Diagnose fuer die zentrale methodische Frage dieser Arbeit: greift die Motion
    Correction in das Driftband ein? Ein Verfahren, das unterhalb 0.01 Hz Leistung
    entfernt, nimmt genau den Anteil weg, den die Driftregressoren modellieren sollen --
    dann bestimmt die Vorverarbeitung das Ergebnis statt des Driftmodells, und der
    Familienvergleich wird verfaelscht (Betreuungshinweis 2026-07-11).

    Rueckgabe: {Bandname: Verhaeltnis}, 1.0 = unveraendert.
    """
    fs = 1.0 / float(np.median(np.diff(od_before.time.values)))
    out = {}
    a0 = np.asarray(od_before.values, float).reshape(-1, od_before.sizes["time"])
    a1 = np.asarray(od_after.values, float).reshape(-1, od_after.sizes["time"])
    a0 = a0 - a0.mean(-1, keepdims=True)
    a1 = a1 - a1.mean(-1, keepdims=True)
    freq = np.fft.rfftfreq(a0.shape[-1], d=1.0 / fs)
    P0 = np.abs(np.fft.rfft(a0, axis=-1)) ** 2
    P1 = np.abs(np.fft.rfft(a1, axis=-1)) ** 2
    for name, lo, hi in DRIFT_BANDS:
        m = (freq >= lo) & (freq < hi)
        out[name] = float(np.median(P1[:, m].sum(-1) / (P0[:, m].sum(-1) + 1e-30)))
    return out


#: Amplitudenbereich, der nichts verwirft -- fuer Datensaetze ohne dunkle Population.
AMP_RANGE_OFF = (0.0, 1e12)


def amp_range_from_data(rec, min_gap: float = 3.0, max_share: float = 0.1):
    """Amplitudengrenzen (dunkel/gesaettigt) aus den Daten -- oder bewusst keine.

    Die NinjaNIRS-Grenzen 1e-3..0.84 V aus der Betreuungsvorgabe gelten fuer nn22 und
    sind NICHT uebertragbar: anderes Geraet, andere Aussteuerung, und andere Datensaetze
    haben keine Dunkelmessung, aus der sich ein Rauschboden ableiten liesse.

    Statt einer Perzentil-Faustregel -- die per Konstruktion IMMER etwas verwirft, egal
    wie gut die Daten sind -- wird hier geprueft, ob es ueberhaupt eine ABGETRENNTE
    dunkle Population gibt: die Kanalamplituden werden sortiert und die groesste
    relative Luecke zwischen benachbarten Werten im unteren Bereich gesucht. Nur wenn
    diese Luecke mindestens `min_gap` betraegt und hoechstens `max_share` der Messungen
    darunter liegen, wird dort geschnitten.

    Auf BEIDEN Realdatensaetzen greift das bewusst NICHT, und das ist das Ergebnis, nicht
    ein Versagen. Khan (NIRScout): dunkelste Messung beim 0.116-fachen des Medians,
    groesste Luecke Faktor 1.11 ueber 6624 Messungen -- passend dazu steht in
    "Experimental notes.txt" "Masked channels removed". Multisubject-Fingertapping:
    dunkelste Messung beim 0.21-fachen des Medians, ebenfalls lueckenlos. Zum Vergleich
    nn22: dunkelste Messung beim 0.0001-fachen des Medians, klare Luecke zwischen 50x und
    105x Rauschboden, 46 Kanaele verworfen.

    Rueckgabe (lo, hi); `AMP_RANGE_OFF`, wenn keine Population gefunden wird.
    """
    key = "amp" if "amp" in rec.timeseries else list(rec.timeseries.keys())[0]
    a = rec[key].pint.dequantify() if hasattr(rec[key], "pint") else rec[key]
    mp = np.asarray(a.mean("time").values, dtype=float).ravel()
    mp = np.sort(mp[np.isfinite(mp) & (mp > 0)])
    if mp.size < 10:
        return AMP_RANGE_OFF
    lower = mp[: max(int(mp.size * max_share), 1) + 1]
    if lower.size < 2:
        return AMP_RANGE_OFF
    ratios = lower[1:] / lower[:-1]
    i = int(np.argmax(ratios))
    if ratios[i] < min_gap:
        return AMP_RANGE_OFF                     # keine abgetrennte dunkle Population
    return float(np.sqrt(lower[i] * lower[i + 1])), 1e12    # Schnitt in die Luecke


def to_conc(od: xr.DataArray, geo3d, dpf: float = DEFAULT_DPF) -> xr.DataArray:
    """Optical Density -> Haemoglobinkonzentration [µM], dequantifiziert."""
    dpf_da = xr.DataArray(
        [dpf] * od.sizes["wavelength"],
        dims="wavelength",
        coords={"wavelength": od.wavelength},
    )
    conc = cedalion.nirs.cw.od2conc(od, geo3d, dpf_da, spectrum="prahl")
    return conc.pint.to("uM").pint.dequantify()


@dataclass
class ODStage:
    """Zwischenstand: Ruhedaten als Optical Density, VOR der Motion Correction.

    Genau hier wird die synthetische Aktivierung eingemischt (`to_od_activation`),
    damit die Motion Correction anschliessend ueber Signal UND Rauschen laeuft -- so
    wie auf echten Daten. Wuerde man erst danach einmischen, koennte die Korrektur die
    HRF per Konstruktion nicht beschaedigen, und ein Verfahren wie TDDR saehe kuenstlich
    gut aus (gemessen: es daempft das Band 0.01-0.1 Hz, in dem die HRF liegt, auf 54 %).
    """

    od: xr.DataArray         # (channel, wavelength, time), ungeprunt, unkorrigiert
    baseline: xr.DataArray
    amp_raw: xr.DataArray
    geo3d: object
    aux: object
    dropped_nonpositive: list[str]


def to_od_stage(rec) -> ODStage:
    """Rohamplitude -> Optical Density (inkl. Positivitaets-Gate und Baseline)."""
    # nn22 liefert die Amplitude dimensionslos -> als Volt quantifizieren, damit die
    # spaeteren Amplitudengrenzen (dunkel/gesaettigt) eine physikalische Einheit haben.
    amp_raw = rec["amp"].pint.dequantify().pint.quantify("V")
    amp, dropped_nonpos = gate_positive(amp_raw)
    od, baseline = to_od(amp)
    return ODStage(od=od, baseline=baseline, amp_raw=amp_raw, geo3d=rec.geo3d,
                   aux=rec.aux_ts, dropped_nonpositive=dropped_nonpos)


def to_od_activation(activation_conc: xr.DataArray, geo3d, wavelength,
                     dpf: float = DEFAULT_DPF) -> xr.DataArray:
    """Konzentrations-Aktivierung [µM] -> Optical Density, zum Einmischen.

    `conc2od` ist die exakte Umkehrung von `od2conc` (beides das modifizierte
    Beer-Lambert-Gesetz). Dadurch bleibt die Ground Truth in µM definiert und
    interpretierbar: ohne Motion Correction ergibt der Weg
    conc -> od -> (nichts) -> conc die Aktivierung exakt zurueck. Weicht sie ab, ist
    das genau der Eingriff der Korrektur -- und damit die Groesse, die gemessen werden soll.

    Args:
        activation_conc: (time, channel, chromo) in µM, dequantifiziert.
        geo3d: Optodengeometrie (fuer die Kanalabstaende).
        wavelength: Wellenlaengen-Koordinate der Ziel-OD.
        dpf: Differentieller Pfadlaengenfaktor, identisch zu `to_conc`.
    """
    dpf_da = xr.DataArray([dpf] * len(wavelength), dims="wavelength",
                          coords={"wavelength": wavelength})
    conc = activation_conc
    if conc.pint.units is None:
        conc = conc.pint.quantify("uM")
    return cedalion.nirs.cw.conc2od(conc, geo3d, dpf_da, spectrum="prahl")


def finish(
    stage: ODStage,
    od_in: xr.DataArray | None = None,
    *,
    motion_method: str = DEFAULT_MOTION,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    amp_range: tuple[float, float] = DEFAULT_AMP_RANGE,
    sd_range: tuple[float, float] = DEFAULT_SD_RANGE,
    dpf: float = DEFAULT_DPF,
    masks: dict[str, xr.DataArray] | None = None,
) -> Preprocessed:
    """Zweite Haelfte der Kette: Motion Correction -> Amplitude -> Masken -> Pruning -> Konzentration.

    Args:
        stage: Ergebnis von `to_od_stage`.
        od_in: die zu verarbeitende OD. Default `stage.od` (reine Ruhedaten); fuer die
            Augmentation wird hier `stage.od + Aktivierung` uebergeben.
        motion_method: eines aus `MOTION_METHODS`. `"none"` schaltet die Korrektur ab.
        snr_threshold: SNR-Schwelle (Betreuungsvorgabe: 3).
        amp_range: (dunkel, gesaettigt) in Volt (NinjaNIRS-Vorgabe: 1e-3 .. 0.84).
        sd_range: zulaessiger Quell-Detektor-Abstand in cm.
        dpf: Differentieller Pfadlaengenfaktor fuer die Beer-Lambert-Umrechnung.
        masks: vorgegebene Masken statt neu berechneter. Gebraucht, damit die
            augmentierte und die reine Variante EXAKT dieselben Kanaele behalten --
            die Kanalqualitaet ist eine Eigenschaft der Messung, nicht des
            eingemischten Signals.
    """
    od_raw = stage.od if od_in is None else od_in
    baseline, dropped_nonpos = stage.baseline, stage.dropped_nonpositive
    amp_raw = stage.amp_raw
    od = motion_correct(od_raw, motion_method)
    # Zurueck zur Amplitude: dort -- und nur dort -- sind "dunkel" und "gesaettigt"
    # definiert. Die Qualitaetsmasken (Schritt 1.5) setzen auf amp_corr auf.
    amp_corr = to_amp(od, baseline)

    if masks is None:
        masks = quality_masks(amp_corr, stage.geo3d, snr_threshold=snr_threshold,
                              amp_range=amp_range, sd_range=sd_range)

    # Erst jetzt verwerfen -- und danach wieder auf OD weiterarbeiten (Vorgabe).
    # amp_raw/amp_corr bleiben ungeprunt: sie dokumentieren die Stufe, AUF der die
    # Masken bestimmt wurden.
    od_pruned, dropped_quality = prune(od, masks)
    conc = to_conc(od_pruned, stage.geo3d, dpf)
    # Die unkorrigierte OD auf dieselben Kanaele beschneiden, sonst vergleicht die
    # Diagnose (band_power_ratio) unterschiedliche Kanalmengen.
    od_raw = od_raw.sel(channel=od_pruned.channel)

    return Preprocessed(
        conc=conc,
        od=od_pruned,
        amp_raw=amp_raw,
        amp_corr=amp_corr,
        baseline=baseline,
        geo3d=stage.geo3d,
        aux=stage.aux,
        dropped=list(dropped_nonpos) + dropped_quality,
        masks={"nonpositive": dropped_nonpos, **masks},
        motion_method=motion_method,
        od_uncorrected=od_raw,
    )


def run(rec, **kwargs) -> Preprocessed:
    """Komplette Kette auf einem Recording, ohne Augmentation.

    Bequemlichkeits-Wrapper um `to_od_stage` + `finish`. Fuer die Augmentation wird
    stattdessen `to_od_stage` -> Aktivierung einmischen -> `finish` benutzt, damit die
    Motion Correction die HRF mit sieht (siehe `ODStage`).
    """
    return finish(to_od_stage(rec), **kwargs)


if __name__ == "__main__":
    import sys
    import time

    import cedalion.data

    method = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MOTION
    t0 = time.time()
    P = run(cedalion.data.get_nn22_resting_state(), motion_method=method)
    bl = np.asarray(P.baseline.pint.dequantify().values, dtype=float)
    print(f"Kanaele roh        : {P.amp_raw.sizes['channel']}")
    print(f"  nicht positiv    : {len(P.masks['nonpositive'])} verworfen "
          f"{P.masks['nonpositive']}")
    print(f"Kanaele            : {P.conc.sizes['channel']}")
    print(f"Zeitpunkte         : {P.conc.sizes['time']}")
    print(f"Baseline [V]       : min {bl.min():.3e}  median {np.median(bl):.3e}  "
          f"max {bl.max():.3e}")
    print(f"OD                 : {dict(P.od.sizes)}")
    print(f"Konzentration [µM] : {dict(P.conc.sizes)}")

    # Rueckweg-Kontrolle: ohne Korrektur muss od2int(int2od(amp)) == amp gelten.
    # Damit ist belegt, dass die Baseline den Rueckweg exakt traegt.
    amp_in = gate_positive(P.amp_raw)[0].pint.dequantify().values
    amp_out = P.amp_corr.pint.dequantify().values
    rel = np.abs(amp_out - amp_in) / np.abs(amp_in)
    hi, med = float(np.nanmax(rel)), float(np.nanmedian(rel))
    if hi < 1e-9:
        print(f"\nRueckweg OD->Amp   : max. rel. Abweichung {hi:.2e} -- exakt "
              f"(Baseline traegt den Rueckweg verlustfrei)")
    else:
        # Erwartet, sobald korrigiert wurde: die Abweichung IST die Korrektur.
        # Im Amplitudenraum wirkt sie exponentiell (amp = baseline * exp(-od)),
        # eine OD-Aenderung von d entspricht dem Faktor exp(d) -- das Maximum wird
        # daher von einzelnen Spikes in dunklen Kanaelen dominiert.
        print(f"\nRueckweg OD->Amp   : median {100 * med:.2f} %, max {hi:.2e} "
              f"(entspricht {np.log(hi + 1):.1f} OD) -- das ist die Korrektur selbst")

    print(f"\nMotion Correction  : {P.motion_method}   ({time.time() - t0:.1f}s gesamt)")
    if P.od_uncorrected is not None and method != "none":
        d1 = np.abs(np.diff(np.asarray(P.od_uncorrected.values, float), axis=-1))
        d2 = np.abs(np.diff(np.asarray(P.od.values, float), axis=-1))
        print(f"  groesster Sprung : {d1.max():.4f} -> {d2.max():.4f} OD "
              f"(Spitzen der zeitlichen Ableitung)")
        print("  Restleistung je Band (100 % = unveraendert):")
        for name, r in band_power_ratio(P.od_uncorrected, P.od).items():
            flag = "  <-- Driftband!" if name.startswith("Drift") and r < 0.9 else ""
            print(f"    {name} {100 * r:6.1f} %{flag}")

    nf = dark_noise_floor(P.aux)
    if nf is not None:
        mp = P.amp_raw.mean("time").pint.dequantify().values.ravel()
        print(f"\nDunkelmessung      : Rauschboden {nf:.3e} V (robust, MAD)")
        print(f"  Untergrenze 1e-3 V entspricht dem {1e-3 / nf:.0f}-fachen; "
              f"Mediansignal dem {np.median(mp) / nf:.0f}-fachen")
        counts = {k: int((mp < k * nf).sum()) for k in (10, 20, 50, 100)}
        print("  Messungen unter k x Rauschboden: "
              + ", ".join(f"{k}x:{v}" for k, v in counts.items())
              + f"  (Vorgabe: {int((mp < 1e-3).sum())})")

    print("\nQualitaetsmasken auf der korrigierten Amplitude (CLEAN = True):")
    n_ch = P.amp_corr.sizes["channel"]
    keep_all = None
    for key in ("snr", "mean_amp", "sd_dist"):
        m = P.masks[key]
        # ein Kanal ueberlebt nur, wenn er in ALLEN uebrigen Dims (z.B. beide
        # Wellenlaengen) sauber ist -- dieselbe Logik wie in xrutils.apply_mask.
        keep = m.all(dim=[d for d in m.dims if d != "channel"])
        keep_all = keep if keep_all is None else (keep_all & keep)
        print(f"  {key:9s}: {int(keep.sum()):4d} / {n_ch} behalten "
              f"({n_ch - int(keep.sum())} verworfen)")
    print(f"  {'kombiniert':9s}: {int(keep_all.sum()):4d} / {n_ch} behalten "
          f"({n_ch - int(keep_all.sum())} verworfen)")

    print(f"\nErgebnis der Kette : {P.amp_raw.sizes['channel']} roh -> "
          f"{P.conc.sizes['channel']} verwertbar  "
          f"({len(P.dropped)} verworfen: {len(P.masks['nonpositive'])} nicht positiv, "
          f"{len(P.dropped) - len(P.masks['nonpositive'])} Qualitaet)")
    print(f"OD nach Pruning    : {dict(P.od.sizes)}")
    print(f"Konzentration      : {dict(P.conc.sizes)}")
