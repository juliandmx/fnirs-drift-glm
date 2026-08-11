"""Smoke-Tests fuer den Bildraum-Pfad und den Multisubject-Datensatz.

Bewusst auf der 28-Kanal-Montage des Multisubject-Datensatzes, wo es geht: dieselben
Tests auf nn22 wuerden dieselben Eigenschaften pruefen, aber je Test rund eine Minute
kosten (567 Kanaele x 25 000 Vertices).
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def ims():
    from drift_glm.core import imagespace
    return imagespace


@pytest.fixture(scope="module")
def head(ims):
    return ims.head()


@pytest.fixture(scope="module")
def adot_ms(ims):
    from drift_glm.data import multisubject as ms
    return ims.adot(ms.DATASET)


# ------------------------------------------------------------- Kopfmodell und Geometrie

def test_head_model_is_icbm152(ims, head):
    """Betreuungsvorgabe: durchgaengig ICBM152, nicht Colin27.

    Der Test fixiert nicht nur den Namen, sondern die Vertexzahl -- damit faellt auf,
    falls irgendwo doch das Colin27-Modell durchschlaegt (25 000 gegen 15 002 Vertices).
    """
    assert ims.HEAD_MODEL == "icbm152"
    assert head.brain.nvertices == 25000
    assert head.brain.crs == "mni"          # apply_transform(t_ijk2ras) ist passiert


def test_motor_seeds_are_lateralised(ims, head):
    """C3 und C4 liegen auf verschiedenen Hemisphaeren und weit auseinander."""
    xyz = ims.vertex_coords_mm(head)
    c3, c4 = ims.seed_vertex(head, "C3"), ims.seed_vertex(head, "C4")
    assert c3 != c4
    assert np.linalg.norm(xyz[c3] - xyz[c4]) > 60.0     # gegenueberliegende Hemisphaeren
    lpa = np.asarray(head.landmarks.sel(label="LPA").pint.dequantify().values, float)
    rpa = np.asarray(head.landmarks.sel(label="RPA").pint.dequantify().values, float)
    axis = (rpa - lpa) / np.linalg.norm(rpa - lpa)      # zeigt nach rechts
    mid = 0.5 * (lpa + rpa)
    assert (xyz[c3] - mid) @ axis < 0                   # C3 links
    assert (xyz[c4] - mid) @ axis > 0                   # C4 rechts


# ------------------------------------------------------------------- Vorwaertsmodell

def test_adot_brain_vertices_come_first(ims, adot_ms):
    """Die Blob-Vertices passen nur dann auf Adot, wenn die Hirnvertices vorn stehen.

    `build_spatial_activation` liefert einen Wert je HIRN-Vertex, ohne Koordinate. Waeren
    Hirn- und Kopfhautvertices in Adot gemischt, wuerde die Zuordnung stillschweigend
    verrutschen -- kein Fehler, nur falsche Ergebnisse.
    """
    ib = np.asarray(adot_ms.is_brain.values, bool)
    n = int(ib.sum())
    assert ib[:n].all() and not ib[n:].any()
    assert n == 25000


def test_forward_model_hits_the_expected_hemisphere(ims, head, adot_ms):
    """Der C3-Blob muss die linken, der C4-Blob die rechten Kanaele treffen.

    Auf dieser Montage sind die Quellen S1-S4 links und S5-S8 rechts. Das ist die
    scharfste Kontrolle des gesamten Vorwaertswegs: Kopfmodell, Blob, Adot und
    Kanalzuordnung muessen alle zueinander passen, sonst landet die Aktivierung auf der
    falschen Seite.
    """
    img = ims.spatial_activation(head, ("C3", "C4"))
    chan = ims.to_channel_space(ims.brain_adot(adot_ms), img)
    for tt, expected in (("Stim C3", "1234"), ("Stim C4", "5678")):
        v = np.abs(chan.sel(trial_type=tt).pint.dequantify()).max("wavelength")
        top = [str(c) for c in v.channel.values[np.argsort(v.values)[-5:]]]
        assert all(c[1] in expected for c in top), f"{tt}: {top}"


def test_ground_truth_is_calibrated_to_target_peak(ims, adot_ms):
    """Der Kanalraum-Peak muss exakt die Zielamplitude sein (Kalibrierung).

    Ohne diesen Schritt haengt die eingemischte Amplitude an der willkuerlichen
    Vertex-Intensitaet und ist nicht mehr in µM interpretierbar.
    """
    from drift_glm.data import multisubject as ms
    rec = ms.load(ms.paths()[0])
    chans = [str(c) for c in rec["amp"].channel.values]
    gt = ims.ground_truth(ms.DATASET, chans, rec.geo3d, target_uM=0.6)
    sees = np.asarray(gt["sees_cortex"], bool)
    hbo = np.abs(np.asarray(gt["beta_true_map"].sel(chromo="HbO").values, float))
    assert np.isclose(hbo[sees].max(), 0.6, rtol=1e-6)
    # HbR gegenlaeufig, Verhaeltnis nahe der Vorgabe -0.4
    j = int(np.argmax(np.where(sees, hbo, -np.inf)))
    hbr = float(gt["beta_true_map"].sel(chromo="HbR").isel(channel=j))
    assert -0.6 < hbr / 0.6 < -0.2
    # Die Wahrheit im Bildraum ist groesser als im Kanalraum: ein Kanal integriert ueber
    # sein ganzes Sensitivitaetsprofil, ein Vertex ist ein Punkt.
    assert float(np.abs(gt["img"].sel(chromo="HbO")).max()) > 0.6


def test_cortex_channel_mask_separates_short_from_long(ims):
    """Die Hirnsensitivitaet trennt kurze und lange Kanaele um Groessenordnungen.

    Das ist die Grundlage dafuer, dass die Kalibrierung nicht an einem kurzen Kanal
    haengt. Die Trennung MUSS ueber die Sensitivitaet laufen, nicht ueber den Abstand:
    letzterer ist montageabhaengig, erstere ist die physikalische Groesse.
    """
    import cedalion.nirs
    from drift_glm.data import multisubject as ms
    rec = ms.load(ms.paths()[0])
    chans = [str(c) for c in rec["amp"].channel.values]
    Ab = ims.brain_adot(ims.adot(ms.DATASET), channels=chans)
    sees = ims.cortex_channels(Ab)
    d = np.asarray(cedalion.nirs.channel_distances(rec["amp"], rec.geo3d)
                   .pint.to("mm").pint.dequantify().values, float)
    is_short = d < 15.0
    assert is_short.sum() == 8
    # Die Maske muss genau die langen Kanaele behalten.
    assert not sees[is_short].any()
    assert sees[~is_short].all()


def test_calibration_is_anchored_to_a_cortex_channel(ims):
    """Die Zielamplitude muss auf einem LANGEN Kanal liegen, nicht auf einem kurzen.

    Der Fallstrick, den dieser Test einfriert: `od2conc` teilt durch Kanalabstand x DPF.
    Bei 7,3 mm gegen 37 mm ist dieser Nenner fuenfmal kleiner, dieselbe OD wird also zu
    einer fuenfmal groesseren "Konzentration". Ohne die Einschraenkung auf Kanaele mit
    Hirnsensitivitaet lag das globale Maximum der Kanalraum-Wahrheit auf einem
    7,3-mm-Kanal -- die Kalibrierung waere an einem Artefakt verankert gewesen und alle
    langen Kanaele haetten nur einen Bruchteil der Zielamplitude bekommen.
    """
    import cedalion.nirs
    from drift_glm.data import multisubject as ms
    rec = ms.load(ms.paths()[0])
    chans = [str(c) for c in rec["amp"].channel.values]
    gt = ims.ground_truth(ms.DATASET, chans, rec.geo3d, target_uM=0.6)
    d = np.asarray(cedalion.nirs.channel_distances(rec["amp"], rec.geo3d)
                   .pint.to("mm").pint.dequantify().values, float)
    v = np.abs(np.asarray(gt["beta_true_map"].sel(chromo="HbO").values, float))
    sees = np.asarray(gt["sees_cortex"], bool)

    # Der kalibrierte Peak liegt auf einem langen Kanal und ist exakt die Zielamplitude.
    assert np.isclose(v[sees].max(), 0.6, rtol=1e-6)
    assert d[np.argmax(np.where(sees, v, -np.inf))] > 15.0


# -------------------------------------------------------------- Sichtbarkeit / Rueckweg

def test_sensitivity_mask_keeps_the_seeds_but_not_the_deep_brain(ims, head, adot_ms):
    """Die Maske muss die Blob-Zentren enthalten und den Grossteil ausschliessen.

    Sie ist der Grund, warum die Bildraum-Kennzahlen ueberhaupt aussagekraeftig sind:
    ohne sie wird das Verhalten der Tiefenkorrektur gemessen statt der Schaetzung.
    """
    mask = ims.sensitivity_mask(adot_ms)
    assert mask.dtype == bool and mask.size == 25000
    assert mask[ims.seed_vertex(head, "C3")]
    assert mask[ims.seed_vertex(head, "C4")]
    # Eine 28-Kanal-Montage ueber dem Motorkortex sieht nur einen kleinen Teil des Kortex.
    assert 0.01 < mask.mean() < 0.5


def test_conc_map_survives_the_od_roundtrip():
    """Eine beta-Karte muss den Weg Konzentration -> OD -> Konzentration exakt ueberleben.

    Nur dann ist eine im Bildraum gemessene Abweichung ein Effekt der Methode und kein
    Artefakt der Umrechnung -- dieselbe Argumentation wie beim Zeitreihen-Rundweg in
    `test_smoke.test_od_roundtrip_is_exact`, hier fuer die Karten ohne Zeitachse.
    """
    from drift_glm.analysis import imageglm as ig
    from drift_glm.data import multisubject as ms
    from drift_glm.core import preprocess as prep
    import cedalion.nirs
    import xarray as xr

    rec = ms.load(ms.paths()[0])
    P, _ = ms.preprocess_recording(rec, motion_method="none")
    m = P.conc.isel(time=0, drop=True)                 # irgendeine Karte (channel, chromo)
    od = ig.conc_map_to_od(m, P.geo3d, P.od.wavelength, like=P.conc)

    dpf = xr.DataArray([prep.DEFAULT_DPF] * od.sizes["wavelength"], dims="wavelength",
                       coords={"wavelength": od.wavelength})
    ts = od.expand_dims("time").assign_coords(time=[0.0])
    ts.time.attrs["units"] = "second"
    back = cedalion.nirs.cw.od2conc(ts, P.geo3d, dpf, spectrum="prahl")
    back = back.isel(time=0, drop=True).pint.to("uM").pint.dequantify()

    a = np.asarray(m.transpose("channel", "chromo").values, float)
    b = np.asarray(back.transpose("channel", "chromo").values, float)
    assert np.nanmax(np.abs(a - b)) / np.nanmax(np.abs(a)) < 1e-9


# --------------------------------------------------------------- Multisubject-Datensatz

def test_multisubject_has_eight_real_short_channels():
    """Der Grund, warum dieser Datensatz dazugekommen ist."""
    import cedalion.nirs
    from drift_glm.data import multisubject as ms
    rec = ms.load(ms.paths()[0])
    d = np.asarray(cedalion.nirs.channel_distances(rec["amp"], rec.geo3d)
                   .pint.to("mm").pint.dequantify().values, float)
    short = d[d < 15.0]
    assert short.size == 8
    assert short.max() < 10.0            # echte Short Separation, nicht "kurze Distanz"
    assert d[d >= 15.0].min() > 30.0     # klare Luecke zur langen Gruppe


def test_multisubject_events_are_renamed_and_sentinel_dropped():
    """Ohne die Umbenennung waeren '2.0'/'3.0' stumme Fallen wie S25 im Khan-Datensatz."""
    from drift_glm.data import multisubject as ms
    rec = ms.load(ms.paths()[0])
    tt = set(map(str, rec.stim.trial_type))
    assert tt == set(ms.CONDITIONS)
    assert "sentinel" not in tt
    counts = rec.stim.trial_type.value_counts()
    assert all(counts[c] == 30 for c in ms.CONDITIONS)


def test_global_component_subtraction_reduces_systemic_share():
    """Der Abzug muss den Anteil des Short-Regressors tatsaechlich verkleinern.

    Geprueft an der Korrelation der langen Kanaele mit dem Short-Mittelwert: sie muss
    nach dem Abzug deutlich kleiner sein. Damit ist belegt, dass die Variante aus
    Notebook 50b greift -- und der Vergleich mit der Designmatrix-Variante (msglm.py) ist
    kein Vergleich zwischen "wirkt" und "wirkt nicht".
    """
    import cedalion.models.glm as glm
    import cedalion.nirs
    from drift_glm.data import multisubject as ms
    from drift_glm.core import shortchannel as sc
    from cedalion import units

    rec = ms.load(ms.paths()[0])
    P, _ = ms.preprocess_recording(rec, motion_method="none")
    stim = ms.stim_df(rec, "tapping")
    long, short = cedalion.nirs.split_long_short_channels(
        P.conc, P.geo3d, distance_threshold=ms.SHORT_THRESHOLD)
    basis = glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=0 * units.s)
    corrected = sc.subtract_global_component(long, short, stim, basis,
                                             variant="short_avg", geo3d=P.geo3d)

    ref = np.asarray(short.sel(chromo="HbO").mean("channel").values, float)

    def mean_abs_corr(ts):
        a = np.asarray(ts.sel(chromo="HbO").transpose("channel", "time").values, float)
        return float(np.nanmean([abs(np.corrcoef(ref, row)[0, 1]) for row in a]))

    before, after = mean_abs_corr(long), mean_abs_corr(corrected)
    assert after < 0.6 * before, f"vorher {before:.3f}, nachher {after:.3f}"
