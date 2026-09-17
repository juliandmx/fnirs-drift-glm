"""Analyse-Code zur Bachelorarbeit ueber Driftregressoren im fNIRS-GLM.

    core      Bausteine fuer alle Auswertungen: Vorverarbeitung, Augmentations-Pipeline,
              Short-Channel-Regressoren, Bildraum.
    data      Einlesen der drei Datensaetze und Koregistrierung.
    analysis  Die Auswertungen; jede schreibt eine Tabelle nach results/.
    reports   Abbildungen und Ergebnistabellen aus den Auswertungen.

Abhaengigkeiten laufen nur nach unten (reports -> analysis -> data/core). Aufruf immer
als Modul:  conda run -n cedalion python -m drift_glm.analysis.sweep v4
"""
