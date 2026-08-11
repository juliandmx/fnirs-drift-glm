"""Analyse-Code zur Bachelorarbeit ueber Driftregressoren im fNIRS-GLM.

Vier Schichten, von unten nach oben:

    core      Bausteine, die alle Auswertungen teilen: Vorverarbeitungskette,
              Augmentations-Pipeline, Short-Channel-Regressoren, Bildraum.
    data      Einlesen der drei Datensaetze und die Koregistrierung.
    analysis  Die Auswertungen selbst -- jede schreibt eine Tabelle nach results/.
    reports   Abbildungen und Ergebnistabellen AUS den Ergebnissen der Auswertungen.

Die Abhaengigkeiten laufen nur nach unten: `reports` darf `analysis` benutzen, `analysis`
darf `data` und `core` benutzen, `core` kennt keine der oberen Schichten. Wer das umdreht,
baut einen Importzyklus -- und der faellt sofort auf.

Aufruf immer als Modul, nie als Datei:

    conda run -n cedalion python -m drift_glm.analysis.sweep v4
"""
