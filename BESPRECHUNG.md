# Was mit der Betreuung zu klären ist

> Stand: 2026-08-04. Grundlage: Gesprächsnotizen aus `anmerkungen/`, umgesetzt und geprüft in
> Tag 1–3 (siehe `UMSETZUNGSPLAN.md`). Ergebnisse und Abbildungen: `DOKUMENTATION.md`, Kap. 6.
>
> Sortiert nach Dringlichkeit. **A** = Entscheidung nötig, bevor es weitergeht.
> **B** = Befund, den sie kennen sollte, weil er ihrer Erwartung widerspricht.
> **C** = kleinere Rückfragen.

---

## A1 · Ihre Vorgabe „TDDR + Wavelet" kollidiert mit ihrer eigenen Filter-Regel

**Was sie sagte:** „motion_correction fehlt: tddr, wavelet" (Chat 1, 11:18) — und im Juli:
bei Drift-Modellierung nicht hochpassfiltern, sonst bestimmt der Filter statt des
Driftmodells die HRF-Schätzung.

**Was gemessen wurde:** TDDR **ist** faktisch ein Hochpass. Es dämpft das Driftband
(< 0,01 Hz) auf **55,6 %** und wirkt unterhalb 0,5 Hz wie ein breitbandiger Dämpfer. Wavelet
lässt das Driftband bei 100 % unangetastet.

Schlimmer: TDDR dämpft **die Hirnantwort selbst auf 70 %** ihrer Höhe. Das ist keine
Nebenwirkung, sondern trifft genau das Band, in dem eine Blockantwort liegt (~0,03 Hz).

**Warum es trotzdem gut aussieht:** Die 30 % Dämpfung heben die systemisch bedingte
HbO-Überschätzung zufällig auf — der RMSE halbiert sich. Bei **HbR**, wo es keine
Überschätzung zu kompensieren gibt, verschlechtert TDDR den Fehler von 8 % auf 24 %.

**Vorläufig entschieden:** Beides als Sweep-Achse mitgeführt statt stillschweigend eines
gewählt; Default in Demos ist `wavelet`. Abbildungen dazu: Abb. 10 und 11.

**→ Zu klären:** Ist das in ihrem Sinne? Konkret: soll TDDR in der finalen Empfehlung
auftauchen, und wenn ja — mit welcher Warnung? Meine Lesart ist, dass TDDR für eine Arbeit
*über Driftmodellierung* die Fragestellung untergräbt, weil es den Drift mitentfernt.

---

## A2 · Kanalraum gegen Bildraum — Aufwand klären

**Was sie sagte:** GLM im Kanalraum vs. Bildraum vergleichen, für beide Datensätze; um
Aktivität Hirnregionen zuzuordnen, muss man das inverse Problem lösen. Details „nächstes Mal".

**Stand:** Cedalion bringt alles mit (`cedalion.dot.ImageRecon`, Tutorial 5 — deckt
ausdrücklich auch dünn besetzte Montagen ab, was für 48 Kanäle bei 3 cm die relevante
Variante ist).

**Der Haken:** Für den Ruhedatensatz liegt die Sensitivitätsmatrix fertig vor
(`get_precomputed_sensitivity("nn22_resting", "colin27")`). Für die **NIRScout-Montage der
realen Daten gibt es keine** — die müsste über ein Kopfmodell berechnet werden. Das heißt
Optoden-Koregistrierung und eine Fluence-Simulation pro Optode (bei 32 Optoden GPU-Läufe).

**→ Zu klären:** Lohnt der Aufwand für die 48-Kanal-Montage, oder beschränkt sich der
Bildraum-Vergleich auf den augmentierten Datensatz? Und: hat sie eine fertige Adot oder eine
Koregistrierung für dieses Gerät?

---

## A3 · Die Short-Channel-Regression ist ein Simulationsbefund, kein reales Ergebnis

**Was sie sagte:** Short Channels als Regressor in der Designmatrix, um Scalp-Aktivität
herauszurechnen; 1,8 cm als Schwelle testen; Analyse nur über die langen Kanäle.

**Umgesetzt** — und es funktioniert: In der Simulation ist `short_avg` dem Global-Regressor
methodisch überlegen (Details in B1).

**Aber:** Der reale Datensatz hat **keine kurzen Kanäle**. Kürzester Abstand 25,9 mm, Median
32,1 mm. Auch keine Bewegungs-Aux. Auf realen Daten bleiben von fünf Konstellationen zwei
übrig: `baseline` und `global`.

**→ Zu klären:** Ist es akzeptabel, dass der Short-Channel-Teil ein reiner Simulationsbefund
bleibt? Alternativ — kennt sie einen zugänglichen Datensatz mit echten Short-Separation-
Kanälen? (Cedalion selbst hat welche: `get_multisubject_fingertapping_snirf_paths()` mit
5 Probanden, und der LUMO-Testdatensatz.)

---

## B1 · Der Global-Regressor ist schlechter als er aussieht

Rein nach Bias schlägt `global` die Short-Channel-Regression (+0,021 gegen +0,056 µM). Das
ist aber eine Verrechnung zweier Fehler: Der Global-Mittelwert mittelt über **alle** Kanäle,
also auch die aktivierten, und enthält dadurch **0,040 µM der eingemischten Hirnantwort**.
Die kurzen Kanäle enthalten konstruktionsbedingt 0,000 µM.

Die Bias-Differenz beträgt 0,035 µM, der Signalanteil des Global-Regressors 0,040 µM — er
„gewinnt", indem er einen Teil des Gesuchten wegrechnet. Dasselbe Muster wie das
Global-Mean-Artefakt aus Sweep v1, nur subtiler.

**→ Relevant für sie,** weil der Global-Regressor in der Praxis oft als unbedenklicher
Ersatz für Short Channels gilt.

---

## B2 · Tiefpass 0,5 Hz und AR-IRLS schließen einander aus

**Was sie sagte:** „testweise auch mal high (0,01) und lowpassfiltern (0,5) im sweep ganz am
ende im concentration space" (Chat 2, 11:45).

**Ergebnis:** Mit Tiefpass kollabiert die Schätzung auf **1e-05 µM** — fünf Größenordnungen
zu klein. Mit OLS liefert derselbe Datensatz normale Werte, ein reiner Hochpass läuft mit
AR-IRLS problemlos.

**Ursache:** AR-IRLS whitened über ein geschätztes Rauschmodell. Bei 3,906 Hz Abtastrate
entfernt ein Tiefpass bei 0,5 Hz drei Viertel des Spektrums (0,5 Hz = 0,256 × Nyquist);
oberhalb hat das Residuum keine Leistung mehr, der Whitening-Filter müsste dort unendlich
verstärken.

**Wichtig:** Das hängt an der Abtastrate. Auf dem 9-Hz-Ruhedatensatz läge 0,5 Hz bei
0,51 × Nyquist und wäre weit unkritischer. Der Befund ist also datensatzspezifisch, aber für
langsam abgetastete fNIRS-Daten allgemein relevant.

**→ Zu klären:** Soll der Tiefpass-Arm nur mit OLS ausgewertet werden, oder ganz entfallen?

---

## B3 · Den Drift zu modellieren schlägt ihn wegzufiltern — jetzt an realen Daten

Der Butterworth-Hochpass als Ersatz für Driftregressoren ist auf realen Daten die
**schlechteste** Option (Reproduzierbarkeit 0,443) — schlechter als *gar kein* Driftmodell
(0,494). Das stützt ihren Juli-Hinweis, jetzt empirisch statt nur als Argument.

Beste Familie in **beiden** Stufen der Arbeit: **DCT mit ~0,02 Hz**.

---

## B4 · Ihre Amplitudengrenzen sind unabhängig bestätigt — gelten aber nur für NinjaNIRS

**Was sie sagte:** „ninja_nirs: upper threshhold: 0,84, lower threshhold: 1e-3" (Chat 1).

Eine rein datengetriebene Lückensuche (ohne Kenntnis ihres Werts) findet auf dem
Ruhedatensatz **1,35e-03** — gleiche Größenordnung, gleiche Lücke. Ihr Wert ist damit
reproduziert statt nur übernommen.

**Auf den realen Daten greift die Grenze nicht:** Die Verteilung ist lückenlos (dunkelste
Messung 0,116 × Median, größte Lücke Faktor 1,11 über 6624 Messungen). Passend dazu steht in
der Datensatz-Dokumentation „Masked channels removed" — die Autoren haben das erledigt.

**→ Nur zur Kenntnis,** keine Entscheidung nötig.

---

## C1 · Abbildung 10: „über alle Bilder gleiche Skala"

Umgesetzt als gemeinsame y-Achse **je Chromophor-Zeile** über alle Fensterlängen — nicht
global über HbO und HbR, weil die sich um etwa das Fünffache unterscheiden und HbR sonst zur
flachen Linie würde. **→ War das so gemeint, oder wirklich eine einzige Skala über alles?**

## C2 · Relativer Scalp-Plot: Limit auf 100 %

Umgesetzt. Kanäle über 100 % werden gesättigt dargestellt, ihre Anzahl steht im Titel — sonst
verschwände stillschweigend Information, die die vorherige Perzentil-Skalierung sichtbar
machte. **→ Ist die Sättigungs-Anzeige erwünscht oder stört sie?**

## C3 · Der Bildraum-Code, den sie schicken wollte

Chat 2, 11:27: „ich kriege code für image reconstruction von elsa". Nie angekommen — wird
aber auch nicht mehr gebraucht, der Algorithmus steckt vollständig in `cedalion.dot`.
Gebraucht würde nur noch die Sensitivitätsmatrix für die NIRScout-Montage (siehe A2).

## C4 · Datensatz-Falle, die sie kennen sollte

Proband **S25** nutzt eine **andere Trigger-Kodierung** als alle anderen: dort ist `0` = Ruhe
und `1`–`5` sind die Finger; sonst ist `3` = Daumen. Wer die Labels naiv übernimmt, wertet bei
S25 den Mittelfinger als Daumen — und nichts schlägt fehl. Ist im Code explizit behandelt.
Dazu Sonderfälle bei S18, S20 und S02 (abweichende Ruhezeiten). Steht in der mitgelieferten
`Experimental notes.txt`, aber nicht im Paper.

---

## Nicht besprechungsbedürftig, nur der Vollständigkeit halber umgesetzt

- SNR-Schwelle 3 statt 10 (Chat 1, 11:08) ✓
- Dunkle/gesättigte Kanäle über `mean_amp` ✓ — inkl. Erklärung, warum das im Bildraum
  kritischer ist als im Kanalraum: dort geht die Kanalvarianz als Gewicht in die
  Pseudoinverse ein, und ein gesättigter Kanal ist durch das Klippen künstlich rauscharm,
  bekommt also maximales Gewicht und verteilt seinen Fehler über sein ganzes
  Sensitivitätsprofil.
- Reihenfolge OD → Bewegungskorrektur → Amplitude → Kanalbewertung → weiter auf OD ✓
- Baseline bei der OD-Umrechnung mitgeben ✓ (`int2od(..., return_baseline=True)`)
- `dark signal`-Aux ausgewertet ✓ (Rauschboden 9,55e-06 V)
- Cedalion aktualisiert ✓ (keine API-Änderung)
- „Normalisierungsfunktion im neuen Update?" ✓ — **gibt es nicht**; vermutlich war
  `return_baseline`/`od2int` gemeint
- Künstliche Aktivierung im Kanalraum statt Bildraum ✓ — und zusätzlich **vor** die
  Bewegungskorrektur gezogen, damit die Korrektur die Hirnantwort überhaupt erreichen kann
- OLS als zweites Rauschmodell ✓

---

# Anhang A · Was jeder Commit gemacht hat

In einfacher Sprache, chronologisch. Wer im Gespräch fragt „was habt ihr eigentlich
geändert?", findet die Antwort hier.

## Phase 1 – Die Vorverarbeitung neu bauen (ihre Anmerkungen aus Chat 1)

| Commit | Was passiert ist |
|---|---|
| `561fa9b` | **Neues Modul für die Vorverarbeitung angelegt.** Bisher steckte sie mitten im Hauptskript. Erster Baustein: die Umrechnung von gemessener Lichtstärke in „optische Dichte" – und dabei wird der Ausgangspegel jedes Kanals **mitgespeichert**. Ohne den käme man später nicht zurück. |
| `4f88996` | **Bewegungskorrektur eingebaut** (TDDR und Wavelet), und zwar auf der optischen Dichte, wie von ihr vorgegeben. Dazu eine Messfunktion, die prüft, ob ein Verfahren das langsame Driften mitentfernt. |
| `0f80d69` | **Rückweg zur Lichtstärke** (mit dem gespeicherten Ausgangspegel) und die drei Kanal-Prüfungen: Rauschabstand, Helligkeitsfenster (zu dunkel / übersteuert), Sender-Empfänger-Abstand. Erst auf der Lichtstärke sind „dunkel" und „übersteuert" überhaupt definiert. |
| `ee577a8` | **Schlechte Kanäle rauswerfen** – und danach wieder mit der optischen Dichte weiterarbeiten. Damit ist ihre Reihenfolge vollständig umgesetzt. |
| `260925b` | **Das Hauptskript benutzt jetzt die neue Kette.** Außerdem so gebaut, dass die Vorverarbeitung nur einmal gerechnet und dann durchgereicht wird – spart im großen Vergleichslauf 45 Minuten. |
| `331aa5b` | **Die Dunkelmessung ausgewertet.** Das Gerät misst nebenbei bei ausgeschaltetem Licht. Daraus ergibt sich ein Rauschboden, und ihre Helligkeitsgrenze von 0,001 V liegt beim 105-Fachen davon – also gut begründet statt nur übernommen. |
| `2b60e44` | **Cedalion aktualisiert** (ihr Punkt „cedalion neu pullen"). Drei Commits, keine Änderung an den benutzten Funktionen. |

## Phase 2 – Ein Konstruktionsfehler, den erst die Messung zeigte

| Commit | Was passiert ist |
|---|---|
| `c2ac5c3` | **Der wichtigste Umbau der ganzen Arbeit.** Bisher wurde die künstliche Hirnantwort *nach* der Bewegungskorrektur eingemischt – die Korrektur konnte sie also gar nicht beschädigen. Das ließ TDDR künstlich gut aussehen. Jetzt wird sie **vorher** eingemischt, so wie es bei echten Daten ist. Ergebnis: TDDR dämpft die Antwort auf 70 %. |
| `c2f0060` | **Zwei Tests, die das absichern.** Der eine prüft, dass die Ein- und Rückrechnung ohne Korrektur exakt ist. Der andere friert den TDDR-Befund ein: falls jemand später die Einmischung wieder verschiebt, schlägt der Test fehl, statt still falsche Zahlen zu liefern. |
| `d5022c7` | **Vergleichsskript**, das die Vorverarbeitungs-Varianten gegen die Rückgewinnung misst. Liefert die Zahlen für Punkt A1. |
| `dd542d0` | **Dokumentation** des TDDR-Befunds, samt der Warnung, den Fehler getrennt nach HbO und HbR anzusehen. |

## Phase 3 – Kurze Kanäle und weitere Vergleichsachsen (Chat 3 und 4)

| Commit | Was passiert ist |
|---|---|
| `a9a5e9c` | **Kurze Kanäle als Regressor**, in den drei von ihr genannten Varianten (Mittelwert, am besten korrelierender, nächstgelegener). Ausgewertet wird nur noch über die langen Kanäle. |
| `2ee854f` | **Keine künstliche Aktivität mehr in die kurzen Kanäle.** Sonst enthielte der Regressor die gesuchte Antwort und würde sie wegrechnen – gemessen hätten die kurzen Kanäle bis zu 0,54 von 0,60 µM abbekommen. |
| `1e1e28a` | **Bewegungskorrektur wird eine eigene Vergleichsachse** statt einer stillen Festlegung, plus die Konfiguration für den großen Lauf. |
| `0a6a546` | **Ihre Plot-Anmerkung umgesetzt:** relativer Kopf-Plot auf ±100 % begrenzt. |
| `4c0e265` | **Ihre zweite Plot-Anmerkung:** gemeinsame Skala über die Teilbilder, damit der Effekt der Fensterlänge sichtbar wird. Dazu eine neue Abbildung zur Bewegungskorrektur. |
| `a8dff96` | Kleine Reparatur: die Auswertung liest die Vergleichsgruppen jetzt aus den Daten, statt sie fest verdrahtet zu haben. |
| `390669e` | **Ergebnisse des großen Simulationslaufs** – 1800 Auswertungen, 7,5 Stunden. |

## Phase 4 – Die echten Daten

| Commit | Was passiert ist |
|---|---|
| `29e1a80` | **Erste Fassung des Daten-Einlesers**, geschrieben bevor die Dateien da waren. |
| `d5f3821` | **Zwei weitere Vergleichsachsen:** das einfache Schätzverfahren (OLS) neben dem aufwendigen, und der vollständige Filter-Arm mit Hoch-, Tief- und Bandpass. |
| `ad80db3` | **Plan umgestellt:** es gibt keine Tetris-Daten, Stufe 2 läuft auf dem Finger-Tapping-Datensatz. |
| `8e6ec47` | **Einleser fertig, mit den echten Dateien geprüft.** Enthält die Zuordnung der Trigger zu den Fingern – inklusive der Sonderfälle bei S25, S18, S20 und S02 (siehe C4). |
| `bb978fd` | **Die Hauptauswertung der echten Daten.** Zweistufig: erst je Aufnahme, dann über die 25 Probanden. Mit Korrektur für multiples Testen und dem Reproduzierbarkeits-Maß. |
| `e3141da` | Nachtrag: **Parallelisierung**. Das Standardverfahren nutzte nur 1,5 von 8 Rechenkernen. Über mehrere Prozesse verteilt wurde aus 15,8 Stunden 3,6. |
| `29f1a55` | Nachtrag: **die kaputte Kombination markieren.** Tiefpass und AR-IRLS vertragen sich nicht (siehe B2); die betroffenen Zellen werden jetzt ausgewiesen statt stillschweigend in Ranglisten zu landen. |
| `63fd9ec` | **Ergebnisse der echten Daten** – 3864 Auswertungen über 25 Probanden. |
| `4b61f29` | **Abbildungen zu den echten Daten** (siehe Anhang B). |
| `4ae53e9` | Die erzeugten Bilder und die Gruppenkarte. |
| `ccd92fa` | **Alle Ergebnisse in die Dokumentation eingearbeitet.** |

> *Hinweis: Die drei Commits `bb978fd`, `e3141da` und `29f1a55` tragen dieselbe Nachricht.
> Es ist dieselbe Datei in drei Ausbaustufen – die Nachricht wurde beim Nachbessern
> wiederverwendet. Inhaltlich siehe oben.*

---

# Anhang B · Was auf den Abbildungen zu den echten Daten zu sehen ist

## Abb. 15 – Reproduzierbarkeit (`15_real_reliability.png`)

**Die wichtigste Abbildung der zweiten Stufe.**

Bei echten Daten weiß man nicht, was „richtig" gewesen wäre – es gibt keine Wahrheit zum
Vergleichen. Wie soll man dann sagen, welches Driftmodell besser ist? Über die
**Wiederholbarkeit**: Jeder Proband hat die Aufgabe zwei- bis dreimal gemacht. Ein gutes
Modell muss beide Male ungefähr dasselbe Ergebnis liefern. Ein Modell, das zufälliges
Rauschen für Hirnaktivität hält, liefert jedes Mal etwas anderes.

Jeder Balken ist eine Driftregressor-Familie. **Höher = zuverlässiger.** Die volle Farbe ist
das aufwendige Schätzverfahren, die schraffierte das einfache.

Was man sieht: Die **orangen Balken (DCT)** sind am höchsten – diese Familie ist die
stabilste, und zwar bei beiden Verfahren. Die **roten Balken ganz rechts** sind die
Filter-Alternativen, und die sind am *niedrigsten* – sogar niedriger als der braune Balken
links, der für „gar kein Driftmodell" steht. Das ist das Kernergebnis: **den Drift zu
modellieren ist besser, als ihn wegzufiltern, und Wegfiltern ist schlechter als nichts tun.**

Das „n.a." markiert die Kombination, die rechnerisch nicht funktioniert (Punkt B2).

## Abb. 16 – Signifikante Kanäle (`16_real_significant.png`)

Hier geht es um die Frage: **Bei wie vielen der 48 Messkanäle ist die gefundene Aktivierung
statistisch abgesichert?** Weil 48 Kanäle gleichzeitig getestet werden, muss man dafür
korrigieren – sonst findet man allein durch Zufall etwas.

Blau = nur Driftmodell. Orange = zusätzlich ein Regressor für systemische Störungen (Puls,
Blutdruckwellen, Hautdurchblutung).

Was man sieht: Der orange Balken ist beim aufwendigen Verfahren fast überall **halb so hoch**
wie der blaue. Der systemische Regressor entfernt also die Hälfte der scheinbaren
Aktivierung. Das klingt schlecht, ist es aber vermutlich nicht: Ein Teil dessen, was ohne ihn
als „Hirnaktivität" gilt, ist in Wahrheit Körpersignal aus der Kopfhaut. Genau dasselbe
Muster zeigte die Simulation, wo man es nachprüfen konnte.

## Abb. 17 – Wo sitzt die Aktivierung? (`17_real_scalp.png`)

Die vier Bilder zeigen die 48 Messkanäle in ihrer räumlichen Anordnung auf dem Kopf. Oben
die gemessene Stärke, unten die statistische Sicherheit. Links HbO (sauerstoffreiches Blut),
rechts HbR (sauerstoffarmes).

Was man sieht: HbO ist überall **rot** (steigt an), HbR überall **blau** (fällt ab). Das ist
genau das erwartete physiologische Muster – wenn eine Hirnregion arbeitet, strömt
sauerstoffreiches Blut hinein und verdrängt sauerstoffarmes. Dass es so klar und über fast
alle Kanäle auftritt, spricht dafür, dass die Auswertung funktioniert.

**29 von 48 Kanälen** sind bei HbO abgesichert, 16 bei HbR.

**Wichtige Einschränkung – bitte im Gespräch erwähnen:** Der Datensatz enthält keine
Landmarken (Nasenwurzel, Ohrpunkte). Ohne die lässt sich das Kanal-Layout nicht am Kopf
ausrichten. Die Abbildung zeigt deshalb die **Struktur** der Aktivierung, aber **nicht, wo
links und rechts ist**. Die eigentlich naheliegende Kontrolle – beim Tappen mit der rechten
Hand müsste die linke Hirnhälfte stärker reagieren – lässt sich damit nicht führen.
Gemessen unterscheiden sich die beiden Kanal-Gruppen bei HbR signifikant (p = 0,04), bei HbO
nicht (p = 0,12); welche Gruppe welche Hirnhälfte ist, bleibt offen. **Das ist das stärkste
Argument für ihren Bildraum-Vorschlag** (Punkt A2).

## Abb. 18 – Plausibilität (`18_real_plausibility.png`)

Zwei Kontrollen, ob das Gemessene physiologisch sein *kann*. Jeder Punkt ist eine
Driftfamilie.

**Links:** das Verhältnis von HbR zu HbO. Physiologisch erwartet man etwa −0,4 (gestrichelte
Linie): HbR fällt um etwa 40 % dessen, was HbO steigt. Gemessen liegt es bei −0,18 ohne und
bis −0,36 mit systemischem Regressor. Der Regressor bringt die Werte also **näher an das
physiologisch Erwartete** – ein unabhängiges Argument dafür, dass er echte Störungen
entfernt und nicht Signal.

**Rechts:** die Gegenläufigkeit von HbO und HbR, gemessen als Korrelation. Sie liegt bei
**−0,88**, also sehr deutlich gegenläufig. Das ist ein starkes Zeichen dafür, dass ein echtes
hämodynamisches Signal vorliegt und nicht bloß Rauschen.
