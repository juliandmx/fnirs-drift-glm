# Projektdokumentation

**Bachelorarbeit: „Vergleich verschiedener Driftregressoren im General Linear Model zur Schätzung der hämodynamischen Antwortfunktion in fNIRS-Daten"**
TU Berlin · Fachgebiet Neurotechnologie · Stand: 13. Juli 2026

---

## Für wen ist dieses Dokument?

Dieses Dokument erklärt **von Grund auf**, worum es in diesem Projekt geht, wie die
Software funktioniert und was bisher herausgekommen ist. Es ist bewusst so geschrieben,
dass man **keine Informatik- und keine Neurowissenschafts-Kenntnisse** braucht, um zu
verstehen, was hier passiert. Jeder Fachbegriff wird beim ersten Auftauchen erklärt; am
Ende gibt es ein Glossar zum Nachschlagen.

Inhalt:
1. [Das Projekt in einfachen Worten](#1-das-projekt-in-einfachen-worten)
2. [Die wichtigsten Begriffe – mit Alltagsvergleichen](#2-die-wichtigsten-begriffe--mit-alltagsvergleichen)
3. [Das Werkzeug „Cedalion" und seine Notebooks](#3-das-werkzeug-cedalion-und-seine-notebooks)
4. [Wie die Auswertung Schritt für Schritt funktioniert](#4-wie-die-auswertung-schritt-für-schritt-funktioniert)
5. [Der systematische Vergleich (der „Sweep")](#5-der-systematische-vergleich-der-sweep)
6. [Was bisher herausgekommen ist](#6-was-bisher-herausgekommen-ist)
7. [Die Dateien im Überblick](#7-die-dateien-im-überblick)
8. [Wie man alles ausführt](#8-wie-man-alles-ausführt)
9. [Was noch kommt (Ausblick)](#9-was-noch-kommt-ausblick)
10. [Glossar](#10-glossar)

---

## 1. Das Projekt in einfachen Worten

Wenn ein Bereich im Gehirn aktiv wird, verbraucht er mehr Sauerstoff, und der Körper
schickt kurz darauf **mehr Blut** dorthin. Diese Durchblutungsänderung kann man **von
außen durch den Schädel messen** – mit Licht. Genau das macht **fNIRS** (funktionelle
Nahinfrarotspektroskopie): schwaches Infrarotlicht wird auf den Kopf gestrahlt, ein Teil
davon kommt an anderer Stelle wieder heraus, und aus der Abschwächung lässt sich
berechnen, wie viel sauerstoffreiches und sauerstoffarmes Blut gerade unter der Haut
fließt.

Das Problem: Das gemessene Signal enthält **nicht nur** die Hirnaktivität, die uns
interessiert, sondern auch viel „Störung": Herzschlag, Atmung, langsames Wegdriften der
Messwerte, kleine Bewegungen. Besonders lästig ist das langsame **Driften** der Grundlinie
(englisch *drift*) – so, als würde ein Radiosender langsam wegwandern. Wenn man dieses
Driften nicht sauber herausrechnet, misst man die Hirnaktivität falsch.

Um das Driften herauszurechnen, baut man ins Auswertemodell (das **General Linear Model**,
siehe Kapitel 2) sogenannte **Driftregressoren** ein – mathematische „Formvorlagen" für das
langsame Driften. Davon gibt es verschiedene Sorten (Polynome, Cosinus-Funktionen,
Legendre-Polynome, B-Splines; dazu als Alternative einen Filter). **Die zentrale Frage der
Arbeit lautet:**

> **Welche Sorte von Driftregressor liefert unter welchen Bedingungen die genaueste
> Schätzung der Hirnaktivität?**

Für fMRT (die „große" Bildgebung im Kernspintomografen) ist das gut untersucht – für fNIRS
erstaunlicherweise kaum. Diese Lücke füllt die Arbeit.

**Vorgehen in zwei Stufen:**
1. **Simulation (aktueller Stand):** Wir nehmen echte Ruhemessungen (Person tut nichts) und
   „mischen" künstlich eine Hirnaktivität mit **exakt bekannter Stärke** hinein. Weil wir die
   Wahrheit kennen, können wir für jede Driftregressor-Sorte genau messen, wie nah die
   Schätzung an der Wahrheit liegt.
2. **Echte Daten (später):** Anwendung auf echte hochauflösende Messungen (300 Kanäle), bei
   denen Probanden *Tetris* spielen gegen Ruhe.

---

## 2. Die wichtigsten Begriffe – mit Alltagsvergleichen

**fNIRS** – Messung der Hirndurchblutung mit Infrarotlicht durch den Schädel. Nicht-invasiv,
tragbar, günstig.

**HbO und HbR** – die zwei Blut-Bestandteile, die fNIRS unterscheidet:
- **HbO** = oxygeniertes (sauerstoffreiches) Hämoglobin.
- **HbR** = deoxygeniertes (sauerstoffarmes) Hämoglobin.
Bei echter Hirnaktivität steigt **HbO** und sinkt **HbR** gleichzeitig (eine **inverse**
Beziehung). Findet man das im Signal, ist das ein gutes Plausibilitätszeichen.

**Kanal** – ein Messpunkt, gebildet aus einem Lichtsender und einem Lichtempfänger auf dem
Kopf. Viele Kanäle zusammen ergeben eine Karte der Hirnoberfläche. Unser Datensatz hat
**544 nutzbare Kanäle** – das ist „hochkanalig" und heißt auch **DOT** (Diffuse Optische
Tomografie).

**HRF (hämodynamische Antwortfunktion)** – die *typische Form* der Durchblutungsantwort auf
einen kurzen Reiz: Sie steigt über einige Sekunden an, erreicht nach ca. 5–6 s ein Maximum
und klingt dann wieder ab. Vergleich: Wirft man einen Stein ins Wasser, entsteht immer die
gleiche charakteristische Wellenform – die HRF ist diese „Standardwelle" der Durchblutung.

**Aktivierungsstärke β (beta)** – *wie hoch* die HRF-Welle an einem Kanal ausschlägt. Das
ist die eigentliche Zielgröße: β groß = starke Aktivierung. Die ganze Arbeit dreht sich
darum, dieses β möglichst genau zu **schätzen**.

**Drift** – langsames Wegwandern der Messwerte über die Zeit, unabhängig von jeder
Hirnaktivität (z. B. durch Erwärmung der Geräte, langsame Körpervorgänge). Der Störenfried,
den wir modellieren müssen.

**GLM (General Linear Model)** – das Auswertemodell. Idee: Das gemessene Signal wird als
**Summe mehrerer bekannter „Zutaten"** erklärt. Vergleich mit einem Rezept:

> gemessenes Signal ≈ β · (HRF-Zutat) + (Drift-Zutaten) + Rest.

Das GLM findet die Mischungsverhältnisse (u. a. das β), mit denen die Summe der Zutaten das
gemessene Signal am besten nachbildet. Die Tabelle aller Zutaten heißt **Designmatrix**.

**Driftregressor** – eine der „Drift-Zutaten" im Rezept. Es gibt mehrere Familien, die das
Driften unterschiedlich flexibel nachbilden:
- **Polynom (poly n):** einfache Kurven wachsender Krümmung (Gerade, Parabel, …). `n` = wie
  „gewellt" die Kurve sein darf.
- **DCT (diskrete Cosinus-Basis):** eine Sammlung von Cosinus-Wellen bis zu einer
  Grenzfrequenz – wirkt wie ein Hochpassfilter, lässt also nur „schnelle genug" Signale durch.
- **Legendre-Polynome:** mathematisch besonders „saubere" Polynome (gut trennbar).
- **B-Splines:** aus glatten Kurvenstücken zusammengesetzte, flexible Basis.
- **Butterworth-Hochpass:** *kein* Regressor im Rezept, sondern ein **Filter**, der das
  langsame Driften schon vor der Auswertung aus dem Signal entfernt – die *Alternative* zu
  Driftregressoren.
- **„none":** gar kein Drift-Modell (nur ein konstanter Grundwert) – als Vergleichs-Untergrenze.

**AR-IRLS** – das Schätzverfahren, mit dem das GLM gelöst wird. Es berücksichtigt, dass das
Rauschen in fNIRS nicht rein zufällig ist, sondern zeitlich „nachhallt" (autokorreliert).
**Wichtig:** AR-IRLS modelliert das *Rauschen*, nicht den *Drift* – der Drift kommt
ausschließlich über die Regressoren ins Modell.

**Ground Truth (Grundwahrheit)** – der bekannte, wahre Wert. In der Simulation kennen wir
β, weil wir es selbst hineingemischt haben. So können wir Schätzfehler exakt ausrechnen.

**Gütemaße** – womit wir „genau" messen (getrennt für HbO und HbR):
- **Bias (Verzerrung):** systematisches Daneben­liegen (im Schnitt zu hoch/zu niedrig).
- **Varianz / Streuung:** wie stark die Schätzung von Durchlauf zu Durchlauf schwankt.
- **RMSE (Wurzel des mittleren quadratischen Fehlers):** Gesamtfehler, fasst Bias und Streuung
  zusammen. Kleiner = besser.
Vergleich Dartscheibe: **Bias** = die Pfeile treffen konstant neben die Mitte; **Varianz** =
die Pfeile streuen weit; **RMSE** = wie schlecht insgesamt.

---

## 3. Das Werkzeug „Cedalion" und seine Notebooks

**Cedalion** ist ein frei verfügbares Python-Programmpaket zur fNIRS-/DOT-Auswertung,
mitentwickelt an der TU Berlin. Es liefert fertige, geprüfte Bausteine für alle Schritte
(Daten laden, Qualität prüfen, in Konzentrationen umrechnen, GLM rechnen, Aktivierung
simulieren, Karten zeichnen). Wir bauen darauf auf, statt alles selbst zu programmieren.

> **Grundsatz dieses Projekts:** Die offiziellen **Cedalion-Notebooks** (Schritt-für-Schritt-
> Beispiele) und die Dokumentation sind die **maßgebliche Referenz** – vor eigenem Wissen.
> Ein „Notebook" ist ein interaktives Dokument, das Erklärtext und lauffähigen Beispielcode
> mischt.

- **Online-Dokumentation:** https://doc.ibs.tu-berlin.de/cedalion/doc/dev/
- **Notebooks lokal:** im Schwester-Ordner `../cedalion/examples/`
- **Cedalion-Version in diesem Projekt:** 26.5.1 (Entwicklungszweig `dev`)

**Die für dieses Projekt wichtigsten Notebooks:**

| Notebook | Wofür |
|---|---|
| `examples/getting_started_io/13_data_structures_intro.ipynb` | Wie fNIRS-Daten in Cedalion aufgebaut sind |
| `examples/signal_quality/21_data_quality_and_pruning.ipynb` | Kanäle nach Signalqualität aussortieren |
| `examples/modeling/33_glm_illustrative_example.ipynb` | GLM verständlich erklärt |
| `examples/modeling/32_glm_fingertapping_example.ipynb` | GLM an echten Daten + **Kopf-Karten (scalp plot)** der β-Werte |
| `examples/modeling/31_glm_basis_functions.ipynb` | die HRF-„Formvorlagen" (Basisfunktionen) |
| `examples/modeling/35_statsmodels_overview.ipynb` | wie man die Ergebnisse (β, Statistik) ausliest |
| `examples/augmentation/62_synthetic_hrfs_example.ipynb` | **künstliche HRF einmischen** (Kernidee der Simulation) |
| `examples/tutorial/7_data_augmentation.ipynb` | Tutorial zur Datenaugmentation inkl. Kopf-Karten |

---

## 4. Wie die Auswertung Schritt für Schritt funktioniert

Das Herzstück ist die Datei `pipeline.py`. Sie baut aus Ruhedaten einen Testfall mit
**bekannter Wahrheit**. Hier die Schritte in Alltagssprache; in Klammern die
Cedalion-Bausteine und die Notebooks, an denen sie sich orientieren.

### Schritt 1 – Ruhedaten laden
Wir laden einen frei verfügbaren fNIRS-Ruhedatensatz („nn22"): eine Person, die ~6 Minuten
still sitzt. Er enthält **567 Kanäle**, misst mit ~**9 Messungen pro Sekunde**, bei zwei
Lichtfarben (760 und 850 Nanometer).
*(Cedalion: `cedalion.data.get_nn22_resting_state()`. Aufbau der Daten: Notebook
`getting_started_io/13`.)*

### Schritt 2 – Aufräumen: Bewegung herausrechnen, schlechte Kanäle aussortieren
Nicht jeder Kanal misst sauber, und die Person bewegt sich. Beides wird hier behandelt – in
einer **bestimmten Reihenfolge**, die nicht beliebig ist:

1. **Lichtstärke → optische Dichte.** Dabei wird der Ausgangspegel jedes Kanals (die
   „Baseline", also seine mittlere Helligkeit) **mitgespeichert**. Ohne ihn käme man später
   nicht zurück, denn die optische Dichte beschreibt nur *Änderungen* gegenüber dem eigenen
   Mittel, nicht die Helligkeit selbst.
2. **Bewegungsartefakte herausrechnen.** Sie treten in zwei Formen auf: als **scharfer
   Ausschlag** (Spike) und als **ruckartige Verschiebung** der Grundlinie. Dafür gibt es zwei
   Verfahren – *Wavelet* gegen Spikes, *TDDR* gegen Verschiebungen. Sie arbeiten auf der
   optischen Dichte, nicht auf der Helligkeit.
3. **Zurück zur Lichtstärke.** Erst hier lässt sich beurteilen, ob ein Kanal **zu dunkel**
   (kaum Licht, nur Rauschen) oder **gesättigt** ist (Detektor am Anschlag, Signal oben
   abgeschnitten). Beides sind Helligkeits-Begriffe – in der optischen Dichte sind sie gar
   nicht definiert. Deshalb der Umweg.
4. **Kanäle bewerten und aussortieren**, nach drei Kriterien: Signal-Rausch-Verhältnis
   (Schwelle 3), Helligkeitsfenster (0,001 bis 0,84 Volt), Sender-Empfänger-Abstand.
5. **Weiterarbeiten auf der optischen Dichte** und erst dann in Konzentrationen umrechnen.

Von **567** Rohkanälen bleiben **520** übrig: 6 fallen weg, weil ihre Lichtstärke stellenweise
auf null geht (physikalisch unmöglich, unter dem Rauschboden), 41 durch die Qualitätskriterien.

Wie gut ist die Helligkeits-Untergrenze gewählt? Der Datensatz enthält eine **Dunkelmessung**
(das Gerät misst bei ausgeschaltetem Licht mit) – daraus ergibt sich ein Rauschboden von
0,0000095 Volt. Die Grenze 0,001 Volt liegt also beim **105-fachen** davon. Und sie liegt in
einer Lücke: zwischen dem 50- und dem 105-fachen des Rauschbodens fällt **kein einziger**
weiterer Kanal heraus. Die beiden Gruppen – brauchbare und tote Kanäle – sind also klar
getrennt, die genaue Lage der Grenze ist unkritisch.

> **Ein Befund, der für diese Arbeit zentral ist:** Die beiden Bewegungs-Verfahren verhalten
> sich völlig unterschiedlich gegenüber dem **langsamen Driften**, um das es in dieser Arbeit
> geht. *Wavelet* lässt es unangetastet (100 % bleiben übrig). *TDDR* dagegen entfernt
> **fast die Hälfte davon** (nur 55,6 % bleiben) – es wirkt unterhalb von 0,5 Hz wie ein
> breiter Dämpfer. Damit würde ein Teil des Driftens schon vor der Auswertung verschwinden,
> und die Driftregressoren hätten weniger zu tun, als sie eigentlich sollten. Weil das die
> Kernfrage der Arbeit berührt, wird die Wahl des Verfahrens **nicht** festgelegt, sondern
> als eigene Vergleichsachse mitgeführt (Kapitel 5).
>
> **Und es trifft nicht nur das Driften, sondern die Hirnantwort selbst.** Die eingemischte
> HRF liegt im Bereich um 0,03 Hz – genau dort, wo TDDR dämpft. Gemessen kommt von der
> eingemischten Höhe nur noch **70 %** an (HbO wie HbR); bei *Wavelet* sind es 100 %.
> Damit unterschätzt TDDR die gesuchte Größe systematisch um rund 30 %.
>
> Warum das trotzdem zunächst wie eine Verbesserung *aussieht*: Ohne Bewegungskorrektur wird
> HbO ohnehin um **50 % überschätzt** (Schätzung 0,596 statt 0,397 – die Driftregressoren
> ziehen systemische Störungen in die Hirnantwort hinein). Die 30 % Dämpfung von TDDR heben
> einen Teil dieser Überschätzung zufällig wieder auf, der Fehler sinkt auf +0,059. Das ist
> aber **kein besseres Schätzen, sondern das Verrechnen zweier Fehler**. Sichtbar wird das an
> **HbR**: dort gibt es keine Überschätzung, gegen die sich etwas verrechnen könnte – und
> prompt verschlechtert TDDR den Fehler von +0,018 auf +0,071. Dieselbe Dämpfung, aber ohne
> den Gegenfehler.
>
> *(Zahlen aus `compare_preprocessing.py`, 3 Wiederholungen, Fenster 180 s, 20 aktivste
> Kanäle → `results/preprocessing_comparison.csv`. Der Vergleich läuft auf einer gemeinsamen
> Kanalbasis; deshalb sind „alte" und „neue" Kette ohne Bewegungskorrektur identisch – die
> Umstellung ändert nicht die Schätzung auf einem Kanal, sondern welche Kanäle eingehen.)*
>
> **Wichtig für die Aussagekraft:** Damit das überhaupt messbar ist, wird die künstliche
> Hirnantwort seit dem Umbau **vor** der Bewegungskorrektur eingemischt (in die optische
> Dichte), nicht danach. Sonst könnte die Korrektur die Antwort gar nicht erreichen – sie
> dürfte nur das Rauschen putzen und sähe künstlich gut aus. Auf echten Daten steckt die
> Hirnantwort ebenfalls im Signal, wenn korrigiert wird.

*(Cedalion: `cedalion.nirs.cw.int2od(..., return_baseline=True)` / `od2int(...)`,
`cedalion.sigproc.motion.tddr(...)` / `wavelet(...)`, `cedalion.sigproc.quality.snr / mean_amp /
sd_dist / prune_ch`. Notebooks `signal_quality/21`, `signal_quality/22`, Tutorial `3`.
Umgesetzt in `preprocess.py`; die Reihenfolge entspricht der Betreuungsvorgabe und zugleich
der von Cedalion empfohlenen Kette.)*

### Schritt 3 – Licht in Blutkonzentrationen umrechnen
Aus der optischen Dichte werden die Konzentrationsänderungen von **HbO** und **HbR** je Kanal
(in Mikromol, µM) – über das **modifizierte Beer-Lambert-Gesetz**. Das ist eine feste
physikalische Umrechnung ohne freie Entscheidungen.
*(Cedalion: `cedalion.nirs.cw.od2conc(...)`.)*

### Schritt 4 – Eine künstliche Hirnaktivität mit bekannter Stärke einmischen
Das ist der Trick, mit dem wir eine **Grundwahrheit** erzeugen. Wir nehmen die
Standard-HRF-Welle (eine **Gamma-Form**), skalieren sie auf **Höhe 1** (normieren) und
mischen sie mit einer selbst gewählten Stärke β in die Ruhedaten. Weil die Welle auf Höhe 1
normiert ist, **ist die eingemischte Stärke danach exakt gleich dem β**, das die Auswertung
zurückliefern soll. So können wir „Geschätzt vs. Wahrheit" direkt vergleichen.

Wichtige Feinheit (**räumlicher Blob**, seit Version 2): Eine echte Hirnaktivierung sitzt
**lokal** an einer Stelle, nicht überall gleich. Deshalb mischen wir die Aktivität als
**räumlichen Fleck** ein: Kanäle nahe dem Zentrum bekommen die volle Stärke (0,6 µM für
HbO), weiter entfernte Kanäle immer weniger (glockenförmiger Abfall, „Gauß-Blob", Breite
σ = 30 mm). Konkret sind ~24 Kanäle stark und ~95 überhaupt merklich aktiviert. HbR bekommt
dieselbe Form, aber mit umgekehrtem Vorzeichen und 40 % der Höhe (die inverse HbO/HbR-Beziehung).

*(Cedalion-Bezug: Idee und Bausteine aus Notebook `augmentation/62` und Tutorial `7`. Dort
wird der Fleck zuerst auf der Hirnoberfläche erzeugt und dann über ein Kopfmodell auf die
Kanäle projiziert. Wir definieren ihn stattdessen **direkt im Kanal-Raum** über die
Kanalpositionen – das entspricht der Betreuungsvorgabe („künstliche Aktivierung nicht im
Bildraum einfügen, sondern im Kanalraum, um realistischer zu sein"): eingemischt wird genau
dort, wo Ruhedaten und Aktivierung zusammengeführt werden, ohne den Umweg über ein
Vorwärtsmodell und dessen eigene Näherungen. Funktion `_spatial_beta` in `pipeline.py`.
Ein Kopfmodell **läge durchaus vor** – `cedalion.data.get_precomputed_sensitivity(
"nn22_resting", "colin27")` liefert die fertige Sensitivitätsmatrix für genau diesen
Datensatz –, es wird für die Einmischung aber bewusst nicht verwendet. Für den Vergleich
**im Bildraum** (Kapitel 9) kommt es dann zum Einsatz.)*

### Schritt 5 – Die Designmatrix (das „Rezept") zusammenstellen
Jetzt legen wir die „Zutaten" fest, mit denen das GLM das Signal erklären soll:
- die **HRF-Zutat** (dieselbe normierte Welle, die wir eingemischt haben – so ist der
  Vergleich fair),
- die **Drift-Zutaten** (je nach untersuchter Familie: Polynome, DCT, Legendre, B-Splines …),
- optional weitere Zutaten (siehe „Konstellationen" in Kapitel 5).
*(Cedalion: `hrf_regressors(...)`, `drift_regressors(...)`, `drift_cosine_regressors(...)`,
`drift_legendre_regressors(...)`; HRF-Formen aus Notebook `modeling/31`, Zusammenbau aus
`modeling/33`.)*

![Designmatrix](figures/01_designmatrix.png)
*Abb. 1 – Die Designmatrix (das „Rezept"). Oben die HRF-Zutat: die auf Höhe 1 normierte
Standardwelle, an jedem Stimulus wiederholt. Unten die Polynom-Drift-Zutaten, mit denen das
langsame Wegwandern der Grundlinie nachgebildet wird.*

### Schritt 6 – Das GLM lösen (schätzen)
Das GLM bestimmt die Mischungsverhältnisse aller Zutaten – insbesondere das geschätzte **β̂**
(sprich „beta-Dach") je Kanal. Wir nutzen das robuste Verfahren **AR-IRLS**.
*(Cedalion: `cedalion.models.glm.fit(..., noise_model="ar_irls")`, Ergebnis-Auslesen mit
`.sm.params`; Notebooks `modeling/32` und `35`.)*

### Schritt 7 – Vergleichen: Wie gut wurde die Wahrheit getroffen?
Für jeden aktiven Kanal vergleichen wir das geschätzte β̂ mit dem eingemischten wahren β und
berechnen **Bias, Varianz und RMSE** – getrennt für HbO und HbR. Zusätzlich prüfen wir die
**HbO/HbR-Plausibilität** (ist HbR etwa −0,4-mal HbO, wie eingemischt?).

![Ein-Kanal-Fit](figures/02_kanal_fit.png)
*Abb. 2 – Ein einzelner Kanal über die Zeit. Grau: gemessenes Ruhesignal mit eingemischter
Aktivität. Rot: die eingemischte Wahrheit (Ground Truth). Blau: die vom GLM zurückgewonnene
HRF. Gepunktet: der Gesamt-Fit inklusive Drift. Grüne Streifen markieren die Stimulus-Zeiten.
Dass die blaue Kurve der roten Wahrheit folgt, zeigt: die Aktivität wird zurückgewonnen.*

![Rückgewinnung: geschätzt vs. wahr](figures/03_beta_recovery.png)
*Abb. 3 – Geschätzte gegen wahre Aktivierungsstärke, ein Punkt je Kanal. Die gestrichelte
Linie ist die Ideallinie (Schätzung = Wahrheit). Punkte darüber = Überschätzung. Bei HbO
liegen die aktiven Kanäle überwiegend oberhalb → systematische Überschätzung. Die vielen
Punkte bei „wahres β ≈ 0" sind inaktive Kanäle (reines Rauschen).*

### Schritt 8 – Bilder zeichnen
Zur Kontrolle erzeugt `demo_figures.py` Diagramme, u. a. **Kopf-Karten (scalp plots)**, auf
denen jeder Kanal an seiner Position auf dem Kopf farbig dargestellt wird (z. B. „wie weit
lag die Schätzung daneben?").
*(Cedalion: `cedalion.vis.anatomy.scalp_plot(...)`; Vorbild Notebook `tutorial/7` und
`modeling/32`.)*

![Kopf-Karte des Schätzfehlers](figures/04_scalp_abweichung.png)
*Abb. 4 – Kopf-Karte des Schätzfehlers (β̂ − Wahrheit) je Kanal, getrennt für HbO und HbR.
Jeder Kanal ist an seiner Position auf dem Kopf eingezeichnet; die Farbe zeigt die Abweichung
(blau = zu niedrig, rot = zu hoch, weiß = korrekt). Rote/blaue Quadrate sind die Sender/
Empfänger auf dem Kopf.*

![Kopf-Karte der relativen Abweichung](figures/05_scalp_rel_abweichung.png)
*Abb. 5 – Relative Abweichung, nur in der aktiven Region gezeigt (graue Kanäle liegen
außerhalb der Aktivierung). Oben: relativer Fehler der Amplitude; unten: ein Formfehler-Maß.*

---

## 5. Der systematische Vergleich (der „Sweep")

Ein einzelner Durchlauf beantwortet die Forschungsfrage noch nicht. Dafür wiederholen wir
die Schritte 4–7 **systematisch für alle Kombinationen** von Einflussgrößen. Das erledigt
`sweep.py`. „Sweep" heißt sinngemäß „alles einmal durchfahren".

Variiert werden sechs Achsen:
1. **Driftregressor-Familie** (die Hauptfrage): 15 Varianten – Polynom Ordnung 1–5, DCT mit
   3 Grenzfrequenzen, Legendre Ordnung 1/3/5, **B-Splines** (5 und 8 Knoten), „kein Drift",
   Butterworth-Hochpass. Dazu als Filter-Alternativen Tiefpass 0,5 Hz und Bandpass
   0,01–0,5 Hz.
2. **Fensterlänge:** Wie lang ist der ausgewertete Zeitausschnitt? (90 s, 180 s und 368 s =
   die ganze Aufnahme.) Kürzere Fenster sind schwieriger.
3. **Konstellation:** Welche *weiteren* Zutaten sind im Rezept? – `baseline` (nur HRF+Drift),
   `motion` (Bewegungssignale aus dem eingebauten Beschleunigungssensor), `global`
   (ein „Gesamtsignal"-Regressor als Ersatz für systemische Störungen) sowie **echte
   Short-Channel-Regression** in zwei Varianten: `short_avg` (Mittel aller kurzen Kanäle)
   und `short_maxcorr` (je langem Kanal der am stärksten mit ihm korrelierende kurze).
4. **Bewegungskorrektur:** `wavelet` oder `tddr+wavelet`. Warum das eine eigene Achse ist,
   steht im Kasten in Kapitel 4, Schritt 2 – kurz: die beiden Verfahren gehen völlig
   unterschiedlich mit dem Driften *und* mit der Hirnantwort um.
5. **Schätzverfahren:** `ar_irls` (das Standardverfahren dieser Arbeit) und `ols` (das
   einfache Verfahren ohne Rauschmodell). AR-IRLS ist rund 50-mal langsamer, die Achse
   kostet also fast nichts.
6. **Zufalls-Wiederholungen (Seeds):** Die künstlichen Reize werden zu leicht anderen
   Zeitpunkten platziert (mehrere Wiederholungen). Erst der Vergleich **über diese
   Wiederholungen** liefert eine echte Bias-/Varianz-Aussage (nicht der Vergleich über Kanäle).

Das ergibt in der Hauptversion (v4) 15 × 3 × 5 × 2 × 4 = **1800 Auswertungen** in 7,5 Stunden.
Ergebnisse landen in `results/` (Tabelle `sweep_summary.csv`, Rohdaten
`sweep_per_channel.nc`), Abbildungen und Markdown-Tabellen erzeugt `sweep_report.py`.

**Zu den kurzen Kanälen.** Ein Kanal mit kleinem Abstand zwischen Sender und Empfänger hat
eine flache „Lichtbanane": sein Licht erreicht das Gehirn gar nicht und misst nur Kopfhaut
und Schädel – also **systemische Störungen ohne Hirnsignal**. Nimmt man ihn als Regressor
auf, lässt sich dieser Anteil aus den langen Kanälen herausrechnen; ausgewertet wird dann
nur über die langen. Der Ruhedatensatz hat keine echten Short-Separation-Kanäle (< 10 mm),
aber bei einer Grenze von **1,8 cm** fallen 37 der 520 Kanäle in die kurze Gruppe
(15,8–17,8 mm). Genau diese Grenze war die Vorgabe aus dem Betreuungsgespräch; dasselbe
Vorgehen nutzt das Cedalion-Workshop-Notebook 36 mit 22,5 mm. **Als Einschränkung zu nennen:**
15,8–17,8 mm sehen noch etwas Kortex, der Regressor entfernt also potenziell auch echtes
Signal. Die Grenze ist außerdem nicht unempfindlich – bei 1,9 cm wären es schon 114 Kanäle.

Damit der Vergleich fair ist, wird die künstliche Aktivität **nicht in die kurzen Kanäle
eingemischt** (sie sehen ja kein Gehirn). Ohne diese Vorsichtsmaßnahme enthielte der
Short-Channel-Regressor die Hirnantwort selbst und würde sie aus den langen Kanälen
herausrechnen – gemessen bekämen die kurzen Kanäle sonst bis zu 0,54 µM der eingemischten
0,60 µM ab.

> **Wichtiger methodischer Hinweis (aus dem Betreuungsgespräch):** Bei Auswertungen *mit*
> Driftregressoren wird das Signal **nicht** hochpassgefiltert – ein Hochpassfilter würde
> genau den Drift entfernen, den die Regressoren modellieren sollen, und den Vergleich
> verfälschen. Der Butterworth-Hochpass ist deshalb die **eigenständige Alternative**
> (Filter *statt* Regressoren), nicht zusätzlich.

### Zusatz-Analyse 1: Formfehler mit flexibler Formvorlage

Bisher nutzen Einmischung und Auswertung dieselbe HRF-Formvorlage (die Gamma-Welle). Dadurch
kann sich zwischen „eingemischt" und „geschätzt" nur die **Höhe** unterscheiden, nicht die
**Form**. Um zu prüfen, ob eine schlechte Drift-Modellierung auch die *Form* der geschätzten
Antwort verzerrt, wird zusätzlich mit einer **flexiblen** Formvorlage ausgewertet
(`GaussianKernels` – eine Reihe von Glockenkurven, die zusammen praktisch jede Kurvenform
annehmen können). Als Maß dient die **Form-Treue**: die Korrelation zwischen zurückgewonnener
und eingemischter HRF-Kurve. Diese ist *skaleninvariant* – sie misst also reine Form,
unabhängig von der Höhe (1,0 = perfekte Form). Skript: `flex_basis.py`.
*(Cedalion: `glm.GaussianKernels`; HRF-Formvorlagen aus Notebook `modeling/31`.)*

### Zusatz-Analyse 2: Statistische Absicherung – Signifikanz und Mehrfachvergleichs-Korrektur

Für die spätere Auswertung echter Daten – und als methodische Vollständigkeit – wird die
statistische Absicherung demonstriert: Für jeden Kanal wird geprüft, ob die geschätzte
Aktivierung **statistisch signifikant** von null verschieden ist (Signifikanz = Schätzung
geteilt durch ihren Standardfehler, daraus ein p-Wert). Weil **sehr viele Kanäle gleichzeitig**
getestet werden, muss für **multiples Testen korrigiert** werden (Verfahren Benjamini-Hochberg,
„FDR", Niveau 5 %) – sonst entstünden allein durch Zufall zahlreiche Falsch-Positive.

Da wir in der Simulation die Wahrheit kennen (welche Kanäle wirklich aktiv sind = der Blob),
lässt sich die **Detektionsgüte** direkt messen:
- **Sensitivität** – Anteil der wirklich aktiven Kanäle, die auch erkannt werden;
- **Spezifität** – Anteil der Ruhe-Kanäle, die korrekt als inaktiv gelten;
- **Youden-J** (= Sensitivität + Spezifität − 1) als Gesamtmaß.

So zeigt sich, welche Driftfamilie die zuverlässigste Aktivierungs-Erkennung liefert. Dieses
Skript (`detection.py`) fittet über **alle** 544 Kanäle (nur so sind Falsch-Positive
bewertbar) und demonstriert damit die komplette Inferenz-Pipeline für die realen DOT-Daten.
*(Cedalion: `.sm.params` + `.sm.regressor_variances()`; FDR über `statsmodels`.)*

---

## 6. Was bisher herausgekommen ist

### Teil A – Simulation (bekannte Wahrheit)

Die Simulationsstudie lief in vier Ausbaustufen. **v1** mischte die Aktivität überall gleich
stark ein (räumlich „flach") – das führte zu einem Trugschluss beim Global-Regressor und wurde
in **v2** durch den räumlichen Blob behoben. **v3** ergänzte B-Splines, lange Fenster und die
beiden Zusatz-Analysen. **v4** ist der aktuelle Stand: 15 Driftfamilien × 3 Fenster ×
5 Konstellationen (jetzt mit echter Short-Channel-Regression) × 2 Bewegungskorrekturen ×
4 Wiederholungen = **1800 Auswertungen** in 7,5 Stunden. Die folgenden Zahlen stammen aus
**v4**. Alle Tabellen dazu: `results/tables.md`.

**Ergebnis 1 – Systemische Störungen sind der größte Hebel (für HbO).**
Ohne systemischen Regressor wird HbO deutlich **überschätzt** (Bias +0,104 µM auf eine
Wahrheit von 0,600 µM bei 90-s-Fenstern sogar +0,570). Ein systemischer Regressor bringt das
weitgehend in Ordnung. Grund: Benachbarte aktive Kanäle teilen sich systemische Körpersignale
(z. B. Blutdruckwellen); ein reines Drift-Modell zieht diese teils fälschlich in die
Hirnaktivität. → **Der systemische Regressor wirkt sich stärker aus als die Wahl der
Driftfamilie.** Bewegungs*regressoren* aus dem Beschleunigungssensor bleiben dagegen praktisch
ohne Wirkung – erwartet, da die *synthetische* HRF nicht mit echter Bewegung koppelt.

**Ergebnis 1b – Short-Channel-Regression schlägt den Global-Regressor, auch wenn die Zahlen
zunächst das Gegenteil sagen.** Rein nach Bias sieht `global` besser aus (+0,021 gegen +0,056
bei `short_avg`). Das ist aber kein Vorteil, sondern eine Verrechnung zweier Fehler: der
Global-Regressor mittelt über **alle** Kanäle, also auch über die aktivierten, und enthält
dadurch im Mittel **0,040 µM der eingemischten Hirnantwort** – die kurzen Kanäle enthalten
konstruktionsbedingt **0,000 µM**. Der Unterschied im Bias beträgt 0,035 µM, der Hirnsignal-
Anteil des Global-Regressors 0,040 µM: er „gewinnt", indem er einen Teil des Gesuchten
wegrechnet. **`short_avg` ist der methodisch saubere Regressor.**

**Ergebnis 2 – HbR ist unempfindlicher als HbO.** Der HbR-Fehler (~0,07 µM) ändert sich über
alle Konstellationen kaum – HbR ist weniger von systemischen Störungen betroffen (physiologisch
plausibel, eigenständiges Ergebnis).

**Ergebnis 3 – Die Fensterlänge dominiert.** HbO-RMSE (baseline, Wavelet): **90 s ≈ 0,30 →
180 s ≈ 0,21 → 368 s ≈ 0,09 µM** – die ganze Aufnahme ist rund dreimal genauer als ein
90-s-Fenster. Mehr Daten (und mehr Reize im Fenster) → stabilere Schätzung.

**Ergebnis 4 – Familienwahl: kontextabhängig, insgesamt moderat.**
- **Lange Fenster (368 s):** Alle Driftregressor-Familien rücken eng zusammen (HbO-RMSE
  0,119–0,125). → *Dass* ein Drift-Modell da ist, zählt; *welche* Familie, kaum.
- **Kurze Fenster (90 s):** Hier trennen sich die Familien; **sehr hohe Ordnungen überanpassen**
  (B-Spline mit 8 Knoten fällt auf 0,619 gegenüber 0,49–0,52 der übrigen).
- **B-Splines** liegen ansonsten gleichauf mit Polynom/Legendre/DCT – kein Vor-, kein Nachteil.

**Ergebnis 4b – Die Bewegungskorrektur ist wichtiger als die Driftfamilie, und sie ist eine
Falle.** TDDR halbiert den HbO-Fehler (RMSE 0,136 gegen 0,199) und beseitigt scheinbar den
Bias (−0,040 gegen +0,070). Das ist aber wieder keine bessere Schätzung, sondern die
Verrechnung zweier Fehler: TDDR **dämpft die eingemischte Hirnantwort auf 70 %** ihrer Höhe,
und diese Dämpfung hebt die systemisch bedingte Überschätzung zufällig auf. Sichtbar wird das
an **HbR**, wo es keine Überschätzung zu kompensieren gibt – dort verschlechtert TDDR den
Fehler von +0,020 auf +0,057, also von 8 % auf 24 % Betragsunterschätzung. Über alle
Driftfamilien hinweg ist der Bias praktisch identisch; der Sprung kommt allein von der
Vorverarbeitung (Abb. 14).

**Konsequenz für die Auswertung:** Bias **getrennt nach HbO und HbR** betrachten, nicht nur
den RMSE. Wer nur den RMSE ansieht, hält TDDR für das bessere Verfahren.

**Ergebnis 5 – Plausibilität.** Das zurückgewonnene HbR/HbO-Verhältnis liegt bei −0,25 bis
−0,34 (eingemischt −0,4); es bessert sich mit systemischem Regressor und mit längerem Fenster
(368 s: −0,342) und verschlechtert sich leicht mit TDDR – konsistent mit dessen HbR-Dämpfung.

**Ergebnis 6 – Die Form der HRF wird treu zurückgewonnen (Zusatz-Analyse 1).** Mit der
flexiblen Formvorlage ist die **Form-Treue hoch (~0,87–0,91) und über die Driftfamilien
ähnlich** → die Drift-Wahl verzerrt die *Form* der geschätzten Antwort nur wenig (die *Höhe*
schon, s. Ergebnis 1/5). Nebenbefund: die flexible Basis **überschätzt die Amplitude**
(+30–70 %) – der bekannte Bias-Varianz-Preis flexibler HRF-Modelle.

**Ergebnis 7 – Aktivierungs-Detektion nach FDR (Zusatz-Analyse 2).** Über alle 544 Kanäle mit
FDR-Korrektur (q = 0,05): **moderate Drift-Modelle detektieren am besten** (Youden-J HbO/HbR:
DCT 0,01 ≈ Polynom 3 ≈ Legendre 3 ≈ B-Spline 5), „kein Drift" hat die höchste **Spezifität**
(wenige Falsch-Positive), aber die niedrigste **Sensitivität** (verpasst echte Aktivierungen).
Die Inferenz-Pipeline (Signifikanz + FDR) steht damit bereit für die realen Daten.

### Teil B – Reale Daten (keine Wahrheit bekannt)

Stufe 2 der Arbeit läuft auf einem öffentlichen Finger-Tapping-Datensatz (Khan, Nazeer &
Mirtaheri 2026): **25 Probanden**, 69 Aufnahmen, 48 Kanäle, Einzelfinger-Tapping der rechten
Hand, 10 s Block gegen 10 s Ruhe, 15 Tapping-Blöcke je Aufnahme. Ausgewertet wurden
14 Driftfamilien × 2 Konstellationen × 2 Schätzverfahren über alle Aufnahmen =
**3864 Auswertungen** in 3,6 Stunden (`realglm.py`).

**Der entscheidende Unterschied:** Hier gibt es **keine Ground Truth**. Der Fehler gegen die
Wahrheit, an dem in der Simulation alles hing, existiert nicht. Stattdessen drei Kriterien,
die ohne Wahrheit auskommen – und eines davon trägt die Hauptlast:

> **Reproduzierbarkeit.** Die Probanden haben je 2–3 Durchgänge. Die Korrelation der
> β-Karten zwischen den Durchgängen **eines** Probanden misst, wie stabil eine Driftfamilie
> schätzt. Eine Familie, die Rauschen als Aktivierung modelliert, ist zwischen Durchgängen
> inkonsistent. Über 74 Durchgangspaare gemittelt ist das ein belastbares Gütemaß.

**Ergebnis 6 – Den Drift zu modellieren schlägt ihn wegzufiltern.** Der Butterworth-Hochpass
ist die **schlechteste** Option (Reproduzierbarkeit 0,443 mit AR-IRLS) – schlechter als *gar
kein* Driftmodell (0,494). Das bestätigt den methodischen Hinweis aus dem Betreuungsgespräch
jetzt an echten Daten statt nur als Argument.

**Ergebnis 7 – DCT mit ~0,02 Hz ist die stabilste Familie**, und zwar unabhängig vom
Schätzverfahren (0,566 mit AR-IRLS, 0,564 mit OLS). Danach DCT 0,01 (0,545/0,547) und
B-Spline 8 (0,535/0,535). Die Simulation hatte DCT ~0,01–0,02 Hz ebenfalls vorn – die beiden
Stufen der Arbeit stützen sich gegenseitig.

**Ergebnis 8 – Das Driftmodell zählt bei OLS mehr als bei AR-IRLS.** Spannweite der
Reproduzierbarkeit über die Familien: **0,200 bei OLS gegen 0,123 bei AR-IRLS**. AR-IRLS
fängt einen Teil der niederfrequenten Struktur über sein Rauschmodell ab und ist dadurch
robuster gegen eine schlechte Driftwahl. Bei OLS trägt allein die Designmatrix – dort bricht
`poly:3` auf 0,440 ein, während `dct:0.02` bei 0,564 bleibt.

**Ergebnis 9 – Tiefpassfilterung und AR-IRLS schließen einander aus.** Mit einem Tiefpass bei
0,5 Hz kollabiert die Schätzung auf **1e-05 µM**, fünf Größenordnungen zu klein; mit OLS
liefert derselbe Datensatz normale Werte, und ein reiner Hochpass läuft mit AR-IRLS
problemlos. Ursache ist die Prewhitening-Stufe: AR-IRLS schätzt ein Rauschmodell und wendet
dessen Inverse an. Bei 3,906 Hz Abtastrate entfernt ein Tiefpass bei 0,5 Hz rund **drei
Viertel des Spektrums** (0,5 Hz = 0,256 × Nyquist); oberhalb davon hat das Residuum keine
Leistung mehr, der Whitening-Filter müsste dort unendlich verstärken. Übrig bleibt
numerisches Rauschen. **Das ist kein Rechenfehler, sondern eine Eigenschaft der Kombination**
– und sie hängt an der Abtastrate: auf dem 9-Hz-Ruhedatensatz läge 0,5 Hz bei 0,51 × Nyquist
und wäre weit unkritischer.

**Ergebnis 10 – Plausibilität auf realen Daten.** HbO und HbR sind mit **−0,88** deutlich
antikorreliert, das Signal ist also physiologisch. Das HbR/HbO-Verhältnis liegt bei −0,18
(baseline) und verbessert sich mit dem systemischen Regressor auf −0,29 (AR-IRLS) bzw. −0,36
(OLS); erwartet wird ~−0,4. Der systemische Regressor halbiert dabei die Zahl signifikanter
Kanäle (27–31 → 10–19 von 48) – dasselbe Muster wie in der Simulation.

### Nutzungsempfehlung (Stand: beide Stufen)

1. **Systemischen Regressor verwenden** – der größte Genauigkeits-Hebel für HbO. Wenn die
   Montage kurze Kanäle hat, **Short-Channel-Regression** statt Global-Mittelwert: der
   Global-Regressor enthält das gesuchte Signal mit und rechnet es teilweise weg.
2. **Möglichst lange Auswertefenster.**
3. **Beim Drift: DCT mit Grenzfrequenz ~0,01–0,02 Hz.** In beiden Stufen der Arbeit die
   stabilste Wahl. Sehr hohe Ordnungen bei kurzen Fenstern meiden.
4. **Den Drift modellieren, nicht wegfiltern.** Der Hochpass als Ersatz für Driftregressoren
   war auf realen Daten die schlechteste Option.
5. **Tiefpass und AR-IRLS nicht kombinieren** (Ergebnis 9).
6. **Bei der Bewegungskorrektur genau hinsehen:** TDDR dämpft die Hirnantwort auf 70 % und
   kann dadurch Fehler kaschieren, die im RMSE gut aussehen. Bias getrennt nach HbO und HbR
   prüfen.

### Abbildungen zum Hauptsweep

![RMSE je Driftfamilie](figures/10_sweep_rmse_by_family.png)
*Abb. 6 – Gesamtfehler (RMSE) je Driftregressor-Familie, für HbO (oben) und HbR (unten), je
Fensterlänge (90 / 180 / 368 s). Niedriger = besser. Sichtbar: längere Fenster sind durchweg
genauer; bei 368 s liegen die Familien eng beieinander.*

![Bias-Varianz-Zerlegung](figures/11_sweep_bias_var.png)
*Abb. 7 – Aufteilung des Fehlers je Familie in Bias (systematische Verzerrung) und Streuung.*

![Konstellations-Effekt](figures/12_sweep_constellation_effect.png)
*Abb. 8 – Wirkung der Konstellation. Der Global-Regressor (grün/rot) senkt den HbO-Fehler
(oben) deutlich gegenüber baseline/motion (blau/orange). Bei HbR (unten) ist der Effekt klein.*

![HbO/HbR-Plausibilität](figures/13_sweep_plausibility.png)
*Abb. 9 – Plausibilität: zurückgewonnenes HbR/HbO-Verhältnis je Familie; gestrichelt der wahre,
eingemischte Wert (−0,4).*

### Abbildungen zu den Zusatz-Analysen

![Form-Treue je Familie](figures/20_flex_shape_corr.png)
*Abb. 10 – Form-Treue (Korrelation rückgewonnene vs. injizierte HRF) je Driftfamilie, flexible
Formvorlage. Hoch und über Familien ähnlich → die Drift-Wahl verzerrt die HRF-Form kaum.*

![Rückgewonnene HRF-Formen](figures/21_flex_shape_curves.png)
*Abb. 11 – Beispiel: rückgewonnene HRF-Kurven (flexible Basis) je Driftfamilie gegen die
injizierte Ground-Truth-HRF (schwarz), ein Kanal.*

![Detektion nach FDR](figures/22_detection.png)
*Abb. 12 – Aktivierungs-Detektion nach FDR-Korrektur (q = 0,05) je Driftfamilie:
Sensitivität (blau) und Spezifität (orange), HbO/HbR. Moderate Drift-Modelle geben die beste
Balance; „kein Drift" ist spezifisch, aber unsensitiv.*

![Motion-Achse](figures/14_sweep_motion_axis.png)
*Abb. 13 – Die Bewegungskorrektur als eigene Achse. Balken = Bias, Striche = RMSE. Links HbO,
rechts HbR. Entscheidend ist, dass der Bias über **alle** Driftfamilien praktisch konstant
bleibt – der Sprung zwischen den beiden Balkenfarben kommt allein von der Vorverarbeitung.
TDDR (orange) drückt den HbO-Bias auf null, verdreifacht ihn aber bei HbR: dort fehlt die
Überschätzung, gegen die sich die Dämpfung verrechnen könnte.*

### Abbildungen zu den realen Daten

![Reproduzierbarkeit](figures/30_real_reliability.png)
*Abb. 14 – Das Hauptergebnis der realen Daten. Median-Korrelation der β-Karten zwischen den
Durchgängen eines Probanden, über 74 Durchgangspaare. Höher = stabilere Schätzung. Die
DCT-Familie (orange) liegt vorn, die Filter-Alternativen (rot) klar hinten – schlechter als
gar kein Driftmodell (braun). „n.a." markiert die mit AR-IRLS nicht auswertbare Kombination
(Ergebnis 9).*

![Signifikante Kanäle](figures/31_real_significant.png)
*Abb. 15 – Zahl der Kanäle, die die FDR-Korrektur überstehen (von 48). Der systemische
Regressor (orange) entfernt bei AR-IRLS rund die Hälfte der Signifikanz – dasselbe Muster
wie in der Simulation.*

![Gruppen-Aktivierung auf dem Kopf](figures/32_real_scalp.png)
*Abb. 16 – Die eigentliche Frage: **wo** sitzt die Aktivierung? Gruppen-β über 25 Probanden
(oben) und t-Werte (unten), HbO links, HbR rechts. Erwartet wird die stärkste Antwort über
dem linken, kontralateralen Motorkortex, da mit der rechten Hand getappt wurde.*

![Plausibilität real](figures/33_real_plausibility.png)
*Abb. 17 – HbR/HbO-Verhältnis (links, gestrichelt der physiologisch erwartete Wert −0,4) und
Antikorrelation zwischen HbO und HbR (rechts). Je Punkt eine Driftfamilie.*

---

## 7. Die Dateien im Überblick

Alle Dateien liegen in diesem Repository-Ordner `fnirs-drift-glm/` (als eigenständiges
Git-Repo neben `../cedalion/` nutzbar).

| Datei | Aufgabe |
|---|---|
| `preprocess.py` | **Vorverarbeitung** (Kapitel 4, Schritt 2): OD-Umrechnung mit Baseline, Bewegungskorrektur, Rückweg zur Amplitude, Kanalmasken, Pruning. Enthält auch die Diagnose, ob ein Verfahren ins Driftband eingreift. |
| `shortchannel.py` | **Kurze Kanäle:** Distanzanalyse, Long/Short-Split bei 1,8 cm, die drei Regressor-Varianten. |
| `pipeline.py` | **Kernstück der Simulation.** Mischt die künstliche HRF als räumlichen Blob in die optische Dichte ein – *vor* der Bewegungskorrektur – und baut die Designmatrix. |
| `sweep.py` | Der **systematische Vergleich** über alle Kombinationen (Kapitel 5). Schreibt Ergebnisse + eine Live-Fortschrittsdatei. |
| `compare_preprocessing.py` | Vergleicht die Vorverarbeitungs-Varianten gegen die β-Rückgewinnung (Grundlage für Ergebnis 4b). |
| `realdata.py` | **Stufe 2:** Einlesen der realen SNIRF-Dateien, Inventar, Stimulus-Zuordnung (inkl. der abweichenden Kodierung bei S25), datengetriebene Amplitudengrenzen. |
| `realglm.py` | **Stufe 2, Kern:** GLM je Driftfamilie über alle Probanden, Gruppen-t-Test, FDR, Reproduzierbarkeit zwischen Durchgängen. |
| `realglm_report.py` | Abbildungen zu den realen Daten (Abb. 14–17), inkl. der Gruppen-β-Karte auf dem Kopf. |
| `sweep_report.py` | Erzeugt aus den Sweep-Ergebnissen die **Abbildungen** und **Markdown-Ergebnistabellen** (`results/tables.md`). |
| `flex_basis.py` | **Zusatz-Analyse 1:** flexible HRF-Formvorlage → Form-Treue (Formfehler unabhängig von der Höhe). |
| `detection.py` | **Zusatz-Analyse 2:** Signifikanz je Kanal + FDR-Korrektur → Detektionsgüte gegen die Ground Truth. |
| `demo_recovery.py` | **End-to-end-Demo (Zahlen):** ein Durchlauf, gibt Fehlerkennzahlen aus. |
| `demo_figures.py` | **End-to-end-Demo (Bilder):** Designmatrix, Ein-Kanal-Fit, β̂-vs-Wahrheit, Kopf-Karten. |
| `tests/` | Schnelle **Smoke-Tests** (pytest) für die Pipeline. |
| `results/` | Ausgaben: `sweep_summary.csv`, `sweep_per_channel.nc`, `tables.md`, `flex_basis_summary.csv`, `detection_summary.csv`, `*_progress.txt`. |
| `figures/` | Alle erzeugten Abbildungen (PNG). |
| `environment.lock.txt` | Exakte Versionen der Kern-Pakete (Reproduzierbarkeit). |
| `README.md` | Knappe technische Ausführ-/Reproduktions-Anleitung. |
| `../cedalion/` | Cedalion-Framework (Schwester-Ordner) inkl. Beispiel-Notebooks unter `../cedalion/examples/`. |

---

## 8. Wie man alles ausführt

Vorbereitet ist eine Python-Umgebung namens `cedalion`. Alle Befehle aus diesem Repo-Ordner:

```bash
# Umgebung aktivieren (einmal pro Terminal) und in den Repo-Ordner wechseln
source ~/anaconda3/etc/profile.d/conda.sh
cd fnirs-drift-glm

# 1) End-to-end-Demo mit Zahlen (~einige Minuten)
conda run -n cedalion python demo_recovery.py

# 2) Kontroll-Abbildungen erzeugen
conda run -n cedalion python demo_figures.py

# 3) Schnelltests (Sekunden–Minuten)
conda run -n cedalion python -m pytest tests -q          # Smoke-Tests
conda run -n cedalion python sweep.py pilot              # alle Code-Pfade, ~5 min
conda run -n cedalion python flex_basis.py test          # ~1 min
conda run -n cedalion python detection.py test           # ~1 min

# 4) Hauptstudie v4 (1800 Auswertungen, ~7,5 Stunden)
conda run -n cedalion python sweep.py v4
#    Fortschritt live verfolgen (in einem zweiten Terminal):
cat results/sweep_progress.txt

# 5) Zusatz-Analysen (schreiben ebenfalls *_progress.txt)
conda run -n cedalion python flex_basis.py               # ~20 min
conda run -n cedalion python detection.py                # ~60–75 min (Fits über ALLE Kanäle)
conda run -n cedalion python compare_preprocessing.py    # ~17 min

# 6) Auswertungs-Abbildungen + Ergebnistabellen erzeugen
conda run -n cedalion python sweep_report.py
```

**Stufe 2 – die realen Daten.** Sie liegen als SNIRF unter
`../FingerTappingDataset_Published2025/` (Unterordner je Proband):

```bash
# Inventar: prüft die Angaben aus dem Paper gegen die Dateien
conda run -n cedalion python realdata.py

# Hauptauswertung (3864 Auswertungen, ~3,6 Stunden)
conda run -n cedalion python realglm.py
cat results/realglm_progress.txt        # Fortschritt

# Abbildungen (die Kopf-Karte rechnet ~8 min nach; "quick" lässt sie weg)
conda run -n cedalion python realglm_report.py
conda run -n cedalion python realglm_report.py quick
```

*(Hinweis: Der Datensatz enthält zu jeder Aufnahme eine von den Autoren vorgefilterte
Fassung `*_CC_filtered.snirf`. Diese ist für den Driftregressor-Vergleich **unbrauchbar** –
sie hat den Hochpass bei 0,01 Hz bereits angewandt, also genau den Drift entfernt, um den es
geht. `realdata.find_files()` blendet sie deshalb standardmäßig aus.)*

*(Technischer Hinweis für WSL/Windows: Dateien immer aus der Linux-Umgebung heraus
bearbeiten, nicht mit nativen Windows-Werkzeugen – sonst werden durch Zeilenende-Umschreibung
scheinbar alle Cedalion-Dateien „verändert".)*

---

## 9. Was noch kommt (Ausblick)

**Beide Stufen der Arbeit sind rechnerisch abgeschlossen.** Die Simulation liegt als v4 vor
(1800 Auswertungen), die realen Daten als Gruppenanalyse über 25 Probanden (3864
Auswertungen). Was seit der letzten Fassung dazukam:

- **Vorverarbeitung nach Betreuungsvorgabe** – Bewegungskorrektur auf der optischen Dichte,
  danach zurück zur Amplitude, erst dann Kanalbewertung (Kapitel 4, Schritt 2).
- **Echte Short-Channel-Regression** bei 1,8 cm statt nur des Global-Mittelwerts.
- **Bewegungskorrektur und Schätzverfahren als eigene Achsen** – beide stellten sich als
  einflussreicher heraus als die Driftfamilie selbst.
- **Stufe 2 auf realen Daten** mit Gruppenstatistik, FDR und dem wahrheitsfreien
  Reproduzierbarkeits-Kriterium.

**Noch offen – Kanalraum gegen Bildraum.** Alle bisherigen Auswertungen laufen im
**Kanalraum**: das Ergebnis ist ein Wert je Messkanal. Um Aktivität einer **Hirnregion**
zuzuordnen, muss man in den **Bildraum** – also aus den Kanalwerten zurückrechnen, wo im
Gehirn die Änderung entstanden ist. Das ist ein *inverses Problem* (viele mögliche
Hirnbilder erklären dieselben Kanalwerte, man braucht Zusatzannahmen, um eines auszuwählen).

Offen ist damit die Frage: **Macht es einen Unterschied, ob man das GLM im Kanalraum oder im
Bildraum rechnet?** – und zwar für beide Datensätze. Cedalion bringt das Verfahren mit
(`cedalion.dot.ImageRecon`, Tutorial-Notebook 5, das ausdrücklich auch dünn besetzte Montagen
abdeckt). Für den Ruhedatensatz liegt die nötige Sensitivitätsmatrix fertig vor; für die
48-Kanal-Montage der realen Daten müsste sie über ein Kopfmodell berechnet werden – das ist
der Punkt, an dem der Aufwand zu klären ist.

**Abschließend – Verschriftlichung (durch den Autor):**
- Methoden-, Ergebnis- und Diskussionsteil der Bachelorarbeit; Einordnung in die Fachliteratur
  und Ableitung der finalen **Nutzungsempfehlung**, welcher Driftregressor unter welchen
  Bedingungen die zuverlässigste Schätzung liefert. (Diese Dokumentation und die
  Ergebnistabellen in `results/tables.md` liefern dafür das Material.)

---

## 10. Glossar

- **AR-IRLS** – robustes GLM-Schätzverfahren, das zeitlich „nachhallendes" Rauschen
  berücksichtigt (modelliert das Rauschen, nicht den Drift).
- **β / β̂** – wahre bzw. geschätzte Aktivierungsstärke an einem Kanal.
- **Bias** – systematische Verzerrung der Schätzung (im Mittel zu hoch/zu niedrig).
- **B-Splines** – stückweise glatte Kurvenstücke; hier eine Driftfamilie (eigene Regressorspalten).
- **Butterworth-Hochpass** – Filter, der langsames Driften vor der Auswertung entfernt;
  Alternative zu Driftregressoren.
- **Cedalion** – das verwendete Python-Framework für fNIRS/DOT.
- **DCT** – diskrete Cosinus-Basis; Driftfamilie aus Cosinus-Wellen bis zu einer Grenzfrequenz.
- **Designmatrix** – Tabelle aller „Zutaten" (Regressoren) im GLM.
- **DOT** – Diffuse Optische Tomografie; hochkanaliges fNIRS mit räumlicher Auflösung.
- **Drift** – langsames, nicht-hirnbezogenes Wegwandern der Messwerte.
- **FDR (False Discovery Rate)** – Korrektur für multiples Testen (Benjamini-Hochberg):
  begrenzt den erwarteten Anteil an Falsch-Positiven, wenn viele Kanäle gleichzeitig getestet werden.
- **Flexible Formvorlage** – HRF-Basis, deren Kurvenform frei ist (`GaussianKernels`);
  ermöglicht die Messung des Formfehlers unabhängig von der Höhe.
- **fNIRS** – funktionelle Nahinfrarotspektroskopie; Hirndurchblutungsmessung mit Licht.
- **GLM** – General Linear Model; erklärt das Signal als gewichtete Summe bekannter Zutaten.
- **Ground Truth** – der bekannte wahre Wert (in der Simulation von uns eingemischt).
- **HbO / HbR** – sauerstoffreiches / sauerstoffarmes Hämoglobin; steigen bzw. fallen bei
  Aktivierung.
- **HRF** – hämodynamische Antwortfunktion; typische Form der Durchblutungsantwort auf einen Reiz.
- **Kanal** – Messpunkt aus Sender+Empfänger; viele Kanäle = Karte der Hirnoberfläche.
- **Konstellation** – welche zusätzlichen Regressoren (Bewegung, Global) im Modell sind.
- **Legendre-Polynome** – mathematisch besonders „saubere" Polynome; eine Driftfamilie.
- **RMSE** – Gesamtfehlermaß (kleiner = besser); fasst Bias und Varianz zusammen.
- **Scalp plot** – Kopf-Karte: jeder Kanal farbig an seiner Kopfposition dargestellt.
- **Seed** – Startwert des Zufallsgenerators; unterschiedliche Seeds = unterschiedliche
  (reproduzierbare) Zufalls-Wiederholungen.
- **Sensitivität (TPR)** – Anteil der wirklich aktiven Kanäle, die (nach FDR) erkannt werden.
- **SNR** – Signal-Rausch-Verhältnis; Maß für Kanalqualität.
- **Spezifität (TNR)** – Anteil der inaktiven Kanäle, die korrekt als inaktiv gelten.
- **Sweep** – systematisches Durchrechnen aller Einstellungs-Kombinationen.
- **Varianz** – wie stark die Schätzung zwischen Wiederholungen schwankt.
- **Youden-J** – Gesamtmaß der Detektion: Sensitivität + Spezifität − 1 (1 = perfekt).

---

*Diese Dokumentation beschreibt den Stand vom 13. Juli 2026. Die inhaltliche Roadmap steht
in Kapitel 9 (Ausblick).*
