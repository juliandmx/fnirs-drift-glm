#!/usr/bin/env bash
# Vollstaendige Bildraum-Laeufe (Version 5), SEQUENZIELL.
#
# Sequenziell und nicht parallel, weil der Speicher der Engpass ist: die
# nn22-Sensitivitaetsmatrix belegt ~300 MB, eine ImageRecon-Instanz darauf aehnlich viel.
# Auf dieser Maschine (7,8 GB) wurden zwei parallele Bildraum-Laeufe vom OOM-Killer beendet.
#
# Fortschritt live:  tail -f results/imageglm_progress.txt results/msglm_progress.txt
# Logs:              results/run_v5_imageglm.log  results/run_v5_msglm.log
set -u
cd "$(dirname "$0")"

# Sperre. Zwei Ketten gleichzeitig killen sich gegenseitig ueber den Speicher -- genau das
# ist beim Entwickeln passiert: die erste Kette wurde beim imageglm-Schritt abgebrochen,
# ist daraufhin auf msglm weitergeschaltet, und der Neustart lief dagegen. Ergebnis: der
# neue imageglm-Lauf wurde nach einer Minute vom OOM-Killer beendet.
LOCK="results/.run_v5.lock"
mkdir -p results
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "Es laeuft schon eine Kette (PID $(cat "$LOCK")). Abbruch." >&2
    echo "Falls das ein Ueberrest ist: kill -9 -\$(cat $LOCK); rm $LOCK" >&2
    exit 1
fi
echo $$ > "$LOCK"
# Beim Beenden -- auch bei Abbruch -- die ganze Prozessgruppe mitnehmen, damit kein
# Teilschritt weiterlaeuft und in den naechsten Lauf hineinrechnet.
cleanup() { rm -f "$LOCK"; kill -- -$$ 2>/dev/null; }
trap cleanup EXIT INT TERM

source ~/anaconda3/etc/profile.d/conda.sh
CR="conda run --no-capture-output -n cedalion python -u"

echo "=== $(date +%H:%M) imageglm (Simulation, Bildraum) ==="
$CR imageglm.py > results/run_v5_imageglm.log 2>&1
echo "    exit=$?  $(date +%H:%M)"

echo "=== $(date +%H:%M) msglm (Multisubject) ==="
$CR msglm.py > results/run_v5_msglm.log 2>&1
echo "    exit=$?  $(date +%H:%M)"

echo "=== $(date +%H:%M) Abbildungen 19-23 ==="
$CR imagespace_report.py > results/run_v5_report.log 2>&1
echo "    exit=$?  $(date +%H:%M)"
echo "=== fertig $(date +%H:%M) ==="
