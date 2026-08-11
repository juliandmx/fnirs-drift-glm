#!/usr/bin/env bash
# Vollstaendige Bildraum-Laeufe (Version 5), SEQUENZIELL.
#
# Sequenziell und nicht parallel, weil der Speicher der Engpass ist: die
# nn22-Sensitivitaetsmatrix belegt ~300 MB, eine ImageRecon-Instanz darauf aehnlich viel.
# Auf dieser Maschine (7,8 GB) wurden zwei parallele Bildraum-Laeufe vom OOM-Killer beendet.
#
# Fortschritt live:  tail -f results/logs/*_progress.txt
# Logs:              results/logs/run_v5_*.log
set -u
cd "$(dirname "$0")"

# Sperre. Zwei Ketten gleichzeitig killen sich gegenseitig ueber den Speicher -- genau das
# ist beim Entwickeln passiert: die erste Kette wurde beim imageglm-Schritt abgebrochen,
# ist daraufhin auf msglm weitergeschaltet, und der Neustart lief dagegen. Ergebnis: der
# neue imageglm-Lauf wurde nach einer Minute vom OOM-Killer beendet.
mkdir -p results/logs
LOCK="results/logs/.run_v5.lock"
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
CR="conda run --no-capture-output -n cedalion python -u -m"

# Welche Schritte laufen sollen: ohne Argumente alle, sonst nur die genannten.
# Beispiel:  ./run_v5.sh msglm report      (imageglm ueberspringen)
STEPS="${*:-imageglm msglm report}"

run_step() {                       # $1 = Name, $2 = Modul
    case " $STEPS " in *" $1 "*) ;; *) return 0 ;; esac
    local log="results/logs/run_v5_$1.log"
    # Erst loeschen, dann neu anlegen. Auf dem WSL-/drvfs-Mount kann ein Verzeichniseintrag
    # nach einem `mv` in einem Zustand landen, in dem er sich nicht mehr zum Schreiben
    # oeffnen laesst ("No such file or directory", `ls` zeigt `-????????`). Das hat hier
    # einen msglm-Lauf sofort abgebrochen.
    rm -f "$log"; : > "$log" || { echo "!! kann $log nicht schreiben" >&2; return 1; }
    echo "=== $(date +%H:%M) $1 ($2) ==="
    $CR "$2" >> "$log" 2>&1
    echo "    exit=$?  $(date +%H:%M)"
}

run_step imageglm drift_glm.analysis.imageglm
run_step msglm    drift_glm.analysis.msglm
run_step report   drift_glm.reports.imagespace_report
echo "=== fertig $(date +%H:%M) ==="
