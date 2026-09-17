#!/usr/bin/env bash
# Version-6-Kette: alle Analyse- und Report-Schritte nacheinander in einer Shell.
#
# Sequenziell, weil der Speicher der Engpass ist: ein dct:0.02-AR-IRLS-Kanalfit belegt
# rund 2,9 GB, parallele Laeufe werden vom OOM-Killer beendet. msglm (~19 h) kommt zuletzt.
#
# Schritte (ohne Argumente alle, sonst nur die genannten), ungefaehre Laufzeiten:
#   demo       Demo-Abbildungen inkl. Scalp GT vs. Schaetzung (Abb. 1-5, 24)
#   residuals  Residual-Analyse + HRF-Formvergleich nn22 (Abb. 27/28)
#   mshrf      Multisubject: HRF je Driftfamilie + Fit-Metriken (Abb. 29/30)
#   sweep      Hauptstudie v4: Bildraum-GT + R^2/Residual-Metriken (~7,5 h)
#   sweeprep   Sweep-Auswertung (Abb. 6-10, 25/26, tables.md)
#   flex       flexible Recovery-Basis (~1 h)
#   detection  Detektions-Analyse (~1,5 h)
#   imageglm   Bildraum-Rangliste (~1-3 h)
#   report     Bildraum-/Multisubject-Report (Abb. 19-23)
#   msglm      Multisubject-GLM-Restzellen (Resume; dct:0.02 x AR-IRLS, ~19 h)
#   tables     Abb. 21-23 aktualisieren, nachdem msglm vollstaendig ist
#
# Aufruf:            ./run_v6.sh [schritt ...]
# Fortschritt live:  tail -f results/logs/*_progress.txt
# Logs:              results/logs/run_v6_*.log
set -u
cd "$(dirname "$0")"

mkdir -p results/logs
# Lock-Datei mit der PID, damit nicht zwei Ketten gleichzeitig laufen.
LOCK="results/logs/.run_v6.lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "Es laeuft schon eine Kette (PID $(cat "$LOCK")). Abbruch." >&2
    exit 1
fi
echo $$ > "$LOCK"
# Der EXIT-Trap entfernt nur den Lock; bei INT/TERM werden zusaetzlich die Kindprozesse beendet.
cleanup() { trap - EXIT; rm -f "$LOCK"; }
on_signal() { trap - EXIT INT TERM; rm -f "$LOCK"; pkill -TERM -P $$ 2>/dev/null; exit 143; }
trap cleanup EXIT
trap on_signal INT TERM

source ~/anaconda3/etc/profile.d/conda.sh
CR="conda run --no-capture-output -n cedalion python -u -m"

STEPS="${*:-demo residuals mshrf sweep sweeprep flex detection imageglm report msglm tables}"

run_step() {                       # $1 = Name, $2 = Modul, $3.. = Argumente
    case " $STEPS " in *" $1 "*) ;; *) return 0 ;; esac
    local name="$1" mod="$2"; shift 2
    local log="results/logs/run_v6_$name.log"
    rm -f "$log"; : > "$log" || { echo "!! kann $log nicht schreiben" >&2; return 1; }
    echo "=== $(date +%H:%M) $name ($mod $*) ==="
    $CR "$mod" "$@" >> "$log" 2>&1
    local rc=$?
    echo "    exit=$rc  $(date +%H:%M)"
    return $rc
}

run_step demo      drift_glm.reports.demo_figures
run_step residuals drift_glm.analysis.residuals
run_step mshrf     drift_glm.analysis.mshrf
run_step sweep     drift_glm.analysis.sweep v4
run_step sweeprep  drift_glm.reports.sweep_report
run_step flex      drift_glm.analysis.flex_basis
run_step detection drift_glm.analysis.detection
run_step imageglm  drift_glm.analysis.imageglm
run_step report    drift_glm.reports.imagespace_report
run_step msglm     drift_glm.analysis.msglm
run_step tables    drift_glm.reports.imagespace_report tables
echo "=== fertig $(date +%H:%M) ==="
