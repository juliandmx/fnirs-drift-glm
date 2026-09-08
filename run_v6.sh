#!/usr/bin/env bash
# Version-6-Kette: Umsetzung der Gespraechsnotizen vom 08.09. -- SEQUENZIELL.
#
# Sequenziell aus demselben Grund wie run_v5.sh: der Speicher ist der Engpass.
# Ein einzelner dct:0.02-AR-IRLS-Kanalfit belegt 2,9 GB (gemessen 08.09.); parallele
# Laeufe wurden dreimal vom OOM-Killer beendet.
#
# Schritte (ohne Argumente alle, sonst nur die genannten):
#   mshrf      Multisubject: HRF je Driftfamilie + Fit-Metriken (Abb. 29/30)
#   residuals  Residual-Analyse + HRF-Formvergleich nn22 (Abb. 27/28)
#   demo       Demo-Abbildungen inkl. Scalp GT vs. Schaetzung (Abb. 1-5, 24)
#   report     Bildraum-/Multisubject-Report (Abb. 19-23)
#   sweep      Hauptstudie v4 NEU mit R^2/Residual-Metriken (~7,5 h)
#   sweeprep   Sweep-Auswertung (Abb. 6-10, 25/26, tables.md)
#   msglm      Multisubject-GLM-Restzellen (Resume; dct:0.02 x AR-IRLS, ~19 h!)
#   tables     Abb. 21-23 aktualisieren, nachdem msglm vollstaendig ist
#
# Fortschritt live:  tail -f results/logs/*_progress.txt
# Logs:              results/logs/run_v6_*.log
set -u
cd "$(dirname "$0")"

mkdir -p results/logs
LOCK="results/logs/.run_v6.lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "Es laeuft schon eine Kette (PID $(cat "$LOCK")). Abbruch." >&2
    exit 1
fi
# Auch gegen run_v5-Ketten sperren -- gleicher Speicher, gleiche Maschine.
LOCK5="results/logs/.run_v5.lock"
if [ -e "$LOCK5" ] && kill -0 "$(cat "$LOCK5" 2>/dev/null)" 2>/dev/null; then
    echo "Es laeuft eine run_v5-Kette (PID $(cat "$LOCK5")). Abbruch." >&2
    exit 1
fi
echo $$ > "$LOCK"
cleanup() { rm -f "$LOCK"; kill -- -$$ 2>/dev/null; }
trap cleanup EXIT INT TERM

source ~/anaconda3/etc/profile.d/conda.sh
CR="conda run --no-capture-output -n cedalion python -u -m"

STEPS="${*:-mshrf residuals demo report sweep sweeprep msglm tables}"

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

run_step mshrf     drift_glm.analysis.mshrf
run_step residuals drift_glm.analysis.residuals
run_step demo      drift_glm.reports.demo_figures
run_step report    drift_glm.reports.imagespace_report
run_step sweep     drift_glm.analysis.sweep v4
run_step sweeprep  drift_glm.reports.sweep_report
run_step msglm     drift_glm.analysis.msglm
run_step tables    drift_glm.reports.imagespace_report tables
echo "=== fertig $(date +%H:%M) ==="
