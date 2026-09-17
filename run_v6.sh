#!/usr/bin/env bash
# Version-6-Kette: alle Analyse- und Report-Schritte nacheinander in einer Shell.
#
# Sequenziell, weil der Speicher der Engpass ist: ein dct:0.02-AR-IRLS-Kanalfit belegt
# rund 2,9 GB, parallele Laeufe werden vom OOM-Killer beendet.
#
# Schritte (ohne Argumente alle, sonst nur die genannten), ungefaehre Laufzeiten:
#   demo       Demo-Abbildungen inkl. Scalp GT vs. Schaetzung (Abb. 1-5, 24)
#   residuals  Residual-Analyse + HRF-Formvergleich nn22 (Abb. 27/28)
#   mshrf      Multisubject: HRF je Driftfamilie + Fit-Metriken (Abb. 29/30)
#   sweep      Hauptstudie v4 (~7,5 h); archiviert vorhandene sweep_*-Ergebnisse (--replace)
#   sweepxy    Kontrollarm butterxy:0.01 (120 Fits, ~25 min) + Merge in die Haupttabelle
#   sweeprep   Sweep-Auswertung (Abb. 6-10, 25/26, tables.md)
#   flex       flexible Recovery-Basis (~1 h)
#   detection  Detektions-Analyse (~1,5 h)
#   imageglm   Bildraum-Rangliste (~1-3 h)
#   report     Bildraum-/Multisubject-Report (Abb. 19-23)
#   msglm      Multisubject-GLM, 81-Zellen-Raster (OLS voll, AR-IRLS none/poly:3/butter)
#   realglm    Khan-Datensatz (14 Familien, ~3,6 h) und Ergaenzungsarme (lowpass, butterxy)
#   tables     Abb. 21-23 aktualisieren, nachdem msglm vollstaendig ist
#
# Fehlerverhalten: Ein fehlgeschlagener Schritt beendet die Kette (Exit 1); nachfolgende
# Berichte laufen dann nicht auf veralteten Ergebnisdateien. Der Status jedes Schritts
# steht in results/logs/run_v6_status.json.
#
# Aufruf:            ./run_v6.sh [schritt ...]
# Fortschritt live:  tail -f results/logs/*_progress.txt
# Logs:              results/logs/run_v6_*.log
set -u -o pipefail
cd "$(dirname "$0")"

mkdir -p results/logs
# Atomare Sperre ueber flock (kein Check-then-Write-Fenster wie bei einer PID-Datei).
LOCK="results/logs/.run_v6.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "Es laeuft schon eine Kette (Sperre $LOCK). Abbruch." >&2
    exit 1
fi
echo $$ >&9
on_signal() { trap - INT TERM; pkill -TERM -P $$ 2>/dev/null; write_status "aborted"; exit 143; }
trap on_signal INT TERM

source ~/anaconda3/etc/profile.d/conda.sh
CR="conda run --no-capture-output -n cedalion python -u -m"

STEPS="${*:-demo residuals mshrf sweep sweepxy sweeprep flex detection imageglm report msglm realglm tables}"
STATUS="results/logs/run_v6_status.json"
declare -a DONE_STEPS=()

write_status() {                   # $1 = Gesamtstatus
    {
        echo "{"
        echo "  \"chain\": \"run_v6\", \"finished\": \"$(date -Iseconds)\", \"overall\": \"$1\","
        echo "  \"git_commit\": \"$(git rev-parse HEAD 2>/dev/null)\","
        echo "  \"steps\": ["
        local sep=""
        for s in "${DONE_STEPS[@]}"; do echo "    $sep$s"; sep=","; done
        echo "  ]"
        echo "}"
    } > "$STATUS"
}

run_step() {                       # $1 = Name, $2 = Modul, $3.. = Argumente
    case " $STEPS " in *" $1 "*) ;; *) return 0 ;; esac
    local name="$1" mod="$2"; shift 2
    local log="results/logs/run_v6_$name.log"
    rm -f "$log"; : > "$log" || { echo "!! kann $log nicht schreiben" >&2; return 1; }
    local t0; t0=$(date +%s)
    echo "=== $(date +%H:%M) $name ($mod $*) ==="
    $CR "$mod" "$@" >> "$log" 2>&1
    local rc=$?
    echo "    exit=$rc  $(date +%H:%M)  ($(( $(date +%s) - t0 )) s)"
    DONE_STEPS+=("{\"step\": \"$name\", \"module\": \"$mod\", \"args\": \"$*\", \"exit\": $rc, \"seconds\": $(( $(date +%s) - t0 ))}")
    if [ "$rc" -ne 0 ]; then
        echo "!! Schritt $name fehlgeschlagen (exit=$rc), Kette abgebrochen. Log: $log" >&2
        write_status "failed"
        exit 1
    fi
    return 0
}

run_step demo      drift_glm.reports.demo_figures
run_step residuals drift_glm.analysis.residuals
run_step mshrf     drift_glm.analysis.mshrf
run_step sweep     drift_glm.analysis.sweep v4 --replace
run_step sweepxy   drift_glm.analysis.sweep butterxy --replace
run_step sweepxy   drift_glm.analysis.sweep merge --addition-prefix sweep_butterxy
run_step sweeprep  drift_glm.reports.sweep_report
run_step flex      drift_glm.analysis.flex_basis
run_step detection drift_glm.analysis.detection
run_step imageglm  drift_glm.analysis.imageglm
run_step report    drift_glm.reports.imagespace_report
run_step msglm     drift_glm.analysis.msglm full --no-resume
run_step msglm     drift_glm.analysis.msglm butterxy --no-resume
run_step realglm   drift_glm.analysis.realglm full
run_step realglm   drift_glm.analysis.realglm supplement
run_step realglm   drift_glm.analysis.realglm merge
run_step realglm   drift_glm.analysis.khan_lateralisation
run_step tables    drift_glm.reports.imagespace_report tables
write_status "ok"
echo "=== fertig $(date +%H:%M) ==="
