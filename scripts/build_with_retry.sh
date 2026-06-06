#!/bin/bash
# Resilient 3-subset DB build: per-subset resume + retry + macOS notifications.
# Runs detached (nohup) and caffeinate-wrapped so it survives idle-sleep and
# harness background-task limits. Watch it with:  tail -f runs/logs/build_status.txt
set -u

ROOT=/Users/arupanandaswain/PycharmProjects/trainDronisight
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
YDB=/Volumes/dronisight/yolo_train_db
CDB=/Volumes/dronisight/RF_DETR_Faster_RCNN_train_db
LOG="$ROOT/runs/logs/build_runner.log"
STATUS="$ROOT/runs/logs/build_status.txt"
SUBSETS=(pole component_above_1000 component_below_1000)
MAX_TRIES=30
mkdir -p "$ROOT/runs/logs"

note() {  # title, message -> log + status file + macOS notification (with sound)
  local title="$1" msg="$2" ts
  ts="$(date '+%F %T')"
  echo "$ts | $msg" | tee -a "$LOG"
  echo "$ts | $msg" > "$STATUS"
  osascript -e "display notification \"$msg\" with title \"$title\" sound name \"Glass\"" 2>/dev/null || true
}

note "Dronisight build" "runner started (pid $$)"

for sub in "${SUBSETS[@]}"; do
  # per-subset resume: a finished subset has both a manifest and dataset_meta.json
  if [ -f "$YDB/$sub/dataset_meta.json" ] && [ -f "$YDB/$sub/manifest.csv" ]; then
    note "Dronisight build" "$sub already complete - skipping"
    continue
  fi
  try=0
  while :; do
    try=$((try + 1))
    if [ "$try" -gt "$MAX_TRIES" ]; then
      note "Dronisight build" "GAVE UP on $sub after $MAX_TRIES tries - see build_runner.log"
      exit 1
    fi
    if ! mount | grep -qi dronisight; then
      note "Dronisight build" "SSD not mounted - waiting 30s ($sub try $try)"
      sleep 30; continue
    fi
    rm -rf "$YDB/$sub" "$CDB/$sub"            # clear any partial run for this subset
    note "Dronisight build" "building $sub (try $try)..."
    caffeinate -i -m "$PY" -u -m data_prep.build_dataset --subset "$sub" >> "$LOG" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$YDB/$sub/dataset_meta.json" ]; then
      note "Dronisight build" "$sub done OK (try $try)"
      break
    fi
    note "Dronisight build" "$sub STOPPED (rc=$rc) - retrying in 15s"
    sleep 15
  done
done

find "$YDB" -name '*.cache' -delete 2>/dev/null
if "$PY" -m data_prep.verify_dataset --subset pole >> "$LOG" 2>&1 \
   && "$PY" -m data_prep.verify_dataset --subset component_above_1000 >> "$LOG" 2>&1 \
   && "$PY" -m data_prep.verify_dataset --subset component_below_1000 >> "$LOG" 2>&1; then
  note "Dronisight build" "ALL 3 SUBSETS BUILT + VERIFIED - done"
else
  note "Dronisight build" "built but VERIFY FAILED - check build_runner.log"
fi
