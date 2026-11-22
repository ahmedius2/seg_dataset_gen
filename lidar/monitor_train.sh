#!/usr/bin/env bash
# Watch every gnd_seg_dnn training session.
#
#   ./monitor_train.sh                      # all sessions, refresh every 5 s
#   ./monitor_train.sh --once               # all sessions, one snapshot, exit
#   ./monitor_train.sh --once my_run_dir    # one run only
#   ./monitor_train.sh --interval 2         # faster refresh
#   ./monitor_train.sh --log                # follow the newest training log
#
# Run from lidar/. The work is done by monitor_train.py next to this script.

set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done
if [[ -z "$PY" ]]; then
    echo "python3 not found on PATH" >&2
    exit 1
fi

exec "$PY" "$HERE/monitor_train.py" "$@"
