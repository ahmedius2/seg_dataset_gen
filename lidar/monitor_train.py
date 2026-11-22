#!/usr/bin/env python3
"""Snapshot one or more gnd_seg_dnn training sessions.

Discovers live ``gnd_seg_dnn.train`` processes, maps each to the run directory
it is writing, and prints status, progress and the recent epoch table. Finished
runs are shown too, so the same command works before and after a run ends.

Usage (normally via ``./monitor_train.sh``):

    python monitor_train.py --once            # every session, then exit
    python monitor_train.py                   # refresh continuously
    python monitor_train.py --interval 2
    python monitor_train.py my_run_dir            # restrict to one run
    python monitor_train.py --log             # follow a log file
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LIDAR_DIR = Path(__file__).resolve().parent
LOG_DIR = Path("/tmp/opencode")
TRAIN_MARKER = "gnd_seg_dnn.train"


# --------------------------------------------------------------------------- #
#  Process discovery
# --------------------------------------------------------------------------- #

def _read(path, default=None):
    try:
        return Path(path).read_text()
    except OSError:
        return default


def _cmdline(pid):
    raw = _read(f"/proc/{pid}/cmdline", "")
    return [a for a in raw.split("\0") if a]


def _pss_mb(pid):
    """Proportional set size, avoiding double-counted shared pages."""
    text = _read(f"/proc/{pid}/smaps_rollup")
    if text is None:
        return 0
    for line in text.splitlines():
        if line.startswith("Pss:"):
            return int(line.split()[1]) // 1024
    return 0


def _children(pid):
    text = _read(f"/proc/{pid}/task/{pid}/children", "") or ""
    return [p for p in text.split() if p]


def _tree_pss_mb(pid):
    total, stack, seen = 0, [str(pid)], set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        total += _pss_mb(current)
        stack.extend(_children(current))
    return total


def _uptime(pid):
    out = subprocess.run(["ps", "-o", "etime=", "-p", str(pid)],
                         capture_output=True, text=True).stdout.strip()
    return out or "?"


def _stdout_log(pid):
    """Exact log path via the process's own fd 1, if it is a regular file."""
    try:
        target = os.readlink(f"/proc/{pid}/fd/1")
    except OSError:
        return None
    if target.startswith("/") and target.endswith(".log") and os.path.exists(target):
        return Path(target)
    return None


def running_sessions():
    """[(pid, config_path, cwd)] for every live trainer."""
    sessions = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        argv = _cmdline(entry.name)
        if not any(TRAIN_MARKER in a for a in argv) or "--config" not in argv:
            continue
        try:
            config = argv[argv.index("--config") + 1]
        except IndexError:
            continue
        try:
            cwd = Path(os.readlink(f"/proc/{entry.name}/cwd"))
        except OSError:
            cwd = LIDAR_DIR
        sessions.append((int(entry.name), config, cwd))
    return sorted(sessions)


# --------------------------------------------------------------------------- #
#  Run directory / config
# --------------------------------------------------------------------------- #

def read_config(path):
    """Parse a training YAML, falling back to a few regexes without PyYAML."""
    text = _read(path)
    if text is None:
        return {}
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except Exception:
        cfg = {}
        for key in ("out_dir", "model_variant", "loss_type", "dataset"):
            m = re.search(rf"^{key}:\s*(\S+)", text, re.M)
            if m:
                cfg[key] = m.group(1)
        for key in ("cell_size", "batch_size", "num_workers", "epochs"):
            m = re.search(rf"^{key}:\s*([0-9.]+)", text, re.M)
            if m:
                cfg[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
        return cfg


def resolve_run_dir(config_path, cwd):
    """Config's out_dir -> the actual (possibly _NNN suffixed) run directory."""
    cfg = read_config(config_path if Path(config_path).is_absolute()
                      else cwd / config_path)
    out_dir = str(cfg.get("out_dir") or "").strip()
    if not out_dir:
        return None
    base = Path(out_dir)
    if not base.is_absolute():
        base = cwd / base
    pattern = re.compile(rf"^{re.escape(base.name)}(_\d{{3}})?$")
    candidates = [p for p in base.parent.glob(f"{base.name}*")
                  if p.is_dir() and pattern.match(p.name)]
    if not candidates:
        return None
    # The trainer appends _NNN for an existing out_dir, so the newest wins.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def all_run_dirs():
    out = []
    for hist in LIDAR_DIR.glob("*/config.yaml"):
        out.append(hist.parent)
    return [p for p in out if (p / "history.json").exists() or (p / "splits.txt").exists()]


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #

def load_history(run_dir):
    try:
        return json.loads((run_dir / "history.json").read_text())
    except Exception:
        return []


def geometry(cfg):
    try:
        cell = float(cfg["cell_size"])
        w = round((float(cfg["x_max"]) - float(cfg["x_min"])) / cell)
        h = round((float(cfg["y_max"]) - float(cfg["y_min"])) / cell)
        return f"cell {cell:g} m | grid {int(w)}x{int(h)}"
    except Exception:
        return ""


def log_tail(log, lines=400):
    if not log or not Path(log).exists():
        return []
    out = subprocess.run(["tail", "-n", str(lines), str(log)],
                         capture_output=True, text=True).stdout
    return out.splitlines()


def progress_lines(log):
    """Latest step line and latest finished-epoch line from a log."""
    tail = log_tail(log)
    step = next((l.strip() for l in reversed(tail) if " step " in l), None)
    epoch = next((l.strip() for l in reversed(tail)
                  if re.match(r"^epoch \d+ \(", l)), None)
    return step, epoch


def report(run_dir, pid=None, log=None, table_rows=12):
    cfg = read_config(run_dir / "config.yaml")
    rows = load_history(run_dir)
    name = run_dir.name

    geo = geometry(cfg)
    if cfg:
        print(f"run      {name}")
        print(f"config   {geo} | {cfg.get('model_variant', '?')} "
              f"| bs {cfg.get('batch_size', '?')} "
              f"| workers {cfg.get('num_workers', '?')} "
              f"| loss {cfg.get('loss_type', '?')}")
    else:
        print(f"run      {name}")

    if pid:
        print(f"status   RUNNING  pid {pid}  uptime {_uptime(pid)}  "
              f"RAM {_tree_pss_mb(pid)} MB")
    else:
        last = rows[-1]["epoch"] if rows else None
        print(f"status   not running"
              + (f"  (last epoch {last})" if last is not None else ""))

    if log and Path(log).exists():
        age = int(time.time() - Path(log).stat().st_mtime)
        print(f"log      {log}  (updated {age} s ago)")
    step, epoch_line = progress_lines(log)
    if step:
        print(f"now      {step}")
    if epoch_line:
        print(f"last     {epoch_line}")

    if not rows:
        print("         no completed epochs yet")
        return

    budget = cfg.get("epochs", "?")
    best = max(rows, key=lambda e: e["val"]["iou"])
    print()
    print(f"{'ep':>3} {'tr_iou':>7} {'val_iou':>8} {'val_prec':>9} "
          f"{'val_rec':>8} {'val_loss':>9}")
    print("-" * 50)
    for e in rows[-table_rows:]:
        t, v = e["train"], e["val"]
        print(f"{e['epoch']:>3} {t['iou']:>7.4f} {v['iou']:>8.4f} "
              f"{v['precision']:>9.4f} {v['recall']:>8.4f} "
              f"{v['loss']:>9.4f}")
    print("-" * 50)
    print(f"best     epoch {best['epoch']}  val_iou {best['val']['iou']:.4f}"
          f"  (last epoch {rows[-1]['epoch']}, budget {budget})")


def overview(entries):
    """One line per session, for the top of the output."""
    print(f"{'run':<26} {'status':<9} {'cell':>5} {'ep':>4} {'best val_iou':>13}")
    print("-" * 62)
    for run_dir, pid, _log in entries:
        cfg = read_config(run_dir / "config.yaml")
        rows = load_history(run_dir)
        best = max(rows, key=lambda e: e["val"]["iou"]) if rows else None
        cell = cfg.get("cell_size", "?")
        ep = rows[-1]["epoch"] if rows else "-"
        val = f"{best['val']['iou']:.4f}" if best else "-"
        status = "RUNNING" if pid else "stopped"
        print(f"{run_dir.name:<26} {status:<9} {str(cell):>5} {str(ep):>4} {val:>13}")
    print()


def collect(only=None):
    """Pair every run directory with its live pid (None when not running)."""
    sessions = running_sessions()
    by_run = {}
    for pid, config, cwd in sessions:
        run_dir = resolve_run_dir(config, cwd)
        if run_dir is not None:
            by_run[run_dir] = (pid, _stdout_log(pid))

    entries = []
    for run_dir in all_run_dirs():
        pid, log = by_run.get(run_dir, (None, None))
        if log is None:
            log = guess_log(run_dir, running=pid is not None)
        entries.append((run_dir, pid, log))

    # Live sessions whose run directory could not be resolved still get shown.
    known = {e[0] for e in entries}
    for pid, config, cwd in sessions:
        run_dir = resolve_run_dir(config, cwd)
        if run_dir is not None and run_dir not in known:
            entries.append((run_dir, pid, _stdout_log(pid)))

    def key(entry):
        run_dir, pid, _ = entry
        return (pid is None, -run_dir.stat().st_mtime)

    entries.sort(key=key)
    if only:
        wanted = only.rstrip("/")
        entries = [e for e in entries
                   if e[0].name == wanted or str(e[0]) == wanted]
    return entries


def guess_log(run_dir, running):
    """Best-effort log for a run without a live fd (finished sessions)."""
    logs = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    for log in logs[:12]:
        text = _read(log, "") or ""
        if run_dir.name in text:
            return log
        cfg = read_config(run_dir / "config.yaml")
        out_dir = str(cfg.get("out_dir") or "")
        if out_dir and out_dir in text:
            return log
    return logs[0] if (running and logs) else None


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("run", nargs="?", default=None,
                        help="restrict output to one run directory")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true",
                        help="print one snapshot of every session and exit")
    parser.add_argument("--log", action="store_true",
                        help="follow a training log instead")
    parser.add_argument("--rows", type=int, default=12,
                        help="epoch table rows per run (default 12)")
    parser.add_argument("--no-overview", action="store_true")
    args = parser.parse_args()

    if args.log:
        logs = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime,
                      reverse=True)
        if not logs:
            sys.exit(f"no training log in {LOG_DIR}")
        print(f"following {logs[0]}  (Ctrl-C to stop)")
        os.execvp("tail", ["tail", "-f", str(logs[0])])

    if args.once:
        entries = collect(args.run)
        if not entries:
            sys.exit(f"no training runs found in {LIDAR_DIR}")
        if not args.no_overview and len(entries) > 1:
            overview(entries)
        for index, (run_dir, pid, log) in enumerate(entries):
            if index:
                print()
            report(run_dir, pid, log, args.rows)
        return

    try:
        while True:
            entries = collect(args.run)
            print("\033[2J\033[H", end="")
            print(time.strftime("%H:%M:%S"))
            if not entries:
                print(f"no training runs found in {LIDAR_DIR}")
            else:
                if not args.no_overview and len(entries) > 1:
                    overview(entries)
                for index, (run_dir, pid, log) in enumerate(entries):
                    if index:
                        print()
                    report(run_dir, pid, log, args.rows)
            print(f"\n(refresh {args.interval:g}s, Ctrl-C to stop)")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
