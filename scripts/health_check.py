#!/usr/bin/env python3
"""
scripts/health_check.py
────────────────────────
Checks the health of the ingestion process and raises alerts when something
is wrong.
Designed for cron. Exits non-zero on any alert, so cron sends mail
automatically where the system is configured for it.

    */10 * * * * cd /path/to/repo && .venv/bin/python scripts/health_check.py --quiet

Why it reads a JSON heartbeat rather than the database:
  DuckDB grants an EXCLUSIVE lock to the writing process. While ingestion
  runs, any other process — even in read_only mode — gets "Could not set lock
  on file". main.py therefore publishes its state to data/ingest_status.json
  every 60 s, and this script consumes that.

The checks are not generic: each corresponds to a failure this system has
actually suffered.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Umbrales
# ---------------------------------------------------------------------------

# The heartbeat is written every 60 s. Five minutes without an update means
# the process is hung or dead — this happened: the old shutdown path left
# zombie processes alive, holding the DuckDB lock.
HEARTBEAT_MAX_AGE_S = 300

# No new tick in 30 minutes means the stream is down. Kalshi REST polling runs
# every 30 s and Polymarket streams, so 30 minutes is generous.
LAST_TICK_MAX_AGE_S = 1800

# Minimum throughput. Below this, ingestion is alive but producing nothing.
MIN_TICKS_PER_MIN = 0.5

# Zero resolved markets after 24 h means the refresh loop is not working.
# That is exactly the defect this system had: `resolved_value` was never
# written, and no number of days of ingestion would have fixed it.
RESOLVED_GRACE_HOURS = 24

# Margin before the size cap, so the alert fires BEFORE the writer stops.
DB_SIZE_WARN_FRACTION = 0.85

# Repeated errors in the recent log. Kalshi's WebSocket returned 401 in a loop,
# backing off to 64 s: visible in the log, invisible in the metrics.
LOG_TAIL_LINES = 2000
MAX_STREAM_ERRORS = 20

# Minimum free space on the database's disk.
MIN_FREE_DISK_MB = 1024


@dataclass
class Alert:
    level: str  # "CRIT" | "WARN"
    check: str
    detail: str


def _load_status(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _process_alive(pid_file: Path) -> tuple[bool, int | None]:
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False, None
    try:
        Path(f"/proc/{pid}").stat()
        return True, pid
    except OSError:
        return False, pid


def _count_log_errors(log_file: Path) -> dict[str, int]:
    """Count known error patterns in the tail of the log."""
    if not log_file.exists():
        return {}
    try:
        tail = subprocess.run(  # noqa: S603
            ["tail", "-n", str(LOG_TAIL_LINES), str(log_file)],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}

    patterns = {
        "stream_error": r"stream error \(attempt",
        "traceback": r"^Traceback",
        "db_size_limit": r"db_size_limit_reached",
        "belief_vol_clipped": r"belief_vol_clipped",
        "half_spread_clamped": r"half_spread_clamped",
        "client_error": r"Client error [45]\d\d",
    }
    return {
        name: len(re.findall(pattern, tail, re.MULTILINE))
        for name, pattern in patterns.items()
        if re.search(pattern, tail, re.MULTILINE)
    }


def run_checks(root: Path, status_file: Path, pid_file: Path, log_file: Path) -> list[Alert]:
    alerts: list[Alert] = []

    # --- 1. Proceso vivo ---
    alive, pid = _process_alive(pid_file)
    if not alive:
        alerts.append(
            Alert("CRIT", "process", f"ingestion is not running (pid={pid or 'unknown'})")
        )

    # --- 2. Heartbeat fresco ---
    status = _load_status(status_file)
    if status is None:
        alerts.append(Alert("CRIT", "heartbeat", f"cannot read {status_file}"))
        return alerts  # with no heartbeat there is nothing else to check

    written = datetime.fromisoformat(status["written_at"])
    age = (datetime.now(tz=UTC) - written).total_seconds()
    if age > HEARTBEAT_MAX_AGE_S:
        alerts.append(
            Alert(
                "CRIT",
                "heartbeat",
                f"not updated for {age / 60:.0f} min "
                f"(max {HEARTBEAT_MAX_AGE_S / 60:.0f}) — process hung",
            )
        )

    # --- 3. The data is advancing ---
    tick_age = status.get("last_tick_age_seconds")
    if tick_age is not None and tick_age > LAST_TICK_MAX_AGE_S:
        alerts.append(
            Alert(
                "CRIT",
                "stream",
                f"last tick {tick_age / 60:.0f} min ago "
                f"(max {LAST_TICK_MAX_AGE_S / 60:.0f}) — stream down",
            )
        )

    uptime = status.get("uptime_minutes", 0)
    rate = status.get("ticks_per_min", 0)
    if uptime > 10 and rate < MIN_TICKS_PER_MIN:
        alerts.append(
            Alert(
                "WARN",
                "throughput",
                f"{rate} ticks/min (min {MIN_TICKS_PER_MIN}) after {uptime:.0f} min",
            )
        )

    # --- 4. Resolutions are being captured ---
    resolved = status.get("resolved_markets", 0)
    if uptime > RESOLVED_GRACE_HOURS * 60 and resolved == 0:
        alerts.append(
            Alert(
                "CRIT",
                "resolutions",
                f"0 resolved markets after {uptime / 60:.0f} h — "
                "the refresh loop is not capturing them. Without these there is no "
                "Brier score, no κ calibration and no μ̂ validation.",
            )
        )

    # --- 5. Database size ---
    if status.get("db_size_limit_reached"):
        alerts.append(Alert("CRIT", "db_size", "cap reached — the writer has stopped writing"))
    else:
        size_mb = status.get("db_size_mb", 0)
        limit_mb = _configured_limit_mb(root)
        if limit_mb > 0 and size_mb > limit_mb * DB_SIZE_WARN_FRACTION:
            alerts.append(
                Alert("WARN", "db_size", f"{size_mb:.0f} MB de {limit_mb:.0f} MB permitidos")
            )

    # --- 6. Espacio en disco ---
    free_mb = shutil.disk_usage(root).free / 1024 / 1024
    if free_mb < MIN_FREE_DISK_MB:
        alerts.append(Alert("CRIT", "disk", f"{free_mb:.0f} MB free remaining"))

    # --- 7. Errors in the log ---
    errors = _count_log_errors(log_file)
    if errors.get("traceback"):
        alerts.append(Alert("CRIT", "log", f"{errors['traceback']} tracebacks recientes"))
    if errors.get("stream_error", 0) > MAX_STREAM_ERRORS:
        alerts.append(
            Alert(
                "WARN",
                "log",
                f"{errors['stream_error']} stream errors — reconnecting in a loop?",
            )
        )
    if errors.get("belief_vol_clipped", 0) > 50:
        alerts.append(
            Alert(
                "WARN",
                "sigma_b",
                f"{errors['belief_vol_clipped']} recortes de σ_b — estimador fuera de rango",
            )
        )

    # --- 8. Venues silenciosas ---
    for venue, counts in (status.get("by_venue") or {}).items():
        if counts.get("markets", 0) > 0 and counts.get("features", 0) == 0:
            alerts.append(
                Alert("WARN", "features", f"{venue}: {counts['markets']} markets and 0 features")
            )

    return alerts


def _configured_limit_mb(root: Path) -> float:
    """Read MAX_DB_SIZE_MB from .env without importing the config package."""
    env = root / ".env"
    if not env.exists():
        return 2048.0
    match = re.search(r"^MAX_DB_SIZE_MB\s*=\s*([0-9.]+)", env.read_text(), re.MULTILINE)
    return float(match.group(1)) if match else 2048.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--status", default="./data/ingest_status.json")
    parser.add_argument("--pid", default="./ingest.pid")
    parser.add_argument("--log", default="./ingest.log")
    parser.add_argument("--quiet", action="store_true", help="silent when healthy")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    alerts = run_checks(root, Path(args.status), Path(args.pid), Path(args.log))

    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%SZ")

    if not alerts:
        if not args.quiet:
            status = _load_status(Path(args.status)) or {}
            print(f"[{stamp}] OK — ingestion healthy")
            print(
                f"  uptime={status.get('uptime_minutes', 0):.0f} min  "
                f"ticks={status.get('ticks_total', 0):,}  "
                f"({status.get('ticks_per_min', 0)}/min)  "
                f"resueltos={status.get('resolved_markets', 0)}  "
                f"db={status.get('db_size_mb', 0):.1f} MB"
            )
            for venue, counts in (status.get("by_venue") or {}).items():
                print(
                    f"    {venue:11} markets={counts.get('markets', 0):<5} "
                    f"ticks={counts.get('ticks', 0):<7} features={counts.get('features', 0)}"
                )
        return 0

    crit = [a for a in alerts if a.level == "CRIT"]
    print(f"[{stamp}] {len(crit)} CRITICAL, {len(alerts) - len(crit)} warnings")
    for alert in alerts:
        print(f"  [{alert.level}] {alert.check}: {alert.detail}")

    return 2 if crit else 1


if __name__ == "__main__":
    raise SystemExit(main())
