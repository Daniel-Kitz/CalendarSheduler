import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from scheduler.schedule import (
    build_schedule,
    load_config,
    load_distances_from_csv,
    load_repeating_from_csv,
    load_tasks_from_csv,
)
from scheduler.ics import export_schedule_to_ics


def _banner() -> None:
    print("===========================================")
    print("         Calendar Scheduler (CP-SAT)        ")
    print("===========================================")
    print()


def _format_schedule_table(schedule, tasks):
    headers = ["ID", "Title", "Location", "Start", "End", "Due", "Priority", "Hard Due"]
    rows = []
    fmt_dt = lambda dt: dt.strftime("%Y-%m-%d %H:%M")
    for task in sorted(tasks, key=lambda t: schedule.get(t.id, datetime.max)):
        start_time = schedule.get(task.id)
        end_time = (
            start_time + timedelta(minutes=task.duration_min) if start_time else None
        )
        rows.append(
            [
                str(task.id),
                task.title,
                task.location or "-",
                fmt_dt(start_time) if start_time else "-",
                fmt_dt(end_time) if end_time else "-",
                fmt_dt(task.due) if task.due else "-",
                f"P{task.priority}",
                "Yes" if task.hard_due else "No",
            ]
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row_vals):
        return " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row_vals))

    lines = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CP-SAT calendar scheduler with table output and ICS export"
    )
    parser.add_argument("--tasks", type=Path, default=Path("scheduler/data/tasks.csv"))
    parser.add_argument(
        "--repeating", type=Path, default=Path("scheduler/data/repeating.csv")
    )
    parser.add_argument(
        "--distances", type=Path, default=Path("scheduler/data/distances.csv")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("scheduler/data/config.json")
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help='Optional start datetime "YYYY-MM-DD HH:MM" (defaults to now)',
    )
    parser.add_argument(
        "--ics-out",
        type=Path,
        default=Path("scheduler/schedule.ics"),
        help="Path to write iCal (.ics) file",
    )
    args = parser.parse_args(argv)

    _banner()

    start_dt: Optional[datetime] = (
        datetime.strptime(args.start, "%Y-%m-%d %H:%M") if args.start else None
    )

    try:
        tasks = load_tasks_from_csv(args.tasks)
        repeating = load_repeating_from_csv(args.repeating)
        distances = load_distances_from_csv(args.distances)
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"[error] Missing file: {exc.filename}")
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[error] Failed to load inputs: {exc}")
        return 1

    print("Inputs:")
    print(f"- Tasks:      {args.tasks}")
    print(f"- Repeating:  {args.repeating}")
    print(f"- Distances:  {args.distances}")
    print(f"- Config:     {args.config}")
    print(f"- Start time: {start_dt or 'now'}")
    print()

    try:
        schedule = build_schedule(
            tasks=tasks,
            config=config,
            repeating=repeating,
            distances=distances,
            start=start_dt,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[error] Failed to build schedule: {exc}")
        return 1

    scheduled_count = sum(1 for t in tasks if t.id in schedule)
    print(f"Scheduled {scheduled_count}/{len(tasks)} tasks")
    print()
    print(_format_schedule_table(schedule, tasks))
    print()

    if args.ics_out:
        try:
            export_schedule_to_ics(schedule, tasks, config, args.ics_out)
            print(f"ICS written to {args.ics_out.resolve()}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[error] Failed to write ICS: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
