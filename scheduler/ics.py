from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

from scheduler.schedule import Task


def export_schedule_to_ics(
    schedule: Dict[int, datetime],
    tasks: List[Task],
    config: Dict,
    output_path: Path,
) -> None:
    """Write an .ics file for the computed schedule."""
    tz_name = config.get("timezone", "UTC")
    try:
        tzinfo = ZoneInfo(tz_name)
    except Exception:  # pragma: no cover - fallback
        tzinfo = timezone.utc
        tz_name = "UTC"

    now_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    lines: List[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//CalendarScheduler//EN",
        "CALSCALE:GREGORIAN",
    ]

    task_map = {t.id: t for t in tasks}
    for task_id, start_dt in schedule.items():
        task = task_map.get(task_id)
        if not task:
            continue
        start_local = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=tzinfo)
        end_local = start_local + timedelta(minutes=task.duration_min)
        start_str = start_local.strftime("%Y%m%dT%H%M%S")
        end_str = end_local.strftime("%Y%m%dT%H%M%S")
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{task.id}@calendarscheduler",
                f"DTSTAMP:{now_utc}",
                f"DTSTART;TZID={tz_name}:{start_str}",
                f"DTEND;TZID={tz_name}:{end_str}",
                f"SUMMARY:{task.title}",
                f"LOCATION:{task.location}",
                "END:VEVENT",
            ]
        )

    lines.append("END:VCALENDAR")
    output_path.write_text("\r\n".join(lines))
