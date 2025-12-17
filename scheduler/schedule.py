import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class Task:
    id: int
    title: str
    duration_min: int
    due: datetime
    priority: int
    bundle: str
    depends_on: list[int]
    hard_due: bool
    location: str


@dataclass(frozen=True)
class RepeatingTask:
    task_id: int
    duration_min: int
    start: datetime
    end: datetime
    frequency: str
    interval: int


@dataclass(frozen=True)
class Distance:
    location: str
    location_2: str
    duration_min: int


def _time_of_day_to_minutes(value: str) -> int:
    """Convert HH:MM to minutes since midnight."""
    parsed = datetime.strptime(value, "%H:%M").time()
    return parsed.hour * 60 + parsed.minute


def _travel_lookup(distances: Iterable[Distance]) -> Dict[Tuple[str, str], int]:
    """Return symmetric travel matrix; missing entries default to 0."""
    table: Dict[Tuple[str, str], int] = {}
    for item in distances:
        table[(item.location, item.location_2)] = item.duration_min
        table[(item.location_2, item.location)] = item.duration_min
    return table


def _generate_repeating_occurrences(
    repeating: Iterable[RepeatingTask],
    start_date: datetime,
    horizon_days: int,
) -> List[Tuple[int, datetime, datetime]]:
    """Expand repeating tasks into fixed occurrences within the horizon."""
    occurrences: List[Tuple[int, datetime, datetime]] = []
    horizon_end = start_date + timedelta(days=horizon_days)
    for item in repeating:
        current = item.start
        end = item.end
        delta = timedelta(days=item.interval)
        if delta <= timedelta(0):
            raise ValueError("Repeating interval must be positive")
        if item.frequency == "EVERY_OTHER_DAY":
            delta = timedelta(days=2 * item.interval)
        elif item.frequency == "WEEKLY":
            delta = timedelta(days=7 * item.interval)
        elif item.frequency == "MONTHLY":
            # Month length differs; approximate with 30-day blocks to stay simple.
            delta = timedelta(days=30 * item.interval)
        while current + delta <= start_date:
            current += delta
        while current < horizon_end and current < end:
            occ_end = current + timedelta(minutes=item.duration_min)
            occurrences.append((item.task_id, current, occ_end))
            current += delta
    return occurrences


# --- CSV helpers ----------------------------------------------------------- #


def _parse_datetime(value: str) -> datetime:
    return datetime.strptime(value.strip().strip('"'), "%Y-%m-%d %H:%M")


def load_tasks_from_csv(path: Path) -> List[Task]:
    tasks: List[Task] = []
    with path.open() as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            depends_raw = row.get("depends_on", "") or ""
            depends_list = [
                int(x) for x in depends_raw.replace(",", " ").split() if x.strip()
            ]
            tasks.append(
                Task(
                    id=int(row["id"]),
                    title=row["title"].strip().strip('"'),
                    duration_min=int(row["duration_min"]),
                    due=_parse_datetime(row["due"]),
                    priority=int(row["priority"]),
                    bundle=(row.get("bundle") or "").strip('" '),
                    depends_on=depends_list,
                    hard_due=str(row["hard_due"]).strip().lower() == "true",
                    location=(row.get("location") or "").strip('" '),
                )
            )
    return tasks


def load_repeating_from_csv(path: Path) -> List[RepeatingTask]:
    repeating: List[RepeatingTask] = []
    with path.open() as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            repeating.append(
                RepeatingTask(
                    task_id=int(row["task_id"]),
                    duration_min=int(row["duration_min"]),
                    start=_parse_datetime(row["start"]),
                    end=_parse_datetime(row["end"]),
                    frequency=row["frequency"],
                    interval=int(row["interval"]),
                )
            )
    return repeating


def load_distances_from_csv(path: Path) -> List[Distance]:
    distances: List[Distance] = []
    with path.open() as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            distances.append(
                Distance(
                    location=row["location"].strip().strip('"'),
                    location_2=row["location_2"].strip().strip('"'),
                    duration_min=int(row["duration_min"]),
                )
            )
    return distances


def load_config(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def build_schedule(
    tasks: List[Task],
    config: Dict,
    repeating: Optional[List[RepeatingTask]] = None,
    distances: Optional[List[Distance]] = None,
    start: Optional[datetime] = None,
) -> Dict[int, datetime]:
    """
    Build a schedule using OR-Tools CP-SAT.

    Returns a mapping of task id -> scheduled start datetime.
    Raises a ValueError when no feasible schedule can be built.
    """

    start_date = start or datetime.now()
    repeating = repeating or []
    distances = distances or []

    horizon_days: int = config.get("horizon_days", 7)
    slot_minutes: int = config.get("slot_minutes", 15)
    allowed_hours = config.get("allowed_schedule_hours", [])
    max_work_minutes_per_day: int = int(config.get("max_work_hours_per_day", 12) * 60)

    minutes_per_day = 24 * 60
    horizon_minutes = horizon_days * minutes_per_day

    # Pre-compute travel durations.
    travel = _travel_lookup(distances)

    def travel_time(loc_a: str, loc_b: str) -> int:
        if loc_a == loc_b:
            return 0
        return travel.get((loc_a, loc_b), travel.get((loc_b, loc_a), 0))

    # Allowed starts by task.
    allowed_starts: Dict[int, List[int]] = {}
    for task in tasks:
        starts: List[int] = []
        for day in range(horizon_days):
            base = day * minutes_per_day
            for window in allowed_hours:
                day_start = base + _time_of_day_to_minutes(window["start"])
                day_end = base + _time_of_day_to_minutes(window["end"])
                latest_start = day_end - task.duration_min
                current = day_start
                while current <= latest_start:
                    starts.append(current)
                    current += slot_minutes
        allowed_starts[task.id] = sorted(starts)
        if not allowed_starts[task.id]:
            raise ValueError(
                f"No allowed start slots for task {task.id} ({task.title})"
            )

    model = cp_model.CpModel()

    start_vars: Dict[int, cp_model.IntVar] = {}
    end_vars: Dict[int, cp_model.IntVar] = {}
    day_vars: Dict[int, cp_model.IntVar] = {}
    objective_terms: List[cp_model.LinearExpr] = []

    for task in tasks:
        domain = cp_model.Domain.FromValues(allowed_starts[task.id])
        start_var = model.NewIntVarFromDomain(domain, f"start_{task.id}")
        end_var = model.NewIntVar(0, horizon_minutes, f"end_{task.id}")
        day_var = model.NewIntVar(0, horizon_days - 1, f"day_{task.id}")

        model.Add(end_var == start_var + task.duration_min)
        model.AddDivisionEquality(day_var, start_var, minutes_per_day)

        # Deadlines.
        if task.due:
            due_offset = int((task.due - start_date).total_seconds() // 60)
            if task.hard_due:
                model.Add(end_var <= due_offset)
            else:
                lateness = model.NewIntVar(0, horizon_minutes, f"late_{task.id}")
                model.Add(lateness >= end_var - due_offset)
                model.Add(lateness >= 0)
                # Weighted by priority (lower number = higher priority).
                objective_terms.append(lateness * max(1, task.priority))

        start_vars[task.id] = start_var
        end_vars[task.id] = end_var
        day_vars[task.id] = day_var

    # Non-overlap with travel time between tasks.
    task_list = tasks.copy()
    for i in range(len(task_list)):
        for j in range(i + 1, len(task_list)):
            ti = task_list[i]
            tj = task_list[j]
            before = model.NewBoolVar(f"{ti.id}_before_{tj.id}")
            travel_ij = travel_time(ti.location, tj.location)
            travel_ji = travel_time(tj.location, ti.location)
            model.Add(end_vars[ti.id] + travel_ij <= start_vars[tj.id]).OnlyEnforceIf(
                before
            )
            model.Add(end_vars[tj.id] + travel_ji <= start_vars[ti.id]).OnlyEnforceIf(
                before.Not()
            )

    # Dependencies.
    task_map = {task.id: task for task in tasks}
    for task in tasks:
        for dep_id in task.depends_on:
            if dep_id not in task_map:
                continue
            model.Add(start_vars[task.id] >= end_vars[dep_id])

    # Daily work-hour limits.
    for day in range(horizon_days):
        day_flags: List[Tuple[cp_model.BoolVar, int]] = []
        for task in tasks:
            flag = model.NewBoolVar(f"day_{day}_task_{task.id}")
            model.Add(day_vars[task.id] == day).OnlyEnforceIf(flag)
            model.Add(day_vars[task.id] != day).OnlyEnforceIf(flag.Not())
            day_flags.append((flag, task.duration_min))
        model.Add(
            sum(flag * duration for flag, duration in day_flags)
            <= max_work_minutes_per_day
        )

    # Block repeating tasks as fixed intervals.
    occurrences = _generate_repeating_occurrences(repeating, start_date, horizon_days)
    for occ_id, occ_start_dt, occ_end_dt in occurrences:
        occ_start = int((occ_start_dt - start_date).total_seconds() // 60)
        occ_end = int((occ_end_dt - start_date).total_seconds() // 60)
        occ_task = task_map.get(occ_id)
        occ_location = occ_task.location if occ_task else ""
        for task in tasks:
            before = model.NewBoolVar(f"occ_{occ_id}_before_task_{task.id}")
            travel_1 = travel_time(task.location, occ_location)
            travel_2 = travel_time(occ_location, task.location)
            model.Add(occ_end + travel_2 <= start_vars[task.id]).OnlyEnforceIf(before)
            model.Add(end_vars[task.id] + travel_1 <= occ_start).OnlyEnforceIf(
                before.Not()
            )

    # Objective: minimize total lateness and prefer earlier completion.
    total_end = model.NewIntVar(0, horizon_minutes, "total_end")
    model.AddMaxEquality(total_end, list(end_vars.values()))
    objective_terms.append(total_end)
    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    result = solver.Solve(model)

    if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError("No feasible schedule found with given constraints")

    schedule: Dict[int, datetime] = {}
    for task in tasks:
        start_minute = solver.Value(start_vars[task.id])
        schedule[task.id] = start_date + timedelta(minutes=start_minute)

    return schedule
