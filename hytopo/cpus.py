from __future__ import annotations

"""Platform-aware CPU detection for frame-level multiprocessing.

``os.cpu_count()`` reports the number of processors of the machine, not the
number this process may actually use: scheduler affinity (``taskset``, Windows
processor affinity), container CPU quotas (cgroup v1/v2) and batch-system
allocations (SLURM) all restrict it.  Starting one worker per reported CPU
oversubscribes a shared node and slows the run down, so the default worker
count is derived from the smallest of the available limits.
"""

import os
from typing import Optional

_ENV_OVERRIDES = (
    # explicit hytopo override first, then the usual batch-system variables
    "HYTOPO_NPROCS",
    "SLURM_CPUS_PER_TASK",
    "PBS_NP",
    "NSLOTS",
    "LSB_DJOB_NUMPROC",
)


def _env_limit() -> Optional[int]:
    for name in _ENV_OVERRIDES:
        value = os.environ.get(name)

        if not value:
            continue

        try:
            parsed = int(float(value))
        except ValueError:
            continue

        if parsed > 0:
            return parsed

    return None


def _affinity_limit() -> Optional[int]:
    sched_getaffinity = getattr(os, "sched_getaffinity", None)

    if sched_getaffinity is not None:
        try:
            return len(sched_getaffinity(0))
        except (OSError, NotImplementedError):
            pass

    # Windows: psutil reports the processor affinity mask when installed.
    try:
        import psutil  # type: ignore

        return len(psutil.Process().cpu_affinity())
    except Exception:
        return None


def _cgroup_limit() -> Optional[int]:
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8") as handle:
            fields = handle.read().split()

        if len(fields) == 2 and fields[0] != "max":
            quota, period = int(fields[0]), int(fields[1])

            if quota > 0 and period > 0:
                return max(1, quota // period)
    except (OSError, ValueError):
        pass

    # cgroup v1
    try:
        with open(
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r", encoding="utf-8"
        ) as handle:
            quota = int(handle.read().strip())

        with open(
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r", encoding="utf-8"
        ) as handle:
            period = int(handle.read().strip())

        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass

    return None


def usable_cpu_count() -> int:
    """Number of CPUs this process can actually run on."""
    limits = [limit for limit in (_env_limit(), _affinity_limit(), _cgroup_limit()) if limit]

    reported = os.cpu_count() or 1

    if limits:
        return max(1, min(limits))

    return max(1, reported)


def default_worker_count(n_items: Optional[int] = None) -> int:
    """Default process count, never more workers than work items."""
    workers = usable_cpu_count()

    if n_items is not None:
        workers = min(workers, max(1, int(n_items)))

    return max(1, workers)
