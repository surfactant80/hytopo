from __future__ import annotations

"""Minimal progress reporter (no external dependency)."""

import sys
import time
from typing import Optional, TextIO


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))

    if seconds < 60:
        return f"{seconds:4.1f}s"

    minutes, remainder = divmod(int(seconds), 60)

    if minutes < 60:
        return f"{minutes:2d}m{remainder:02d}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours:2d}h{minutes:02d}m"


class ProgressReporter:
    """Single-line progress bar with rate and ETA."""

    def __init__(
        self,
        total: int,
        enabled: bool = True,
        stream: Optional[TextIO] = None,
        min_interval: float = 0.25,
        width: int = 24,
    ) -> None:
        self.total = max(1, int(total))
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.min_interval = min_interval
        self.width = width
        self.done = 0
        self.start = time.perf_counter()
        self._last = 0.0
        self._printed = False

    def update(self, step: int = 1) -> None:
        self.done += step

        if not self.enabled:
            return

        now = time.perf_counter()

        if now - self._last < self.min_interval and self.done < self.total:
            return

        self._last = now
        elapsed = now - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.done
        eta = remaining / rate if rate > 0 else 0.0
        ratio = min(1.0, self.done / self.total)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)

        print(
            f"  [{bar}] {self.done}/{self.total} ({ratio * 100:5.1f}%) "
            f"{rate:5.2f} frame/s  ETA {_format_duration(eta)}",
            end="\r",
            file=self.stream,
            flush=True,
        )
        self._printed = True

    def finish(self) -> None:
        if not self.enabled:
            return

        if self.done < self.total:
            self._last = 0.0
            self.update(0)

        if self._printed:
            elapsed = time.perf_counter() - self.start
            rate = self.done / elapsed if elapsed > 0 else 0.0
            print(
                f"  done: {self.done} frames in {_format_duration(elapsed)} "
                f"({rate:.2f} frame/s)",
                file=self.stream,
                flush=True,
            )
            self._printed = False
