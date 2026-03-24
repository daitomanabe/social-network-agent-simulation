"""Time management for the simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time


@dataclass
class TimeManager:
    """Tracks simulation time and maps it to real time."""

    sim_start: datetime  # When the simulation begins in sim-time
    sim_days_per_real_hour: float = 12.0  # Acceleration factor

    _sim_current: datetime | None = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False)
    _real_start: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        self._sim_current = self.sim_start

    @property
    def current(self) -> datetime:
        """Current simulation datetime."""
        assert self._sim_current is not None
        return self._sim_current

    @property
    def step(self) -> int:
        return self._step

    @property
    def elapsed_sim_days(self) -> int:
        return (self.current - self.sim_start).days

    @property
    def elapsed_real_seconds(self) -> float:
        return time.monotonic() - self._real_start

    @property
    def sim_date_str(self) -> str:
        return self.current.strftime("%Y-%m-%d")

    def advance(self, days: int = 1) -> datetime:
        """Advance simulation by N days. Returns new sim time."""
        assert self._sim_current is not None
        self._sim_current += timedelta(days=days)
        self._step += 1
        return self._sim_current

    def projected_end_date(self, real_hours: float) -> datetime:
        """What sim-date will we reach after N real hours?"""
        sim_days = real_hours * self.sim_days_per_real_hour
        return self.current + timedelta(days=int(sim_days))

    def real_seconds_per_step(self) -> float:
        """How many real seconds should elapse per sim-day step."""
        # sim_days_per_real_hour days in 3600 seconds
        return 3600.0 / self.sim_days_per_real_hour

    def format_status(self) -> str:
        """Human-readable status string."""
        real_elapsed = self.elapsed_real_seconds
        mins = int(real_elapsed // 60)
        secs = int(real_elapsed % 60)
        return (
            f"Step {self._step} | "
            f"Sim: {self.sim_date_str} (+{self.elapsed_sim_days}d) | "
            f"Real: {mins}m{secs:02d}s"
        )
