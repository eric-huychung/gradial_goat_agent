"""Runtime configuration for the agent.

Only the three knobs documented in .env.example. If you add one, document it
there too — don't leave a knob documented that the code ignores.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    verbose: bool = False
    task_filter: tuple[str, ...] = ()
    max_tiles: int = 60

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AgentConfig:
        env = os.environ if env is None else env
        raw_filter = env.get("TASK_FILTER", "")
        return cls(
            verbose=env.get("VERBOSE") == "1",
            task_filter=tuple(t.strip() for t in raw_filter.split(",")
                              if t.strip()),
            max_tiles=int(env.get("MAX_TILES", AgentConfig.max_tiles)),
        )
