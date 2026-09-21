"""Private state owned by the ScienceAgent run-loop runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RunSessionState:
    """Mutable control state for one invocation of the policy loop.

    Component state such as memory, workspace, and resource leases deliberately
    stays outside this object and remains owned by the corresponding component.
    """

    initial_max_steps: int
    effective_max_steps: int
    round_index: int = 0
    abort_reason: str = ""

    @classmethod
    def start(cls, max_steps: int) -> RunSessionState:
        frozen_max = int(max_steps)
        return cls(
            initial_max_steps=frozen_max,
            effective_max_steps=frozen_max,
        )

    @property
    def aborted(self) -> bool:
        return bool(self.abort_reason)

    def abort(self, reason: str) -> None:
        self.abort_reason = str(reason)

    def advance(self) -> None:
        self.round_index += 1

    def update_effective_max(self, value: int | None) -> None:
        if value is not None:
            self.effective_max_steps = int(value)


__all__ = ("RunSessionState",)
