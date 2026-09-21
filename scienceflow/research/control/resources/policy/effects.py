# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.

"""Validated and idempotent application of resource side effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol


class ResourceEffectKind(str, Enum):
    ACQUIRE = "acquire"
    RELEASE = "release"
    QUEUE_TIMEOUT = "queue_timeout"
    RELEASE_IDLE = "release_idle"
    GRANT_SHARED = "grant_shared"
    REVOKE_SHARED = "revoke_shared"


@dataclass(frozen=True, slots=True)
class ResourceEffectCommand:
    command_id: str
    kind: ResourceEffectKind
    idempotency_key: str
    validated: bool
    parameters: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


@dataclass(frozen=True, slots=True)
class ResourceEffectEvent:
    command_id: str
    kind: ResourceEffectKind
    idempotency_key: str
    applied: bool
    duplicate: bool = False
    result: Mapping[str, Any] = field(default_factory=dict)

    schema_version = "1.0"


class ResourceEffectPort(Protocol):
    def release(self, **kwargs: Any) -> dict[str, Any]: ...


class ResourceEffectExecutor:
    """Only component allowed to apply an already validated resource command."""

    def __init__(self, target: ResourceEffectPort | None) -> None:
        self.target = target
        self._events: dict[str, ResourceEffectEvent] = {}

    def execute(self, command: ResourceEffectCommand) -> ResourceEffectEvent:
        if not command.validated:
            raise PermissionError("resource effect command was not validated")
        key = str(command.idempotency_key or "").strip()
        if not key:
            raise ValueError("resource effect idempotency_key is required")
        previous = self._events.get(key)
        if previous is not None:
            return ResourceEffectEvent(
                command_id=previous.command_id,
                kind=previous.kind,
                idempotency_key=key,
                applied=previous.applied,
                duplicate=True,
                result=previous.result,
            )
        result = self._apply(command)
        event = ResourceEffectEvent(
            command_id=str(command.command_id or ""),
            kind=command.kind,
            idempotency_key=key,
            applied=True,
            result=result,
        )
        self._events[key] = event
        return event

    def _apply(self, command: ResourceEffectCommand) -> Mapping[str, Any]:
        if self.target is None:
            return {"applied": False, "reason": "resource_backend_unconfigured"}
        params = dict(command.parameters)
        dispatch = {
            ResourceEffectKind.ACQUIRE: "queue_try_acquire",
            ResourceEffectKind.RELEASE: "release",
            ResourceEffectKind.QUEUE_TIMEOUT: "queue_timeout",
            ResourceEffectKind.RELEASE_IDLE: "release_idle_lease",
            ResourceEffectKind.GRANT_SHARED: "grant_shared_gpu_lease",
            ResourceEffectKind.REVOKE_SHARED: "revoke_shared_gpu_lease",
        }
        method = getattr(self.target, dispatch[command.kind])
        return dict(method(**params))


__all__ = ["ResourceEffectCommand", "ResourceEffectEvent", "ResourceEffectExecutor", "ResourceEffectKind"]
