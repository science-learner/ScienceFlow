"""Unlimited chat must not leak into caller-owned research preparation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from inquirycraft.runtime import RuntimeOptions

from scienceflow.interfaces.ui.research.preparation import runtime as preparation


@pytest.mark.parametrize("limit,expected", [(None, 20), (5, 5)])
async def test_preparation_retains_finite_turn_limit(
    tmp_path, monkeypatch, limit, expected
):
    created = []

    def create(options, **kwargs):
        result = SimpleNamespace(options=options, aclose=AsyncMock())
        created.append(result)
        return result

    registry = SimpleNamespace(
        resolve=lambda *args, **kwargs: SimpleNamespace(
            models=["fake"],
            api_keys=["fake"],
            base_urls=["https://offline.invalid"],
        )
    )
    monkeypatch.setattr(preparation, "load_model_registry", lambda _: registry)
    owner = SimpleNamespace(
        preparation_runtime_factory=SimpleNamespace(create_runtime=create),
        workspace=tmp_path,
        task_workspace=tmp_path / "task",
        _model_config_path=tmp_path / "models.json",
        _selected_model_aliases=("fake",),
    )

    async def run(prompt, *, show_response, runtime):
        assert runtime.options.max_turns == expected
        return "ready"

    options = RuntimeOptions(model="fake", workspace=tmp_path, max_turns=limit)
    host = SimpleNamespace(runtime=SimpleNamespace(options=options), run_host_task=run)
    assert await preparation.run_preparation_task(owner, host, "prepare") == "ready"
    assert host.runtime.options.max_turns == limit
    created[0].aclose.assert_awaited_once()
