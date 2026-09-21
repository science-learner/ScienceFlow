from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.contracts.run_tool_runtime_contract import (
    DEFAULT_BASELINE,
    capture_tool_runtime_contract,
    compare_contract,
)


@pytest.mark.asyncio
async def test_tool_runtime_contract_matches_frozen_baseline(tmp_path: Path) -> None:
    current = await capture_tool_runtime_contract(tmp_path)
    expected = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    report = compare_contract(current, expected)
    assert report == {
        "benchmark": "tool_runtime_contract",
        "baseline": "baseline_v1",
        "totals": {
            "schema_diff_count": 0,
            "context_diff_count": 0,
            "mechanism_diff_count": 0,
            "error_text_diff_count": 0,
            "workspace_diff_count": 0,
            "failed_categories": 0,
        },
        "failed_categories": [],
        "category_sha256": {
            "schema": "4c4ed9c67937a77168138376de904e25693157b6b97715c5a64c520dfc1bbbaf",
            "context": "100d5268101cb0a29494252457f25d2608ec86807ad2c9ee682bd128f600163f",
            "mechanism": "bff9e71504e6e9a909595ce1412907f696f20464db23593957561f222dccc8ef",
            "error_text": "1d9f95d9c5481eee2ca362bb1fe88879a57208531a75b0ddcac0d21a0b903b5b",
            "workspace": "59646dd397ec346c554f1c4a3b973273e961100695a3119e6aa4a390076771df",
        },
        "verdict": "pass",
    }
