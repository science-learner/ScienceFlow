"""Failed provider attempts with measured usage still count towards research cost."""

import json
from types import SimpleNamespace

from inquirycraft.llm import Usage
from inquirycraft.runtime import ProviderObservation

from scienceflow.runtime.observability.telemetry.agent.audit.runtime import (
    ScienceFlowProviderAudit,
)


def test_failed_provider_usage_updates_counters_and_audit(tmp_path):
    calls = []
    host = SimpleNamespace(
        _run_tokens_in=0,
        _run_tokens_out=0,
        _run_tokens_cached=0,
        _run_llm_calls=0,
        _record_llm_call=lambda *args, **kwargs: calls.append(kwargs),
    )
    audit = ScienceFlowProviderAudit(host, tmp_path)
    observation = ProviderObservation(
        phase="failed",
        model="fake",
        attempt=1,
        finish_reason="length",
        usage=Usage(input_tokens=100, output_tokens=16, cached_input_tokens=50),
    )
    context = SimpleNamespace(
        turn_id=1, session_id="s", run_id="r", agent_id="w00", metadata={}
    )
    audit.observe(observation, context)
    assert (host._run_tokens_in, host._run_tokens_out, host._run_tokens_cached) == (
        100,
        16,
        50,
    )
    assert host._run_llm_calls == 1
    assert calls[0]["tokens_input"] == 100 and calls[0]["tokens_output"] == 16
    row = json.loads((tmp_path / "agent_provider_calls.jsonl").read_text())
    assert row["phase"] == "failed" and row["usage_status"] == "known"
    assert row["finish_reason"] == "length" and row["tokens_output"] == 16
    audit.observe(ProviderObservation(phase="failed", model="fake", attempt=2), context)
    assert host._run_tokens_out == 16 and host._run_llm_calls == 1
    assert calls[1]["tokens_output"] is None
