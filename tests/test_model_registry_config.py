from __future__ import annotations

import json

from scienceflow.foundation.config.llm.model_registry import (
    apply_model_registry,
    apply_registry_to_stage,
    load_model_registry,
)
from scienceflow.foundation.config.schema.settings import Config, StageConfig
from scienceflow.interfaces.ui.model_config import (
    apply_research_model_choice,
    configure_draft_models,
    model_alias_completions,
    parse_chat_model_selection,
    research_model_overrides,
    research_model_question,
)
from scienceflow.research.onboarding.manifest import build_parallel_manifest
from scienceflow.research.onboarding.session import LongResearchDraft
from scienceflow.research.onboarding.support.answers import parse_constraints
from scienceflow.runtime.observability.telemetry.agent.llm_cost import (
    load_model_config_prices,
)


def _write_registry(path):
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "deepseek": {
                        "model": "deepseek-v4-flash",
                        "pricing": {
                            "input_usd_per_1m": 0.14,
                            "cached_input_usd_per_1m": 0.0028,
                            "output_usd_per_1m": 0.28,
                        },
                        "endpoints": [
                            {"url": "https://a.example/v1", "key": "a"},
                            {"url": "https://b.example/v1", "key": "b"},
                        ],
                    },
                    "qwen": {
                        "model": "qwen3-coder",
                        "endpoints": [{"url": "https://c.example/v1", "key": "c"}],
                    },
                },
                "defaults": {
                    "code_models": ["deepseek", "qwen"],
                    "feedback_models": ["deepseek"],
                    "selection": "auto",
                },
            }
        ),
        encoding="utf-8",
    )


def test_registry_populates_code_and_feedback_stages(tmp_path):
    path = tmp_path / "models.json"
    _write_registry(path)
    cfg = Config()

    assert apply_model_registry(cfg, path)
    assert cfg.agent.code.model_aliases == ["deepseek", "qwen"]
    assert cfg.agent.code.models == [
        "deepseek-v4-flash",
        "deepseek-v4-flash",
        "qwen3-coder",
    ]
    assert cfg.agent.code.api_keys == ["a", "b", "c"]
    assert cfg.agent.feedback.model_aliases == ["deepseek"]
    assert cfg.agent.feedback.models == ["deepseek-v4-flash", "deepseek-v4-flash"]
    prices = load_model_config_prices(path)
    assert prices["deepseek-v4-flash"].output_usd_per_1m == 0.28


def test_task_stage_can_select_registry_subset(tmp_path):
    path = tmp_path / "models.json"
    _write_registry(path)
    stage = StageConfig(model_aliases=["qwen"], model_config_path=str(path))

    assert apply_registry_to_stage(stage, role="code")
    assert stage.models == ["qwen3-coder"]
    assert stage.api_keys == ["c"]
    assert stage.api_routing_mode == "sticky_failover"
    assert stage.reasoning_replay == "preserve"


def test_registry_applies_reasoning_replay_and_stage_override(tmp_path):
    path = tmp_path / "models.json"
    _write_registry(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"]["deepseek"]["reasoning_replay"] = "required"
    path.write_text(json.dumps(payload), encoding="utf-8")

    required = StageConfig(model_aliases=["deepseek"], model_config_path=str(path))
    assert apply_registry_to_stage(required, role="code")
    assert required.reasoning_replay == "required"

    overridden = StageConfig(
        model_aliases=["deepseek", "qwen"],
        model_config_path=str(path),
        reasoning_replay="omit",
    )
    assert apply_registry_to_stage(overridden, role="code")
    assert overridden.reasoning_replay == "omit"


def test_registry_default_selection_is_applied_when_stage_does_not_override(tmp_path):
    path = tmp_path / "models.json"
    _write_registry(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["defaults"]["selection"] = "fixed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    stage = StageConfig(model_aliases=["deepseek"], model_config_path=str(path))

    assert apply_registry_to_stage(stage, role="code")
    assert stage.model_selection == "fixed"
    assert stage.api_routing_mode == "sticky"


def test_each_stage_can_use_its_own_registry_path(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_registry(first)
    second.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "judge": {
                        "model": "provider/judge",
                        "endpoints": [{"url": "https://judge.example/v1", "key": "judge"}],
                    }
                },
                "defaults": {
                    "code_models": ["judge"],
                    "feedback_models": ["judge"],
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = Config()
    cfg.agent.code.model_aliases = ["deepseek"]
    cfg.agent.code.model_config_path = str(first)
    cfg.agent.feedback.model_aliases = ["judge"]
    cfg.agent.feedback.model_config_path = str(second)

    assert apply_model_registry(cfg)
    assert cfg.agent.code.models == ["deepseek-v4-flash", "deepseek-v4-flash"]
    assert cfg.agent.feedback.models == ["provider/judge"]


def test_legacy_role_prefixes_migrate_to_one_clean_alias(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "code-gpt-5-6-luna": {
                        "model": "gpt-5.6-luna",
                        "endpoints": [
                            {"url": "https://code.example/v1", "key": "code-key"}
                        ],
                    },
                    "feedback-gpt-5-6-luna": {
                        "model": "gpt-5.6-luna",
                        "endpoints": [
                            {
                                "url": "https://feedback.example/v1",
                                "key": "feedback-key",
                            }
                        ],
                    },
                },
                "defaults": {
                    "code_models": ["code-gpt-5-6-luna"],
                    "feedback_models": ["feedback-gpt-5-6-luna"],
                    "selection": "auto",
                },
            }
        ),
        encoding="utf-8",
    )

    registry = load_model_registry(path)

    assert tuple(registry.models) == ("gpt-5.6-luna",)
    assert registry.default_aliases("code_models") == ("gpt-5.6-luna",)
    assert registry.default_aliases("feedback_models") == ("gpt-5.6-luna",)
    assert registry.resolve(("gpt-5.6-luna",), role="code").api_keys == (
        "code-key",
    )
    assert registry.resolve(("gpt-5.6-luna",), role="feedback").api_keys == (
        "feedback-key",
    )
    assert path.with_name(path.name + ".legacy-aliases.bak").is_file()


def test_long_research_model_aliases_are_non_secret_manifest_inputs(tmp_path):
    parsed = parse_constraints(
        "circle-packing code-models=deepseek,qwen feedback-models=deepseek "
        "model-policy=spread"
    )
    assert parsed["code_models"] == ["deepseek", "qwen"]
    assert parsed["feedback_models"] == ["deepseek"]
    draft = LongResearchDraft(
        task_text="circle packing",
        workspace_base=str(tmp_path / "runs"),
        input_data_dir="none",
        metric_name="radius",
        lower_is_better=False,
        artifact_path="solution.json",
        artifact_command="python evaluator.py {artifact_abs_path}",
        gpu_list="cpu",
        cpu_list="0-3",
        workers=2,
        wall_clock_sec=1200,
        code_models=parsed["code_models"],
        feedback_models=parsed["feedback_models"],
        model_selection=parsed["model_selection"],
    )

    task = build_parallel_manifest(draft)["tasks"][0]

    assert task["agent"] == {
        "code": {"model_aliases": ["deepseek", "qwen"], "model_selection": "spread"},
        "feedback": {"model_aliases": ["deepseek"], "model_selection": "spread"},
    }
    assert "key" not in str(task["agent"]).lower()


def test_lightweight_chat_selection_and_completion_use_aliases(tmp_path):
    path = tmp_path / "models.json"
    _write_registry(path)

    assert parse_chat_model_selection(
        "/models qwen policy=fixed", path, current_policy="auto"
    ) == (("qwen",), "fixed")
    assert parse_chat_model_selection(
        "/models 2", path, current_policy="auto"
    ) == (("qwen",), "auto")
    assert model_alias_completions(path, "/models qw") == [
        ("/models qwen", "Chat model")
    ]
    assert model_alias_completions(path, "/models qwen policy=s") == [
        ("/models qwen policy=spread", "Routing policy")
    ]


def test_long_research_models_are_configured_per_run(tmp_path):
    path = tmp_path / "models.json"
    _write_registry(path)
    draft = LongResearchDraft()
    configure_draft_models(draft, path)

    prompt = research_model_question(path, draft)
    assert "Code deepseek,qwen" in prompt
    assert "Feedback deepseek" in prompt
    assert "model-policy=<auto|fixed>" in prompt
    assert (
        "Fixed example · models=deepseek feedback-models=deepseek "
        "model-policy=fixed"
    ) in prompt
    assert "auto|spread|fixed" not in prompt
    assert research_model_overrides(
        "circle packing workers=2 models=qwen feedback-models=deepseek"
    ) == {
        "code_models": ["qwen"],
        "feedback_models": ["deepseek"],
    }

    apply_research_model_choice(
        path,
        draft,
        "models=qwen feedback-models=deepseek model-policy=fixed",
    )
    assert draft.code_models == ["qwen"]
    assert draft.feedback_models == ["deepseek"]
    assert draft.model_selection == "fixed"
