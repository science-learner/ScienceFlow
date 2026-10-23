"""CLI and research resolve the same registry without a developer configuration."""

import click
import pytest

from scienceflow.foundation.config.llm.model_registry import default_model_config_path
from scienceflow.interfaces.cli import main
from scienceflow.interfaces.cli.commands.run.tui import build_tui_command
from scienceflow.interfaces.ui.long_research import LongResearchInteraction


def test_tui_and_research_share_empty_config_home(tmp_path, isolated_config_home):
    expected = isolated_config_home / "scienceflow" / "models.json"
    assert not expected.exists()
    assert default_model_config_path() == expected
    interaction = LongResearchInteraction(tmp_path)
    assert interaction._model_config_path == expected
    option = next(p for p in build_tui_command().params if p.name == "model_config")
    assert option.get_default(click.Context(build_tui_command())) == expected
    assert not expected.exists()


def test_explicit_registry_overrides_default(tmp_path, research_model_registry):
    explicit = tmp_path / "explicit-models.json"
    explicit.write_bytes(research_model_registry.read_bytes())
    research_model_registry.unlink()
    interaction = LongResearchInteraction(tmp_path)
    interaction.configure_model_registry(explicit, ())
    assert interaction._model_config_path == explicit
    assert interaction._selected_model_aliases == ("test-model",)
    assert not research_model_registry.exists()


@pytest.mark.parametrize("mode", ["agent", "tui"])
def test_registry_path_resolves_at_invocation(tmp_path, monkeypatch, mode):
    command = (
        main.commands["agent"].commands["run"]
        if mode == "agent"
        else build_tui_command()
    )
    config_home = tmp_path / "new-config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    expected = config_home / "scienceflow" / "models.json"
    with command.make_context(command.name, [], resilient_parsing=True) as context:
        assert context.params["model_config"] == expected
    explicit = tmp_path / "explicit.json"
    with command.make_context(
        command.name, ["--model-config", str(explicit)], resilient_parsing=True
    ) as context:
        assert context.params["model_config"] == explicit
