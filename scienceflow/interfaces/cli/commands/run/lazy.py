"""Register optional agent and TUI commands without loading their runtimes."""

from inspect import signature

import click


class LazyAgentGroup(click.Group):
    def __init__(self):
        super().__init__(
            "agent", help="Run the embedded InquiryCraft agent.", hidden=True
        )
        self._resolved = False

    def parse_args(self, ctx, args):
        if not self._resolved:
            from inquirycraft.cli import create_cli

            from scienceflow.foundation.config.llm.model_registry import (
                default_model_config_path,
            )
            from scienceflow.interfaces.cli.user_config import add_environment_options
            from scienceflow.interfaces.ui.llm_cli import (
                PoolRuntimeFactory,
                plural_cli_options,
            )

            options = {
                "name": "agent",
                "runtime_factory": PoolRuntimeFactory(),
                "model_config_default": default_model_config_path(),
                "model_default_key": "code_models",
            }
            if "include_tui" in signature(create_cli).parameters:
                options["include_tui"] = False
            command = plural_cli_options(create_cli(**options))
            add_environment_options(command, ("AGENT",))
            self.__dict__.update(command.__dict__)
            self.hidden = True
            self._resolved = True
        return super().parse_args(ctx, args)


class LazyTuiCommand(click.Command):
    def __init__(self):
        super().__init__(
            "tui", help="Start the full-screen ScienceFlow terminal interface."
        )
        self._resolved = False

    def parse_args(self, ctx, args):
        if not self._resolved:
            from scienceflow.interfaces.cli.commands.run.tui import build_tui_command
            from scienceflow.interfaces.cli.user_config import add_environment_options

            command = build_tui_command()
            add_environment_options(command, ("TUI",))
            self.__dict__.update(command.__dict__)
            self._resolved = True
        return super().parse_args(ctx, args)


agent = LazyAgentGroup()
tui = LazyTuiCommand()
