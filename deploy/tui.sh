#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
venv_python="${repo_root}/.venv/bin/python"
scienceflow_cli="${repo_root}/.venv/bin/scienceflow"

# The CLI loads the private per-user models.json registry.

if [[ ! -x "${venv_python}" || ! -x "${scienceflow_cli}" ]]; then
    printf '%s\n' \
        "ScienceFlow Light is not installed in ${repo_root}/.venv." \
        "Run: uv sync"
    exit 1
fi

supports_tui_interceptor() {
    "${venv_python}" -c \
        "from inspect import signature; from inquirycraft.cli import create_cli; from inquirycraft.llm import ModelRegistry; from inquirycraft.tui import TuiHostPort; cli={'include_tui','tui_interceptor_factory','tui_branding','model_config_default','model_default_key'}; selector=getattr(TuiHostPort,'open_chat_model_selector',None); raise SystemExit(not (cli.issubset(signature(create_cli).parameters) and selector is not None and 'config_hint' in signature(selector).parameters))" \
        >/dev/null 2>&1
}

if ! supports_tui_interceptor; then
    printf '%s\n' \
        "ScienceFlow TUI requires InquiryCraft 0.9.0 with TUI support." \
        "Update the environment with: uv sync" \
        "For development, install the tested InquiryCraft 0.9.0 wheel explicitly."
    exit 1
fi

# Workspace and state paths come from the same CLI/env/user-file configuration.
exec "${scienceflow_cli}" tui "$@"
