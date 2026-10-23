#!/usr/bin/env python3
"""Validate local ScienceFlow/InquiryCraft TUI behavior and CLI startup."""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path

SCIENCEFLOW = Path(__file__).resolve().parents[2]
INQUIRYCRAFT_TESTS = (
    "tests/integration/test_tui_notice_history.py",
    "tests/integration/test_tui_chat_archive.py",
    "tests/integration/test_tui_chat_history.py",
    "tests/unit/test_model_registry.py",
    "tests/contract/test_llm_factory.py",
    "tests/integration/test_cli_pools.py",
    "tests/integration/test_tui_chat_features.py::test_chat_monitor_uses_host_backend_builder_for_model_switch",
)
SCIENCEFLOW_TESTS = (
    "tests/test_inquirycraft_cli_composition.py",
    "tests/test_tui_sessions.py",
    "tests/test_user_config_cli.py",
    "tests/test_web_monitor.py::test_service_reuses_process_and_stop_isolated",
    "tests/test_web_monitor.py::test_port_conflict_selects_another_port",
    "tests/test_model_registry_config.py",
    "tests/test_inquirycraft_llm_factory_contract.py",
    "tests/test_llm_proxy_config.py",
    "tests/test_plural_llm_config.py::test_chat_model_switch_uses_scienceflow_proxy_client_builder",
)


def run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int):
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out after {timeout}s: {' '.join(command)}") from exc


def check_sources(python: Path, inquirycraft: Path, env: dict[str, str]) -> None:
    command = [
        str(python),
        "-c",
        "import inquirycraft, scienceflow; "
        "print(inquirycraft.__file__); print(scienceflow.__file__)",
    ]
    result = run(command, cwd=SCIENCEFLOW, env=env, timeout=30)
    paths = result.stdout.strip().splitlines()
    expected = (inquirycraft / "src/inquirycraft", SCIENCEFLOW / "scienceflow")
    if (
        result.returncode
        or len(paths) != 2
        or any(
            not Path(actual).resolve().is_relative_to(root)
            for actual, root in zip(paths, expected, strict=True)
        )
    ):
        raise RuntimeError(
            "The selected interpreter did not load both local source trees:\n"
            + result.stdout
            + result.stderr
        )
    print(f"Source pair: InquiryCraft {paths[0]} · ScienceFlow {paths[1]}")


def check_tests(label: str, python: Path, tests: tuple[str, ...], *, cwd, env) -> bool:
    result = run(
        [str(python), "-m", "pytest", "-q", *tests], cwd=cwd, env=env, timeout=240
    )
    summary = result.stdout.strip().splitlines()[-1:] or [result.stderr.strip()]
    print(f"{label}: {'PASS' if result.returncode == 0 else 'FAIL'} · {summary[0]}")
    if result.returncode:
        print(result.stdout + result.stderr, file=sys.stderr)
    return result.returncode == 0


def check_cli(cli: list[str], args: list[str], expected: str, *, env) -> bool:
    result = run([*cli, *args], cwd=SCIENCEFLOW, env=env, timeout=30)
    good = result.returncode == 0 and expected in result.stdout
    print(f"CLI {' '.join(args)}: {'PASS' if good else 'FAIL'}")
    if not good:
        print(result.stdout + result.stderr, file=sys.stderr)
    return good


def startup_median(cli: list[str], option: str, samples: int, *, env) -> float:
    durations = []
    for _ in range(samples + 1):
        started = time.perf_counter()
        result = run([*cli, option], cwd=SCIENCEFLOW, env=env, timeout=30)
        elapsed = time.perf_counter() - started
        if result.returncode:
            raise RuntimeError(f"CLI {option} failed: {result.stderr}")
        durations.append(elapsed)
    return statistics.median(durations[1:])  # discard the cold first sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inquirycraft",
        type=Path,
        default=SCIENCEFLOW.parent / "InquiryCraft",
        help="Local InquiryCraft checkout (default: sibling InquiryCraft)",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-help-seconds", type=float, default=3.5)
    parser.add_argument("--max-version-seconds", type=float, default=6.5)
    args = parser.parse_args()
    inquirycraft = args.inquirycraft.expanduser().resolve()
    # Keep the venv entry path: resolving its symlink would run the base
    # interpreter without the virtual environment's installed packages.
    python = Path(os.path.abspath(args.python.expanduser()))
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if not python.is_file() or not (inquirycraft / "src/inquirycraft").is_dir():
        parser.error("Provide an existing Python interpreter and InquiryCraft checkout")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(inquirycraft / "src"), str(SCIENCEFLOW), env.get("PYTHONPATH"))
        if part
    )
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("SCIENCEFLOW_TEST_CLEAN_REPO_ROOT", None)
    prerequisites = run(
        [str(python), "-c", "import click, pytest"],
        cwd=SCIENCEFLOW,
        env=env,
        timeout=30,
    )
    if prerequisites.returncode:
        print(
            "Validation: FAIL · selected Python needs click and pytest", file=sys.stderr
        )
        return 2
    check_sources(python, inquirycraft, env)
    tests_ok = check_tests(
        "InquiryCraft restored history",
        python,
        INQUIRYCRAFT_TESTS,
        cwd=inquirycraft,
        env=env,
    )
    tests_ok = (
        check_tests(
            "ScienceFlow CLI/TUI", python, SCIENCEFLOW_TESTS, cwd=SCIENCEFLOW, env=env
        )
        and tests_ok
    )

    executable = python.with_name("scienceflow")
    expected_version = tomllib.loads((SCIENCEFLOW / "pyproject.toml").read_text())[
        "project"
    ]["version"]
    cli = (
        [str(executable)]
        if executable.is_file()
        else [str(python), "-c", "from scienceflow.interfaces.cli import main; main()"]
    )
    smoke_ok = all(
        (
            check_cli(cli, ["--help"], "tui", env=env),
            check_cli(
                cli,
                ["--version"],
                f"scienceflow, version {expected_version}",
                env=env,
            ),
            check_cli(cli, ["tui", "--help"], "--resume", env=env),
            check_cli(cli, ["agent", "tools", "list"], "shell", env=env),
        )
    )
    if not tests_ok or not smoke_ok:
        print("Validation: FAIL")
        return 1
    help_sec = startup_median(cli, "--help", args.samples, env=env)
    version_sec = startup_median(cli, "--version", args.samples, env=env)
    speed_ok = (
        help_sec <= args.max_help_seconds and version_sec <= args.max_version_seconds
    )
    print(
        f"Startup median ({args.samples} warm samples): --help {help_sec:.2f}s "
        f"(limit {args.max_help_seconds:.2f}s), --version {version_sec:.2f}s "
        f"(limit {args.max_version_seconds:.2f}s)"
    )
    passed = tests_ok and smoke_ok and speed_ok
    print(f"Validation: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Validation: FAIL · {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
