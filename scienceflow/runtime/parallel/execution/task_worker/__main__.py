"""Private subprocess protocol for a single supervised worker."""

from __future__ import annotations

import argparse

from scienceflow.interfaces.cli import _bootstrap_env, _install_cleanup_signals

from . import run_worker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--workspace", "-w", default=".")
    parser.add_argument("--input-data-dir", "-d", default=None)
    parser.add_argument("--type", dest="task_type", choices=("lnr",), default="lnr")
    args = parser.parse_args(argv)
    _bootstrap_env()
    _install_cleanup_signals()
    return run_worker(
        args.task,
        config=args.config,
        workspace=args.workspace,
        input_data_dir=args.input_data_dir,
        task_type=args.task_type,
    )


if __name__ == "__main__":
    raise SystemExit(main())
