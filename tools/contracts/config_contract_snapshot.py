#!/usr/bin/env python3
"""Create a secret-free, portable snapshot of a resolved ScienceFlow config."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


SENSITIVE_KEYS = {
    "api_key",
    "api_keys",
    "base_url",
    "base_urls",
    "headers",
    "model",
    "models",
}
PATH_KEY_RE = re.compile(r"(?:^|_)(?:path|dir|root|workspace)(?:$|_)")


def fingerprint(value: Any) -> dict[str, str]:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "redacted_type": type(value).__name__,
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def portable_path(value: str, repo_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return value
    try:
        return f"<repo>/{path.resolve().relative_to(repo_root.resolve()).as_posix()}"
    except ValueError:
        return f"<external>/{path.name}"


def sanitize(value: Any, *, key: str = "", repo_root: Path) -> Any:
    lowered = key.lower()
    if lowered in SENSITIVE_KEYS:
        return fingerprint(value)
    if isinstance(value, dict):
        return {
            str(child_key): sanitize(child, key=str(child_key), repo_root=repo_root)
            for child_key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [sanitize(child, key=key, repo_root=repo_root) for child in value]
    if isinstance(value, str) and PATH_KEY_RE.search(lowered):
        return portable_path(value, repo_root)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    snapshot = {
        "schema_version": "1.0",
        "source_format": "scienceflow.resolved_config.yaml",
        "config": sanitize(raw, repo_root=args.repo_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
