#!/usr/bin/env python3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.support.runtime_parity.execution.runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
