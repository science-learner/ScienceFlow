"""Evaluator-facing Python execution result built on InquiryCraft process sessions."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inquirycraft.runtime import ProcessExecutionSession, ProcessRequest, ProcessStatus

_FILE_PATTERN = re.compile(r'^\s*File "(.*?)", line (\d+), in (.*)$')
_METRIC_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:score|metric|accuracy|f1|auc|rmsle|rmse|mae|mse|r2|"
    r"val(?:idation)?[\s_]?(?:score|acc|loss))\s*[:=]\s*"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_METRIC_VALUE_LINE = re.compile(r"^\s*METRIC_VALUE\s*=", re.IGNORECASE)
_ID_EXCEPTION_TAIL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_TOKEN_NOTE = re.compile(
    r"Note:\s*Environment variable `[^`]+` is set and is the current active token "
    r"independently from the token you've just configured\.?\s*",
    re.IGNORECASE,
)


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    exec_time: float = 0.0
    term_out: list[str] = field(default_factory=list)
    all_term_out: list[str] = field(default_factory=list)
    metric_lines: list[str] = field(default_factory=list)
    exc_type: str | None = None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None
    output: Any = None
    display_cap_success: int | None = None

    @property
    def success(self) -> bool:
        return self.returncode == 0

    @property
    def display(self) -> str:
        text = self.stderr if self.stderr and not self.success else self.stdout
        if not text and self.term_out:
            text = "\n".join(self.term_out)
        if not text:
            return "[No output]"
        limit = 16_000 if not self.success else self.display_cap_success or 4_000
        return text if len(text) <= limit else text[:limit] + "\n...[Output truncated]"


def _looks_like_exception(stderr: str) -> bool:
    if not stderr.strip():
        return False
    if "Traceback (most recent call last)" in stderr:
        return True
    if any(_FILE_PATTERN.match(line.strip()) for line in stderr.splitlines()):
        return True
    last = stderr.strip().split("\n")[-1].strip()
    head = last.split(":", 1)[0].strip()
    name = head.split()[-1] if head.split() else head
    return bool(
        ":" in last
        and _ID_EXCEPTION_TAIL.match(name)
        and not name.endswith("Warning")
        and (
            name.endswith(("Error", "Exception"))
            or name in {"KeyboardInterrupt", "SystemExit", "GeneratorExit"}
        )
    )


def parse_traceback(
    stderr: str, returncode: int = 0
) -> tuple[str | None, dict | None, list[tuple] | None]:
    if not stderr.strip():
        return None, None, None
    if returncode == 0 and not _looks_like_exception(stderr):
        return None, None, None
    if returncode != 0 and not _looks_like_exception(stderr):
        message = stderr.strip()
        if len(message) > 4000:
            message = message[:3997] + "..."
        return "SubprocessError", {"msg": message}, None
    lines = stderr.strip().split("\n")
    exc_type, exc_info = "SubprocessError", {}
    last = lines[-1].strip()
    if ":" in last and len(last.split(":", 1)[0].split()) < 3:
        exc_type, message = last.split(":", 1)
        exc_info["msg"] = message.strip()
    else:
        exc_info["msg"] = last
    stack: list[tuple] = []
    index = 0
    while index < len(lines):
        match = _FILE_PATTERN.match(lines[index].strip())
        if match:
            source = "code: "
            if (
                index + 1 < len(lines)
                and lines[index + 1].strip()
                and not lines[index + 1].strip().startswith("Traceback")
            ):
                source += lines[index + 1].strip()
                index += 1
            stack.append(
                (f"line: {match.group(2)}", f"type: {match.group(3)}", source)
            )
        index += 1
    return exc_type.strip(), exc_info or None, stack or None


def extract_metric_lines_from_stdout(stdout: str) -> list[str]:
    lines = [line.rstrip() for line in stdout.splitlines() if line.rstrip()]
    exact = list(dict.fromkeys(line for line in lines if _METRIC_VALUE_LINE.match(line)))
    return exact or list(
        dict.fromkeys(line for line in lines if _METRIC_PATTERNS.search(line))
    )


def extract_metric_from_output(stdout: str) -> float | None:
    matches = _METRIC_PATTERNS.findall(stdout or "")
    try:
        return float(matches[-1]) if matches else None
    except (ValueError, IndexError):
        return None


def _strip_working_dir(content: str, working_dir: str) -> str:
    base = working_dir.rstrip("/\\")
    if not content or not base:
        return content

    def replace(match: re.Match) -> str:
        path = match.group(1)
        if not path.startswith(base):
            return match.group(0)
        suffix = path[len(base) :].lstrip("/\\")
        return f'File "{os.path.basename(suffix) if suffix else path}"'

    return re.sub(r'File "([^"]*)"', replace, content)


class PythonEvaluationRunner:
    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        code: str,
        *,
        timeout: float,
        env: dict[str, str] | None = None,
        pre_code: str = "",
    ) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            dir=self.workspace_dir,
            encoding="utf-8",
        ) as stream:
            stream.write(pre_code + code)
            temp_path = Path(stream.name)
        try:
            outcome = await ProcessExecutionSession(
                ProcessRequest(
                    argv=(sys.executable, str(temp_path)),
                    cwd=self.workspace_dir,
                    environment={**os.environ, **(env or {})},
                    timeout_sec=timeout,
                )
            ).run()
            if outcome.status is ProcessStatus.TIMED_OUT:
                return ExecutionResult(
                    stderr=f"TimeoutError: execution exceeded {timeout}s",
                    returncode=-1,
                    exec_time=outcome.elapsed_sec,
                    term_out=[f"TimeoutError: Execution exceeded {timeout}s"],
                    exc_type="TimeoutError",
                    exc_info={"msg": f"Execution exceeded {timeout} seconds"},
                )
            return self._result(
                outcome.output.stdout,
                outcome.output.stderr,
                outcome.elapsed_sec,
                int(outcome.returncode or 0),
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def _result(
        self, stdout: str, stderr: str, elapsed: float, returncode: int
    ) -> ExecutionResult:
        working_dir = str(self.workspace_dir)
        metric_lines = extract_metric_lines_from_stdout(stdout)
        exc_type = exc_info = exc_stack = None
        term_out = [f"Execution time: {elapsed:.1f}s"]
        if stderr.strip():
            exc_type, exc_info, exc_stack = parse_traceback(stderr, returncode)
            if exc_info and isinstance(exc_info.get("msg"), str):
                exc_info["msg"] = _strip_working_dir(exc_info["msg"], working_dir)
        if exc_type:
            term_out.append(f"{exc_type}: {(exc_info or {}).get('msg', '')}")
        elif stdout.strip():
            term_out.extend(stdout.strip().split("\n"))
        if metric_lines:
            existing = "\n".join(term_out)
            extra = [line for line in metric_lines if line not in existing]
            if extra:
                term_out.append("\n[metric lines from stdout]\n" + "\n".join(extra))

        def clean(text: str) -> str:
            text = _strip_working_dir(text, working_dir)
            return _ENV_TOKEN_NOTE.sub("", text).strip("\n")

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            exec_time=elapsed,
            term_out=[clean(item) for item in term_out],
            all_term_out=([stdout] if stdout else [])
            + (["\n[stderr]\n" + stderr] if stderr else []),
            metric_lines=[clean(item) for item in metric_lines],
            exc_type=exc_type,
            exc_info=exc_info,
            exc_stack=exc_stack,
        )


__all__ = [
    "ExecutionResult",
    "PythonEvaluationRunner",
    "extract_metric_from_output",
    "extract_metric_lines_from_stdout",
    "parse_traceback",
]
