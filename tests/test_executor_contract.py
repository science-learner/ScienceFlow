from __future__ import annotations

from scienceflow.runtime.safety.execution.evaluation_process import (
    ExecutionResult,
    PythonEvaluationRunner,
    extract_metric_from_output,
    extract_metric_lines_from_stdout,
    parse_traceback,
)


def test_execution_result_and_parsers_freeze_public_contract() -> None:
    assert ExecutionResult(stdout="ok").success is True
    assert ExecutionResult(stdout="ok").display == "ok"
    assert ExecutionResult(stderr="bad", returncode=1).display == "bad"
    assert ExecutionResult().display == "[No output]"
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/tmp/run.py", line 2, in <module>\n'
        "    raise ValueError('bad')\n"
        "ValueError: bad\n"
    )
    assert parse_traceback(stderr, returncode=1) == (
        "ValueError",
        {"msg": "bad"},
        [("line: 2", "type: <module>", "code: raise ValueError('bad')")],
    )
    assert extract_metric_lines_from_stdout("x\nMETRIC_VALUE=0.25\n") == [
        "METRIC_VALUE=0.25"
    ]
    assert extract_metric_from_output("rmse=1.2\nrmse=0.8") == 0.8


async def test_evaluation_runner_preserves_process_contract(tmp_path) -> None:
    python = await PythonEvaluationRunner(tmp_path).run(
        "print('python-ok')", timeout=10
    )
    assert python.returncode == 0
    assert python.stdout == "python-ok\n"
    assert python.stderr == ""
    assert python.term_out[1:] == ["python-ok"]
    assert not list(tmp_path.glob("*.py"))
