"""Science Agent Repl contracts: construction."""

from __future__ import annotations

from tests._science_agent_repl_support import *  # noqa: F401,F403


def test_candidate_artifact_archive_rejects_path_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside.json").write_text("{}\n", encoding="utf-8")

    result = archive_workspace_candidate_artifact(
        workspace,
        artifact_path="../outside.json",
        snapshot_dir=tmp_path / "snapshots",
        ledger_path=tmp_path / "ledger.jsonl",
    )

    assert result.archived is False
    assert "workspace-relative" in result.message

def test_repl_runtime_options_expose_workspace_git_auto_review() -> None:
    cfg = Config()
    cfg.repl_workspace_git_auto_review = True

    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=True,
    )

    assert opts["workspace_git_auto_review"] is True

def test_repl_runtime_options_expose_workspace_git_auto_checkpoint() -> None:
    cfg = Config()
    cfg.repl_workspace_git_auto_checkpoint = False

    opts = _repl_resolve_runtime_options(
        cfg,
        auto_first_user_enabled=True,
    )

    assert opts["workspace_git_auto_checkpoint"] is False

def test_repl_workspace_git_prompt_is_short_source_control_hint() -> None:
    text = _repl_workspace_git_prompt(True, ["*.py", "*.md"])

    assert "Workspace source checkpoints" in text
    assert "`git status --short`" in text
    assert "`git diff -- '*.py' '*.md'`" in text
    assert "`git log --oneline" in text
    assert "`git restore --source=<commit>" in text
    assert "Do not create commits manually" in text
    assert "datasets, model weights, logs, and submissions" in text
    assert "`*.py`" in text
    assert "Auto-review is enabled" not in text
    assert _repl_workspace_git_prompt(False, ["*.py"]) == ""

    strong = _repl_workspace_git_prompt(
        True,
        ["*.py", "*.md"],
        auto_review=True,
    )
    assert "Auto-review is enabled" in strong
    assert "restore the best tracked source" in strong
    assert "Do not commit manually" in strong

def test_workspace_source_git_initializes_source_doc_repo(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    (tmp_path / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# notes\n", encoding="utf-8")
    (tmp_path / "data.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "description.md").write_text(
        "# dataset\n", encoding="utf-8"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.bin").write_bytes(b"model")

    result = ensure_workspace_source_git(tmp_path, track_globs=["*.py", "*.md"])

    assert result.ready is True
    assert result.initialized is True
    assert result.committed is True
    assert (tmp_path / ".git").is_dir()
    ignore_text = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert render_workspace_gitignore(["*.py", "*.md"]).strip() in ignore_text

    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.splitlines()
    assert set(tracked) == {".gitignore", "notes.md", "solution.py"}

def test_workspace_source_git_ignores_runtime_control_markdown(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    (tmp_path / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    ensure_workspace_source_git(tmp_path, track_globs=["*.py", "*.md"])

    (tmp_path / ".run_results.md").write_text(
        "Final Validation Score: 0.9\n", encoding="utf-8"
    )
    assert workspace_source_changed(tmp_path, track_globs=["*.py", "*.md"]) is False
    result = auto_checkpoint_workspace_source(
        tmp_path, track_globs=["*.py", "*.md"], tool_name="bash"
    )
    assert result.source_changed is False
    assert result.committed is False

    (tmp_path / "solution.py").write_text("print('changed')\n", encoding="utf-8")
    assert workspace_source_changed(tmp_path, track_globs=["*.py", "*.md"]) is True

def test_workspace_source_git_ignores_tmp_runtime_python_files(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    (tmp_path / "solution.py").write_text("print('ok')\n", encoding="utf-8")
    ensure_workspace_source_git(tmp_path, track_globs=["*.py", "*.md"])

    deps = tmp_path / "tmp" / "deps" / "example_pkg"
    deps.mkdir(parents=True)
    (deps / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tmp" / "notes.md").write_text("# runtime note\n", encoding="utf-8")

    assert workspace_source_changed(tmp_path, track_globs=["*.py", "*.md"]) is False
    result = auto_checkpoint_workspace_source(
        tmp_path, track_globs=["*.py", "*.md"], tool_name="bash"
    )
    assert result.source_changed is False
    assert result.committed is False

    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.splitlines()
    assert "tmp/deps/example_pkg/__init__.py" not in tracked
    assert "tmp/notes.md" not in tracked

def test_workspace_source_git_auto_checkpoint_commits_and_snapshots(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    (tmp_path / "util.py").write_text("SEED = 1\n", encoding="utf-8")
    ensure_workspace_source_git(tmp_path, track_globs=["*.py", "*.md"])

    (tmp_path / "util.py").write_text("SEED = 2\n", encoding="utf-8")
    (tmp_path / "submission.csv").write_text("id,y\n1,0.5\n", encoding="utf-8")
    (tmp_path / "ml_run_results.md").write_text(
        "| run_index | timestamp_utc | command | exit_status | validation_metric | metric_direction | submission_path | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 1 | now | python train.py | 0 | 0.1234 | lower_is_better | submission.csv | ok |\n",
        encoding="utf-8",
    )

    result = auto_checkpoint_workspace_source(
        tmp_path,
        track_globs=["*.py", "*.md"],
        tool_name="bash",
    )

    assert result.ready is True
    assert result.committed is True
    assert result.metric_value == pytest.approx(0.1234)
    assert result.submission_snapshot.endswith(".csv")
    assert (tmp_path / result.submission_snapshot).read_text(
        encoding="utf-8"
    ) == "id,y\n1,0.5\n"
    assert (tmp_path / ".scienceflow_checkpoints" / "ledger.jsonl").is_file()
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    assert "ckpt: metric_0p1234 after bash" in log

def test_workspace_source_git_auto_checkpoint_can_bind_current_stage(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable is not available")

    (tmp_path / "train.py").write_text("SEED = 1\n", encoding="utf-8")
    ensure_workspace_source_git(tmp_path, track_globs=["*.py", "*.md"])

    (tmp_path / "submission.csv").write_text("id,y\n1,0.7\n", encoding="utf-8")
    generic_result = auto_checkpoint_workspace_source(
        tmp_path,
        track_globs=["*.py", "*.md"],
        tool_name="bash",
    )

    (tmp_path / "train.py").write_text("SEED = 2\n", encoding="utf-8")
    result = auto_checkpoint_workspace_source(
        tmp_path,
        track_globs=["*.py", "*.md"],
        tool_name="lhr_stage_s01",
        metric_value_override=0.0612,
        stage_id="S01",
    )

    assert result.ready is True
    assert result.committed is True
    assert result.stage_id == "S01"
    assert generic_result.submission_snapshot
    assert result.submission_snapshot != generic_result.submission_snapshot
    assert "_s01_" in result.submission_snapshot
    ledger = tmp_path / ".scienceflow_checkpoints" / "ledger.jsonl"
    row = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert row["stage_id"] == "S01"
    assert row["submission_snapshot"] == result.submission_snapshot
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    assert "ckpt: S01 metric_0p0612 after lhr_stage_s01" in log

def test_stable_system_prompt_omits_round_budget(tmp_path: Path) -> None:
    agent = ScienceAgent(
        llm=MagicMock(),
        memory=Memory(max_messages=50),
        workspace_dir=tmp_path,
        max_steps=10,
        stable_system_prompt=True,
    )
    text = agent._build_system_prompt()
    assert "## Round Budget" not in text
    assert "coding and workspace assistant" in text
