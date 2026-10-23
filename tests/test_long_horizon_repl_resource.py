"""Long Horizon Repl contracts: resource."""

from __future__ import annotations

from inquirycraft.memory import Memory

from scienceflow.agent.factory import AgentFactory
from scienceflow.research.control.ephemeral_agent_session import (
    configure_resource_agent_audit,
)
from scienceflow.research.solver.lnr.orchestration.coordinator.resource import (
    construction,
)
from tests._long_horizon_repl_support import *  # noqa: F401,F403


def test_lhr_first_user_prompt_includes_resource_context_snapshot() -> None:
    text = build_first_user_prompt(
        "Task body",
        wall_clock_budget_sec=300,
        resource_context="allocated_compute:\n  - worker_cpu_list: 32-39\n  - visible_gpu_list: 6",
    )

    assert "ResourceContext snapshot:" in text
    assert "worker_cpu_list: 32-39" in text
    assert "visible_gpu_list: 6" in text

def test_lhr_resource_observer_ignores_short_light_jobs(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    observer = LHRResourceObserver(state_machine=store, worker_id="W00", min_register_sec=600)
    job = observer.job_created(command="ls -la", inferred_class="readonly_cpu", gpu_ids=[], timeout_sec=30)
    assert job is None
    assert not (tmp_path / "lhr_events.jsonl").exists()

def test_lhr_resource_observer_promotes_long_job_with_filtered_signal(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        min_register_sec=5,
        check_interval_sec=2,
        stalled_stdout_sec=20,
        kill_enabled=True,
    )
    job = observer.job_created(command="python3 solution.py", inferred_class="heavy_cpu_candidate", gpu_ids=[], timeout_sec=1800)
    assert job
    assert not (tmp_path / "lhr_events.jsonl").exists()
    decision = observer.active_intervention_decision(
        job,
        elapsed_sec=6,
        stdout_age_sec=1,
        stdout_lines=10,
        stdout_bytes=200,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
    )
    assert decision["enabled"] is True
    assert decision["terminate"] is False
    observer.job_finished(job, status="success", elapsed_sec=7, returncode=0, reason=None)
    events = (tmp_path / "lhr_events.jsonl").read_text().splitlines()
    assert len(events) == 2
    state = __import__("json").loads((tmp_path / "lhr_state.json").read_text())
    assert state["resource_jobs"] == 1
    assert state["workers"]["W00"]["resource_jobs"] == 1
    assert "python3 solution.py" not in (tmp_path / "lhr_events.jsonl").read_text()

def test_lhr_resource_observer_can_request_stalled_termination(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        min_register_sec=0,
        stalled_stdout_sec=3,
        kill_enabled=True,
        kill_mode="auto",
        review_state_enabled=False,
    )
    job = observer.job_created(command="python3 solution.py", inferred_class="heavy_cpu_candidate", gpu_ids=[], timeout_sec=1800)
    decision = observer.active_intervention_decision(
        job,
        elapsed_sec=10,
        stdout_age_sec=4,
        stdout_lines=0,
        stdout_bytes=0,
        saw_training_progress=False,
        saw_final_score=False,
        current_phase="",
    )
    assert decision["terminate"] is True

def test_lhr_resource_observer_does_not_promote_static_gpu_class_without_gpu_id(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    observer = LHRResourceObserver(state_machine=store, worker_id="W00", min_register_sec=600)
    job = observer.job_created(command="python3 solution.py", inferred_class="heavy_gpu_candidate", gpu_ids=[], timeout_sec=1800)
    assert job
    assert not (tmp_path / "lhr_events.jsonl").exists()

def test_lhr_resource_observer_does_not_promote_gpu_env_allocation_alone(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    observer = LHRResourceObserver(state_machine=store, worker_id="W00", min_register_sec=600)
    job = observer.job_created(command="python3 solution.py", inferred_class="heavy_gpu_candidate", gpu_ids=["0"], timeout_sec=1800)
    assert job
    assert not (tmp_path / "lhr_events.jsonl").exists()

def test_lhr_resource_observer_monitor_agent_shadow_event_is_filtered(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path, worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        min_register_sec=0,
        stalled_stdout_sec=99,
        monitor_agent_mode="shadow",
        monitor_agent_min_interval_sec=1,
    )
    job = observer.job_created(command="python3 solution.py --secret /home/user/path", inferred_class="heavy_cpu_candidate", gpu_ids=[], timeout_sec=1800)
    observer.active_intervention_decision(
        job,
        elapsed_sec=2,
        stdout_age_sec=0,
        stdout_lines=3,
        stdout_bytes=120,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
    )
    text = (tmp_path / "lhr_events.jsonl").read_text()
    assert "resource_monitor_agent_shadow" in text
    assert "python3 solution.py" not in text
    assert "/home/user/path" not in text
    state = __import__("json").loads((tmp_path / "lhr_state.json").read_text())
    assert state["resource_monitor_agent_events"] == 1

def test_lhr_gpu_queue_blocks_same_gpu_heavy_job(tmp_path: Path) -> None:
    first = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    second = _lhr_resource_observer_with_gpu_queue(tmp_path, "W01")
    job1 = first.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job1
    acquired = first.queue_try_acquire(job1, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert acquired["enabled"] is True
    assert acquired["acquired"] is True

    job2 = second.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job2
    blocked = second.queue_try_acquire(job2, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert blocked["enabled"] is True
    assert blocked["acquired"] is False
    assert blocked["reason"] == "gpu_slot_unavailable"

    first.job_finished(job1, status="success", elapsed_sec=5, returncode=0)
    acquired_after_release = second.queue_try_acquire(job2, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert acquired_after_release["enabled"] is True
    assert acquired_after_release["acquired"] is True

    first_events = (tmp_path / "W00" / "logs" / "lhr_events.jsonl").read_text()
    second_events = (tmp_path / "W01" / "logs" / "lhr_events.jsonl").read_text()
    assert "resource_gpu_lease_acquired" in first_events
    assert "resource_gpu_lease_released" in first_events
    assert "resource_gpu_queue_wait_started" in second_events
    first_state = json.loads((tmp_path / "W00" / "logs" / "lhr_state.json").read_text())
    second_state = json.loads((tmp_path / "W01" / "logs" / "lhr_state.json").read_text())
    assert first_state["resource_requests"] == 1
    assert first_state["resource_gpu_lease_acquired"] == 1
    assert first_state["resource_gpu_lease_released"] == 1
    assert first_state["resource_active_leases"] == []
    assert second_state["resource_requests"] == 1
    assert second_state["resource_gpu_queue_waits"] == 1
    assert len(second_state["resource_active_leases"]) == 1
    assert second_state["resource_active_leases"][0]["job_id"] == job2
    assert second_state["resource_pending_gpu_jobs"] == []
    aggregate = LHRStateMachineStore.aggregate_logs(
        output_log_dir=tmp_path / "task_logs",
        input_log_dirs=[tmp_path / "W00" / "logs", tmp_path / "W01" / "logs"],
        worker_count=2,
        run_status="running",
        ledger_filename=".run_results.md",
    )
    assert aggregate["resource_gpu_lease_acquired"] == 2
    assert aggregate["resource_gpu_lease_released"] == 1
    assert aggregate["resource_requests"] == 2
    assert aggregate["resource_gpu_queue_waits"] == 1
    assert len(aggregate["resource_active_leases"]) == 1
    assert aggregate["resource_active_leases"][0]["worker_id"] == "W01"
    assert aggregate["resource_pending_gpu_jobs"] == []
    assert [event["event"] for event in aggregate["resource_last_events"]][-1] == "resource_gpu_lease_acquired"

def test_lhr_gpu_tt_light_jobs_share_same_gpu_slot(tmp_path: Path) -> None:
    first = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    second = _lhr_resource_observer_with_gpu_queue(tmp_path, "W01")
    job1 = first.job_created(
        command="CUDA_VISIBLE_DEVICES=0 python3 predict.py --tta 4",
        inferred_class=RESOURCE_GPU_TT_LIGHT,
        gpu_ids=["0"],
        timeout_sec=900,
    )
    assert job1
    acquired = first.queue_try_acquire(job1, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])
    assert acquired["enabled"] is True
    assert acquired["acquired"] is True
    assert acquired["slot_weight"] == 0.25

    job2 = second.job_created(
        command="CUDA_VISIBLE_DEVICES=0 python3 predict.py --tta 4",
        inferred_class=RESOURCE_GPU_TT_LIGHT,
        gpu_ids=["0"],
        timeout_sec=900,
    )
    assert job2
    acquired2 = second.queue_try_acquire(job2, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])
    assert acquired2["enabled"] is True
    assert acquired2["acquired"] is True
    state = json.loads((tmp_path / "W01" / "logs" / "lhr_state.json").read_text())
    assert state["resource_active_leases"][0]["resource_class"] == RESOURCE_GPU_TT_LIGHT

def test_lhr_gpu_tt_light_does_not_share_with_train_by_default(tmp_path: Path) -> None:
    train = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    tt = _lhr_resource_observer_with_gpu_queue(tmp_path, "W01")
    train_job = train.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert train_job
    assert train.queue_try_acquire(train_job, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is True

    tt_job = tt.job_created(
        command="CUDA_VISIBLE_DEVICES=0 python3 predict.py --tta 4",
        inferred_class=RESOURCE_GPU_TT_LIGHT,
        gpu_ids=["0"],
        timeout_sec=900,
    )
    assert tt_job
    blocked = tt.queue_try_acquire(tt_job, inferred_class=RESOURCE_GPU_TT_LIGHT, gpu_ids=["0"])
    assert blocked["enabled"] is True
    assert blocked["acquired"] is False
    assert blocked["reason"] == "gpu_slot_unavailable"
    details = blocked["details"]["blocker_details"]["0"]
    assert "incompatible_active_class" in details["reasons"]

def test_lhr_gpu_queue_timeout_hard_gate_blocks_later_train(tmp_path: Path) -> None:
    first = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    second = _lhr_resource_observer_with_gpu_queue(tmp_path, "W01")
    job1 = first.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job1
    assert first.queue_try_acquire(job1, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is True
    job2 = second.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job2
    assert second.queue_try_acquire(job2, inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE, gpu_ids=["0"])["acquired"] is False
    second.queue_timeout(job2, elapsed_sec=2.1, reason="gpu_train_queue_wait_exceeded_2s")

    later = second.job_created(
        command="torchrun --nproc_per_node 1 train_again.py",
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert later
    decision = second.resource_preflight_decision(
        later,
        inferred_class=RESOURCE_HEAVY_GPU_CANDIDATE,
        gpu_ids=["0"],
    )
    assert decision["allowed"] is False
    assert "RESOURCE_FEEDBACK" in decision["feedback"]
    assert decision["reason"] == "block_train_after_gpu_pressure"
    assert "scope=per_gpu" in decision["feedback"]
    text = (tmp_path / "W01" / "logs" / "lhr_events.jsonl").read_text()
    assert "resource_policy_gate" in text

def test_lhr_gpu_queue_noops_for_non_gpu_heavy_job(tmp_path: Path) -> None:
    observer = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    (tmp_path / "train.py").write_text("print(42)\n", encoding="utf-8")
    job = observer.job_created(
        command="python3 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800,
        workspace_dir=tmp_path,
        classifier_reason="train_entrypoint",
    )
    assert job
    decision = observer.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])
    assert decision == {"enabled": False, "acquired": True}

def test_lhr_gpu_lease_mode_assigns_one_visible_gpu(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path / "W00" / "logs", worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        task_resource_dir=tmp_path / "task_logs" / "resource",
        resource_runtime_enabled=True,
        gpu_queue_enabled=True,
        gpu_pressure_min_free_mem_gb=0.0,
        gpu_pressure_yellow_free_mem_buffer_gb=0.0,
        gpu_pressure_yellow_util_pct=101.0,
        gpu_pool=[],
        gpu_default_request=1,
        gpu_max_request=1,
        gpu_assignment="lease",
    )
    job = observer.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0", "1"],
        timeout_sec=1800,
    )
    assert job
    decision = observer.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=["0", "1"])
    assert decision["enabled"] is True
    assert decision["acquired"] is True
    assert decision["gpu_ids"] == ["0"]
    env = observer.lease_env_updates(job)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL"] == "0"

def test_lhr_gpu_lease_mode_uses_configured_pool_without_visible_env(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path / "W00" / "logs", worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        task_resource_dir=tmp_path / "task_logs" / "resource",
        resource_runtime_enabled=True,
        gpu_queue_enabled=True,
        gpu_pressure_min_free_mem_gb=0.0,
        gpu_pressure_yellow_free_mem_buffer_gb=0.0,
        gpu_pressure_yellow_util_pct=101.0,
        gpu_pool=["2", "3"],
        gpu_default_request=1,
        gpu_max_request=1,
        gpu_assignment="lease",
    )
    job = observer.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=[],
        timeout_sec=1800,
    )
    assert job
    decision = observer.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=[])
    assert decision["acquired"] is True
    assert decision["gpu_ids"] in (["2"], ["3"])
    env = observer.lease_env_updates(job)
    assert env["CUDA_VISIBLE_DEVICES"] == decision["gpu_ids"][0]
    assert env["SCIENCEFLOW_ASSIGNED_CUDA_PHYSICAL"] == decision["gpu_ids"][0]
    assert env["SCIENCEFLOW_TASK_GPU_POOL_PHYSICAL"] == "2,3"

def test_lhr_gpu_lease_mode_honors_torchrun_multi_gpu_request(tmp_path: Path) -> None:
    store = LHRStateMachineStore(log_dir=tmp_path / "W00" / "logs", worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        task_resource_dir=tmp_path / "task_logs" / "resource",
        resource_runtime_enabled=True,
        gpu_queue_enabled=True,
        gpu_pressure_min_free_mem_gb=0.0,
        gpu_pressure_yellow_free_mem_buffer_gb=0.0,
        gpu_pressure_yellow_util_pct=101.0,
        gpu_pool=["0", "1", "2"],
        gpu_default_request=1,
        gpu_max_request=4,
        gpu_assignment="lease",
    )
    job = observer.job_created(
        command="torchrun --nproc_per_node 2 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=[],
        timeout_sec=1800,
    )
    assert job
    decision = observer.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=[])
    assert decision["acquired"] is True
    assert decision["gpu_ids"] == ["0", "1"]
    env = observer.lease_env_updates(job)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["SCIENCEFLOW_ASSIGNED_CUDA_LOGICAL"] == "0,1"

def test_lhr_bash_tool_applies_leased_cuda_env_from_source_hint(tmp_path: Path) -> None:
    from scienceflow.runtime.safety.tooling.bash import BashTool

    (tmp_path / "train.py").write_text(
        "import os\n# torch.cuda source hint\nprint(os.environ.get('CUDA_VISIBLE_DEVICES', ''))\n",
        encoding="utf-8",
    )
    store = LHRStateMachineStore(log_dir=tmp_path / "logs", worker_id="W00")
    observer = LHRResourceObserver(
        state_machine=store,
        worker_id="W00",
        task_resource_dir=tmp_path / "task_logs" / "resource",
        resource_runtime_enabled=True,
        gpu_queue_enabled=True,
        gpu_pressure_min_free_mem_gb=0.0,
        gpu_pressure_yellow_free_mem_buffer_gb=0.0,
        gpu_pressure_yellow_util_pct=101.0,
        gpu_pool=["2"],
        gpu_assignment="lease",
        gpu_source_hint_enabled=True,
        gpu_source_hint_mode="observe",
    )
    tool = BashTool(
        workspace_dir=tmp_path,
        bash_timeout_sec=5,
        bash_timeout_slow_sec=5,
        resource_observer=observer,
    )

    async def _run() -> None:
        result = await tool.execute("python3 train.py")
        assert result.error is None
        assert "\n2" in (result.output or "")

    asyncio.run(_run())
    text = (tmp_path / "logs" / "lhr_events.jsonl").read_text()
    assert "resource_source_hint_detected" in text
    assert "resource_gpu_lease_acquired" in text
    assert "resource_gpu_lease_released" in text

def test_lhr_gpu_queue_timeout_records_event(tmp_path: Path) -> None:
    first = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    second = _lhr_resource_observer_with_gpu_queue(tmp_path, "W01")
    job1 = first.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job1
    assert first.queue_try_acquire(job1, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    job2 = second.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job2
    assert second.queue_try_acquire(job2, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is False
    second.queue_timeout(job2, elapsed_sec=2.1, reason="gpu_queue_wait_exceeded_2s")
    text = (tmp_path / "W01" / "logs" / "lhr_events.jsonl").read_text()
    assert "resource_gpu_queue_timeout" in text
    assert "python3 train.py" not in text
    state = json.loads((tmp_path / "W01" / "logs" / "lhr_state.json").read_text())
    assert state["resource_pending_gpu_jobs"] == []

def test_lhr_gpu_util_sample_is_recorded_without_command_text(tmp_path: Path) -> None:
    observer = _lhr_resource_observer_with_gpu_queue(tmp_path, "W00")
    assert observer.resource_runtime is not None

    def fake_sample(*, gpu_ids: list[str]) -> dict[str, object]:
        return {
            "available": True,
            "reason": "",
            "gpus": [
                {
                    "gpu_id": gpu_ids[0],
                    "utilization_gpu_pct": 42.0,
                    "memory_used_mb": 1024.0,
                    "memory_total_mb": 8192.0,
                }
            ],
        }

    observer.resource_runtime.sample_gpu_util = fake_sample  # type: ignore[method-assign]
    job = observer.job_created(
        command="torchrun --nproc_per_node 1 train.py",
        inferred_class="heavy_gpu_candidate",
        gpu_ids=["0"],
        timeout_sec=1800,
    )
    assert job
    assert observer.queue_try_acquire(job, inferred_class="heavy_gpu_candidate", gpu_ids=["0"])["acquired"] is True
    observer.active_intervention_decision(
        job,
        elapsed_sec=12,
        stdout_age_sec=1,
        stdout_lines=5,
        stdout_bytes=128,
        saw_training_progress=True,
        saw_final_score=False,
        current_phase="training",
    )
    state = json.loads((tmp_path / "W00" / "logs" / "lhr_state.json").read_text())
    assert state["resource_gpu_util_samples"] == 1
    assert state["resource_last_events"][-1]["event"] == "resource_gpu_util_sampled"
    assert state["resource_last_events"][-1]["gpu_util"][0]["utilization_gpu_pct"] == 42.0
    assert "torchrun --nproc_per_node" not in (tmp_path / "W00" / "logs" / "lhr_events.jsonl").read_text()

def test_lhr_resource_state_summary_slot_is_removed_from_memory() -> None:
    feedback = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because active_resource_plan_guard; "
        "mode=YELLOW; scope=worker_plan; blocked=heavy_gpu_train; gpu=0.\n"
    )

    class FakeStorage:
        def __init__(self, owner):
            self.owner = owner

        def clear(self):
            self.owner.messages.clear()

    class FakeChatHistory:
        def __init__(self, owner):
            self.owner = owner
            self.storage = FakeStorage(owner)

        def retrieve(self, window_size=None):
            del window_size
            return [SimpleNamespace(memory_record=SimpleNamespace(message=m)) for m in self.owner.messages]

    class FakeMemory:
        def __init__(self):
            self.messages = [
                Message.user_message("task context"),
                Message.user_message(RESOURCE_STATE_SUMMARY_MARKER + "\nold"),
            ]
            self.chat_history_memory = FakeChatHistory(self)

        def add_message(self, message):
            self.messages.append(message)

    dummy = SimpleNamespace(
        memory=FakeMemory(),
        _resource_feedback_memory_deduper=ResourceFeedbackMemoryDeduper(),
        _resource_state_summary_last_text="",
    )
    setattr(
        dummy,
        "_sync_resource_state_summary_slot",
        MethodType(getattr(ScienceAgent, "_sync_resource_state_summary_slot"), dummy),
    )

    dummy._resource_feedback_memory_deduper.reduce(feedback)
    dummy._sync_resource_state_summary_slot()
    joined = "\n".join(str(m.content or "") for m in dummy.memory.messages)
    assert RESOURCE_STATE_SUMMARY_MARKER not in joined
    assert "repeat_count=1" not in joined
    assert "old" not in joined

    dummy._resource_feedback_memory_deduper.reduce(feedback)
    dummy._sync_resource_state_summary_slot()
    joined = "\n".join(str(m.content or "") for m in dummy.memory.messages)
    assert RESOURCE_STATE_SUMMARY_MARKER not in joined
    assert "repeat_count=2" not in joined
    assert "task context" in joined

def test_lhr_resource_state_summary_slot_cleanup_rewrites_long_term(tmp_path) -> None:
    from scienceflow.research.state.knowledge.memory.records.agent_records import create_agent_memory
    from scienceflow.research.state.knowledge.context.memory_context import MemoryContextManager

    feedback = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because active_resource_plan_guard; "
        "mode=YELLOW; scope=worker_plan; blocked=heavy_gpu_train; gpu=0.\n"
    )
    mem = create_agent_memory(tmp_path / "summary_mem", "ScienceAgent", 200)
    mem.add_message(Message.user_message("You are solving one ML task in a continuous REPL workspace. Task body."))
    mem.add_message(Message.user_message(RESOURCE_STATE_SUMMARY_MARKER + "\nold"))
    ctx = MemoryContextManager(mem, tmp_path / "workspace", budget_chars=60000)
    dummy = SimpleNamespace(
        memory=mem,
        _memory_ctx=ctx,
        _resource_feedback_memory_deduper=ResourceFeedbackMemoryDeduper(),
        _resource_state_summary_last_text="",
    )
    setattr(
        dummy,
        "_sync_resource_state_summary_slot",
        MethodType(getattr(ScienceAgent, "_sync_resource_state_summary_slot"), dummy),
    )

    for _ in range(2):
        dummy._resource_feedback_memory_deduper.reduce(feedback)
        dummy._sync_resource_state_summary_slot()

    messages = [r.memory_record.message for r in mem.chat_history_memory.retrieve(window_size=None)]
    joined = "\n".join(str(m.content or "") for m in messages)
    long_term = tmp_path / "summary_mem" / "ScienceAgent" / "long_term.jsonl"
    long_term_lines = [line for line in long_term.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(long_term_lines) == len(messages)
    assert joined.count("You are solving one ML task in a continuous REPL workspace.") == 1
    assert RESOURCE_STATE_SUMMARY_MARKER not in joined
    assert "repeat_count=2" not in joined

def test_lhr_resource_feedback_repeats_are_deduped_before_memory() -> None:
    dummy = SimpleNamespace(
        _agent_hidden_workspace_filenames=(),
        _agent_hidden_workspace_path_prefixes=(),
        _tool_output_artifacts=None,
        _exec_feedback_max_chars=4000,
    )
    for name in (
        "_get_agent_hidden_workspace_filenames",
        "_normalize_hidden_workspace_path",
        "_get_agent_hidden_workspace_path_prefixes",
        "_text_mentions_hidden_workspace_prefix",
        "_hide_agent_hidden_workspace_filename_mentions",
        "_path_mentions_hidden_workspace_file",
        "_tool_request_mentions_hidden_workspace_file",
        "_hide_agent_hidden_workspace_file_lines",
        "_prepare_tool_feedback_for_memory",
        "_dedup_resource_feedback_for_memory",
    ):
        setattr(dummy, name, MethodType(getattr(ScienceAgent, name), dummy))

    class FakeMemoryCtx:
        def record_tool_result(self, tool_name, args, tool_result, **kwargs):
            return tool_result.output

    dummy._memory_ctx = FakeMemoryCtx()
    feedback = (
        "RESOURCE_FEEDBACK: DENIED_REPLAN because active_resource_plan_guard; "
        "mode=YELLOW; scope=worker_plan; blocked=heavy_gpu_train; gpu=0.\n"
    )

    first = dummy._prepare_tool_feedback_for_memory(
        "bash",
        {"command": "python train.py"},
        ToolResult(output=feedback),
    )
    second = dummy._prepare_tool_feedback_for_memory(
        "bash",
        {"command": "python train.py"},
        ToolResult(output=feedback),
    )
    third = dummy._prepare_tool_feedback_for_memory(
        "bash",
        {"command": "python train.py"},
        ToolResult(output=feedback),
    )

    assert first == ""
    assert second == ""
    assert third == ""
    summary = dummy._resource_feedback_memory_deduper.summary_text()
    assert "RESOURCE_STATE_SUMMARY" in summary
    assert "active_resource_plan_guard" in summary
    assert "repeat_count=3" in summary

def test_lhr_resource_control_agents_use_feedback_stage(tmp_path: Path) -> None:
    async def _run() -> None:
        feedback_stage = SimpleNamespace(model="feedback-model")
        code_stage = SimpleNamespace(model="code-model")
        owner = SimpleNamespace(
            cfg=SimpleNamespace(
                agent=SimpleNamespace(code=code_stage, feedback=feedback_stage),
                exp_id="unit",
            ),
            lhr=SimpleNamespace(
                resource_arbiter_enabled=True,
                resource_arbiter_mode="llm",
                resource_admission_llm_enabled=True,
            ),
            deadline=999999999.0,
            global_log_dir=tmp_path / "global_logs",
            task_root_dir=tmp_path / "task",
            _resource_arbiter_agent=None,
            _resource_admission_agent=None,
        )
        created_overrides = []
        created_agents = []

        class FakeAgent:
            async def run_ephemeral_agentic_route_prompt(
                self, prompt, *, trigger, base_messages=None, llm_role=None
            ):
                assert llm_role == "feedback"
                if trigger == "resource_arbiter":
                    return '{"action":"DENY_KILL","confidence":"high","reason":"unit"}'
                return '{"action":"RUN_NOW","confidence":"high","reason":"unit"}'

        class FakeOrchestrator:
            @staticmethod
            def make_llm_call_tracer(**kwargs):
                assert "llm_role=feedback" in kwargs.get("detail_prefix", "")
                return lambda payload: None

            @staticmethod
            def create_science_agent(**kwargs):
                created_overrides.append(kwargs.get("llm_stage_override"))
                agent = FakeAgent()
                created_agents.append(agent)
                return agent

        owner.orchestrator = FakeOrchestrator()
        owner._agent_factory_service = lambda: AgentFactory(
            owner.orchestrator.create_science_agent
        )
        owner._worker_llm_stage_override = lambda _stage="code": None
        owner._worker_uid_prefix = lambda: "W03"
        arbiter_decider = construction._make_resource_arbiter_decider(owner)
        admission_decider = construction._make_resource_admission_decider(owner)

        arbiter_result = await arbiter_decider({"proposal_id": "p1", "proposal_type": "kill_proposal"})
        admission_result = await admission_decider(
            {"task_id": "t1", "resource_class": "gpu_tt_light"},
            {"action": "RUN_NOW", "lease_grantable_by_llm": True},
        )

        assert arbiter_result["action"] == "DENY_KILL"
        assert "RUN_NOW" in admission_result
        assert created_overrides == [feedback_stage, feedback_stage]
        assert [agent._scienceflow_runtime_log_dir for agent in created_agents] == [
            tmp_path
            / "global_logs"
            / "resource"
            / "agent_runtime_audit"
            / "W03-resource-arbiter",
            tmp_path
            / "global_logs"
            / "resource"
            / "agent_runtime_audit"
            / "W03-resource-admission-arbiter",
        ]
        assert [agent._scienceflow_worker_id for agent in created_agents] == [
            "W03-resource-arbiter",
            "W03-resource-admission-arbiter",
        ]

    asyncio.run(_run())


def test_resource_agent_llm_call_uses_unified_runtime_audit(tmp_path: Path) -> None:
    async def _run() -> None:
        class FakeLLM:
            model = "feedback-model"

            async def ask_tool_stream(self, *, handle, **_kwargs):
                await handle.put('{"action":"RUN_NOW"}')
                handle.finish()
                return SimpleNamespace(
                    content='{\"action\":\"RUN_NOW\"}',
                    reasoning_content="resource reasoning",
                    tool_calls=[],
                )

        workspace = tmp_path / "task" / "workspace"
        workspace.mkdir(parents=True)
        agent = ScienceAgent(
            llm=FakeLLM(),
            memory=Memory(max_messages=20),
            workspace_dir=workspace,
            max_steps=1,
        )
        owner = SimpleNamespace(
            cfg=SimpleNamespace(exp_id="resource-audit-test"),
            global_log_dir=tmp_path / "global_logs",
            task_root_dir=tmp_path / "task",
            _worker_uid_prefix=lambda: "W03",
        )
        configure_resource_agent_audit(owner, agent, "resource-arbiter")

        response = await agent.run_ephemeral_agentic_route_prompt(
            "choose resource action",
            trigger="resource_arbiter",
            base_messages=[],
            llm_role="feedback",
        )

        assert response == '{"action":"RUN_NOW"}'
        audit_dir = (
            tmp_path
            / "global_logs"
            / "resource"
            / "agent_runtime_audit"
            / "W03-resource-arbiter"
        )
        provider = [
            json.loads(line)
            for line in (audit_dir / "agent_provider_calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert provider[-1]["call_kind"] == "ephemeral"
        assert provider[-1]["turn_kind"] == "resource_arbiter"
        assert provider[-1]["route"] == "resource_arbiter"
        assert provider[-1]["llm_role"] == "feedback"

        journals = list((audit_dir / "agent_runtime_operations").glob("*.jsonl"))
        assert len(journals) == 1
        operations = [
            json.loads(line)
            for line in journals[0].read_text(encoding="utf-8").splitlines()
        ]
        intent = next(row for row in operations if row["phase"] == "intent")
        settled = next(row for row in operations if row["phase"] == "settled")
        assert intent["payload"]["turn_kind"] == "resource_arbiter"
        assert intent["payload"]["llm_role"] == "feedback"
        assert intent["payload"]["provider_messages"][-1]["content"] == (
            "choose resource action"
        )
        assert settled["payload"]["message"]["content"] == (
            '{"action":"RUN_NOW"}'
        )
        assert settled["payload"]["message"]["reasoning_content"] == (
            "resource reasoning"
        )

        events = [
            json.loads(line)
            for line in (audit_dir / "agent_runtime_events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert any(row["type"] == "llm.completed" for row in events)

    asyncio.run(_run())
