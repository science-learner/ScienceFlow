<h1 align="center">ScienceFlow</h1>

<p align="center"><b>An End-to-End Autoresearch Agent Framework</b></p>

<p align="center">
  <a href="https://www.noahlab.com.hk/news/212"><b>Project News</b></a>
  ·
  <a href="https://arxiv.org/abs/2608.14354"><b>Paper (arXiv)</b></a>
  ·
  <a href="docs/public/README_CN.md"><b>Chinese</b></a>
</p>

> [!IMPORTANT]
> **Testing preview.** This code line backs the public `preview` branch and is intended
> for evaluation, integration testing, and demonstrations. It includes TUI chat,
> multi-task long research, resume and recovery, resource coordination, and structured
> telemetry. Interfaces, configuration schemas, UI details, and workspace metadata may
> change before the stable release. Use the reviewed lock file and isolated workspaces,
> and do not rely on this preview for unattended production workloads.

ScienceFlow is an end-to-end autoresearch agent framework for productive, stable, and goal-aligned research over hours or days. It organizes research around recoverable executable workspaces, coupling persistent state, adaptive exploration, and evidence-aware execution control so agents can continue, redirect, or recover without losing validated progress.

ScienceFlow documentation uses **iqcraft** as the short name for InquiryCraft. Package,
import, and CLI examples continue to use the canonical `inquirycraft` name.

Across machine learning, scientific modeling, and mathematical optimization, ScienceFlow sustains effective long-horizon research and reaches **70.22 ± 1.18% Any-Medal** on the full 75-task MLE-bench within a 24-hour budget, exceeding the strongest reported baseline by **4.92 percentage points**.

<p align="center">
  <img src="docs/public/scienceflow/assets/mlebench_top10_any_medal.png" alt="Representative full MLE-bench Any-Medal leaderboard" width="100%">
</p>
<p align="center"><sub><b>Figure 1a. Full MLE-bench Any-Medal leaderboard.</b> Mean ± SEM over three independent runs for ScienceFlow.</sub></p>

## News

- **[2026-09]** The testing preview is available on the `preview` branch for integration testing and feedback.
- **[2026-08]** ScienceFlow is open source — the framework code, task packages, and documentation are available in this repository.
- **[2026-08]** The ScienceFlow paper is available on [arXiv](https://arxiv.org/abs/2608.14354).

## Core concepts

1. **Recoverable executable state.** Each persistent LNR worker advances research in an isolated executable workspace. An archived state binds that workspace to compact memory, validation evidence, and resource records.
2. **Stage Gate.** A task-specific result signal invokes `GateService`: the configured Evaluator produces normalized evidence, and the Gate policy decides admission. An accepted result materializes an immutable Stage with ledger facts and a recoverable workspace snapshot.
3. **ESTRA.** At a research boundary, Executable-State Transition through Re-Anchoring makes a two-axis decision: a start point (the current workspace or an archived Stage) and an intent (`continue` or `redirect`). Selecting an archived start point restores its executable state before the next research segment.
4. **Persistent memory.** Add records accepted Stage progress. Fold keeps recent, best-validated, and anchor-relevant evidence explicit while summarizing older records; Unfold/restore retrieves indexed evidence and state, and Assemble constructs the anchor-specific context for the next segment.
5. **Evidence-aware execution control.** Research workers choose scientific routes, while the controller admits, leases, monitors, timeboxes, and stops physical jobs using resource availability, remaining budget, validated progress, and recoverability. Valid worker states are finalized under `merge/finals/final_*`.

## System architecture

<p align="center">
  <img src="docs/public/scienceflow/assets/scienceflow_system_architecture.png" alt="ScienceFlow system architecture" width="100%">
</p>
<p align="center"><sub><b>Figure 2. ScienceFlow system architecture.</b> Research workers operate over recoverable executable states and adapt long-horizon trajectories through boundary-triggered ESTRA transitions, while evidence-aware execution control coordinates physical resource allocation and runtime execution.</sub></p>

## Design boundaries

- LNR is **no-skill by default**: `lnr_skill_tool_enabled: false` and `lnr_skill_auto_read: false`.
- `.scienceflow/skills/data_processing/` is retained only for the dedicated data-prep agent and validation-split workflow.
- `auto` is the default evaluator backend and resolves registered tasks to `task_package`; `artifact_command` remains available for generic command-based evaluation.
- Parallel runs bind CPU/GPU resources at task level, then split CPU capacity across workers. GPU leases support controlled sharing by multiple workers.
- Result signals may create Stages without a submission when the task contract permits it. Merge can only emit finals from candidates that carry the required artifact.

## Repository layout

```text
ScienceFlow/
├── scienceflow/                         # Framework package
│   ├── agent/                           # InquiryCraft host ports, construction, and sessions
│   ├── foundation/                      # Architecture rules, typed config, and contracts
│   │   ├── architecture/                # Executable dependency and size policies
│   │   ├── config/                      # LLM/runtime/schema config and packaged profiles
│   │   └── contracts/                   # Stable cross-domain values
│   ├── research/                        # Research-domain decisions and persistent state
│   │   ├── control/                     # Admission, ESTRA, execution-value, and resources
│   │   ├── quality/                     # Assessment, Evaluator, Gate, and finalization
│   │   ├── solver/lnr/                  # Long-horizon solver, split into five responsibilities
│   │   │   ├── lifecycle/               # Stage, workspace, records, and snapshots
│   │   │   ├── orchestration/           # Coordinator and worker execution
│   │   │   ├── resources/               # Resource feedback and physical runtime
│   │   │   ├── support/                 # Context and prompt support
│   │   │   └── transitions/             # ESTRA, resume, and recovery
│   │   └── state/                       # Dataset, knowledge, and workspace state
│   ├── runtime/                         # Physical execution and operational boundaries
│   │   ├── core/                        # Kernel, process, stage, and shared runtime support
│   │   ├── observability/               # Agent I/O, monitoring, correlation, and traces
│   │   ├── safety/                      # Agent, shell, process, and resource safeguards
│   │   └── parallel/                    # Task scheduling and subprocess execution
│   ├── interfaces/                      # User-facing adapters
│   │   ├── cli/                         # Command composition
│   │   └── ui/                          # Read-only monitor and trace presentation
│   └── cli.py                           # Stable `python -m scienceflow.cli` entry point
├── tasks/                               # Task packages and evaluators
├── scripts/                             # Five canonical manifests and two lifecycle launchers
├── tools/                               # Architecture, contract, and parity developer tools
├── .scienceflow/skills/data_processing/ # Data-preparation skills
└── docs/                                # Public docs, architecture, plans, benchmarks, and evidence
```

The package root is intentionally limited to five ownership namespaces: `foundation`,
`agent`, `research`, `runtime`, and `interfaces`. Foundation defines stable contracts and
configuration; research owns policy and recoverable state; runtime owns physical effects,
safety, and observation; agent and interfaces adapt external interaction. InquiryCraft
remains behind typed ports in `scienceflow/agent`, so ScienceFlow policy does not leak into
the generic agent runtime. The default fan-out limit at every depth is five immediate
subdirectories and five direct business modules (`__init__.py` excluded). A small number of
cohesive product surfaces have explicit path-specific ceilings; increases require review in
`tests/test_leaf_package_structure_v5_2.py` rather than changing the global default.

## Install and run

**Requirements:** Python 3.11+ and [uv](https://docs.astral.sh/uv/).

InquiryCraft `0.9.0` is published on the public package index and pinned by the ScienceFlow
lock file. Create the source environment directly from the reviewed dependency set:

```bash
cd ScienceFlow
uv sync --python 3.12 --group dev
source .venv/bin/activate
```

The Light runtime without test dependencies uses `uv sync`. The Full profile is
`uv sync --extra full --group dev`.

Create the private model registry once, edit it, and start the TUI:

```bash
scienceflow config init
scienceflow config path
scienceflow tui --workspace /path/to/workspace
```

The default registry is `~/.config/scienceflow/models.json` with mode `600`. Use
`--model-config PATH` to test another registry. API keys stay in the registry and are not copied
into manifests, sessions, or reports. See [model configuration](docs/public/LLM_CONFIGURATION.md).

Activation is per shell. To expose this checkout without activation, ensure `~/.local/bin` is on
`PATH` and create a user-level link:

```bash
mkdir -p ~/.local/bin
ln -sfn "$PWD/.venv/bin/scienceflow" ~/.local/bin/scienceflow
scienceflow tui --workspace /path/to/workspace
```

ScienceFlow embeds InquiryCraft as its sole generic Agent Runtime; no second process is required.
Useful entry points are:

```bash
scienceflow agent tools list
scienceflow repl
scienceflow parallel -m scripts/lnr.yaml -j 1
scienceflow monitor --manifest scripts/lnr.yaml --refresh 5
scienceflow web --workspace /path/to/workspace
```

After the preview package is published, install the exact pre-release version from PyPI.
Pinning the version prevents a test environment from changing when a later preview is released.
Container and Compose usage lives in [`deploy/README.md`](deploy/README.md).

```bash
pip install "scienceflow==0.2.0b1"
pip install "scienceflow[full]==0.2.0b1"
```

## Configuration essentials

| Setting | Purpose |
|---|---|
| `lnr.num_workers` | Number of persistent research workers inside one task. |
| `task.cpu_list` / `task.gpu_list` | Task-level CPU and GPU resource boundaries; LNR further splits CPU across workers. |
| `lnr.omp_threads_cap` | CPU thread cap for each worker slice. |
| `lnr.wall_clock_budget_sec` | Total wall-clock budget for the LNR process. |
| `lnr.estra_enabled` / `estra_trigger_stage_count` | Enables ESTRA and sets the trigger for boundary review and context folding. |
| `lnr.resource_runtime_enabled` | Enables evidence-aware resource and execution control. |
| `resume_budget_policy` | Budget accounting for resumed runs; `fresh` adds this round's `time_limit` on top of accumulated time. |
| `evaluator.backend` | Selects `auto` (default), `task_package`, or `artifact_command`. |
| `evaluator.stage_source_mode` | Selects the `shadow`, `adjudicate`, or `primary` Stage source mode. |
| `evaluator.command.python_executable` | Points a task at an isolated Python environment. |
| `profile_overrides.<profile>` | Overrides prompts, Evaluator, and resource behavior per task type. |
| `tasks/**/task.yaml` | Declares the task-level artifact, metric, provider/profile, Evaluator, and Gate policy. |
| `metric.authoritative: true` | Marks a metric as authoritative evidence eligible for high-trust selection. |

## Documentation

[Documentation index](docs/README.md) · [Architecture overview](docs/public/scienceflow/index.html) · [Recoverable states and LNR](docs/public/scienceflow/module-lnr.html) · [Evidence-aware execution control](docs/public/scienceflow/module-resource.html) · [Adding optimization tasks](docs/public/scienceflow/module-opt-solver-onboarding.html) · [Current plans](docs/plans/current/) · [SciModelingBench](tasks/sci_modeling_bench/README.md)

## Paper task coverage

The paper evaluates the same ScienceFlow workflow across three classes of executable research tasks:

- **Machine learning engineering:** all 75 [MLE-bench](https://github.com/openai/mle-bench) tasks through the pipeline-construction interface.
- **Scientific modeling and design:** 12 [SciModelingBench tasks on Hugging Face](https://huggingface.co/datasets/sci-modeling-bench/design-bench) through the candidate-optimization interface.
- **Mathematical and engineering optimization:** [Circle Packing](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/circle_packing), [Ratio Minimization](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/alphaevolve_math_problems/minimizing_max_min_dist), [Uncertainty Inequality](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/alphaevolve_math_problems/uncertainty_ineq), and the easy, medium, and hard [SpOC4 KTTSP](https://www.esa.int/gsp/ACT/news/spoc-2026/) tracks through the candidate-optimization interface.

All task families share the Stage Gate and Evaluator contract. Each `task.yaml` keeps provider/profile, artifact schema, evaluator backend, authoritative status, and Gate policy outside the generic solver. MLE-bench tasks may leave metric direction open for the runtime to resolve from task, stage, and evaluation evidence.

## TUI and managed research

`scienceflow tui` starts a new chat session by default; add `--resume` to restore the latest
session for the same workspace. Ordinary text uses InquiryCraft's Agent Runtime. The main
ScienceFlow commands are:

| Command | Purpose |
|---|---|
| `/models` | Select the chat model and show the registry path. |
| `/long-research DESCRIPTION` | Prepare a task, confirm resources and models, preflight, and launch it. |
| `/tasks`, `/status N` | List tasks or inspect one persistent task number. |
| `/stop N`, `/resume N`, `/attach N` | Control or attach to a research task. |
| `/research-usage N`, `/resources` | Show model usage/cost or host and running-task resources. |
| `/web`, `/web stop` | Start or stop the workspace Web monitor. |

Closing the TUI detaches from supervised research; it does not stop workers. Chat cancellation
also does not stop research—use `/stop`. Managed-run records live under
`~/.local/state/scienceflow/managed_runs/`, while chat sessions live under
`<workspace>/.scienceflow/sessions/`.

`/resources` keeps each `starting` or `running` task on one line. Stopping tasks are hidden, but
their reservations remain excluded from `Available` until the processes release them.

The CLI exposes the same lifecycle when a full-screen terminal is not wanted:

```bash
scienceflow run circle-packing --tui --workspace /path/to/experiment
scienceflow status --all
scienceflow stop RUN_ID
scienceflow resume RUN_ID --tui
```

Do not resume a workspace with a different task profile, dataset, prompt, or artifact contract.
MLE-bench additionally requires the data root, `exp_id`, and `submission.csv` contract to agree.
Model pricing is optional and lives only in `models.<alias>.pricing`; missing prices display
`Cost —` rather than silently using a built-in estimate.

The Web monitor shows task status, workers, metric history, Stage lineage, ESTRA/EEC, usage, and
final reports. See [Web monitor behavior](docs/web-monitor.md) for local and SSH access.

## Verification

Run tests through the active development environment against the pinned, published
InquiryCraft `0.9.0` package:

```bash
.venv/bin/pytest -q \
  tests/test_inquirycraft_dependency_boundary.py \
  tests/test_inquirycraft_cli_composition.py \
  tests/test_long_research_interaction.py
```

The release workflow accepts only a version-matching tag, builds and checks the wheel and
source distribution, installs the wheel in a clean runner, and reruns the public dependency
contracts before publishing through PyPI Trusted Publishing. The full regression command is
`uv run --locked pytest -q`. Optional ML, GPU, MLE-bench, and scientific-design tests require
their corresponding extras.
