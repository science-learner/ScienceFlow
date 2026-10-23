<p align="center">
  <img src="https://raw.githubusercontent.com/science-learner/ScienceFlow/master/docs/assets/brand/scienceflow-banner.svg" alt="ScienceFlow" width="800">
</p>

<p align="center">
  <a href="https://github.com/science-learner/ScienceFlow/actions/workflows/scienceflow-contract-ci.yml"><img src="https://github.com/science-learner/ScienceFlow/actions/workflows/scienceflow-contract-ci.yml/badge.svg?branch=master" alt="CI"></a>
  <a href="https://pypi.org/project/scienceflow/"><img src="https://img.shields.io/pypi/v/scienceflow?label=PyPI&amp;color=0f766e&amp;cacheSeconds=300" alt="PyPI"></a>
  <a href="https://arxiv.org/abs/2608.14354"><img src="https://img.shields.io/badge/arXiv-2608.14354-B31B1B.svg" alt="arXiv paper"></a>
  <a href="https://github.com/science-learner/ScienceFlow/issues"><img src="https://img.shields.io/badge/feedback-open-334155" alt="Feedback"></a>
  <img src="https://img.shields.io/badge/Python-3.11--3.13-3776AB" alt="Python 3.11–3.13">
</p>

<p align="justify">
ScienceFlow turns an executable task with a measurable objective into persistent,
evaluator-guided research. It coordinates parallel workers, promotes valid results into
recoverable stages, preserves the best artifacts, and exposes the full process through a
TUI and Web monitor.
</p>

## News

- **[2026-09]** Released the ScienceFlow harness preview for long-horizon scientific research agents, with TUI and Web interfaces.
- **[2026-08]** Released ScienceFlow [commit ad1f7dd](https://github.com/science-learner/ScienceFlow/commit/ad1f7ddae57873807da5a9c0070c68ba9a740515) and [arXiv](https://arxiv.org/abs/2608.14354).

## Quick start

<p align="justify"><strong>Requirements:</strong> Python 3.11–3.13 and access to a supported model API.</p>

### Install

Create an isolated Conda environment, then install ScienceFlow from PyPI:

```bash
conda create -n scienceflow python=3.12 -y
conda activate scienceflow
python -m pip install scienceflow==0.2.0
scienceflow --help
```

<details>
<summary><strong>Other installation methods</strong></summary>

**pipx**

```bash
pipx install scienceflow==0.2.0
```

**uv**

```bash
uv tool install scienceflow==0.2.0
```

**Python virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install scienceflow==0.2.0
```

</details>

### Configure and start

```bash
scienceflow config init
scienceflow config path     # edit the generated private model registry
scienceflow tui
```

<p align="justify">
Chat normally in the TUI, or enter <code>/long-research</code> to configure and launch a
managed task. Closing the TUI detaches from research without terminating its workers.
</p>

### Minimal TUI example

<p align="justify">
The built-in <code>circle-packing</code> example needs no dataset. After starting the TUI,
complete these prompts in order:
</p>

**1 · Prepare the task**

```text
/long-research circle-packing workers=2 cpu=16 gpu=cpu duration=1h
```

**2 · Start after preflight passes**

```text
run
```

<p align="justify">
ScienceFlow runs preflight checks, starts two CPU workers, evaluates candidate solutions,
and retains the best valid artifact under the workspace. Follow it with
<code>/tasks</code> or <code>/status 1</code>.
</p>

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="center">Command</th>
      <th width="500" align="center">Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="left"><code>/long-research DESCRIPTION</code></td><td align="left">Prepare and launch a research task.</td></tr>
    <tr><td align="left"><code>/tasks</code>, <code>/status N</code></td><td align="left">List tasks or inspect results, model usage, and resources.</td></tr>
    <tr><td align="left"><code>/stop N</code>, <code>/resume N</code>, <code>/attach N</code></td><td align="left">Control or follow a task.</td></tr>
    <tr><td align="left"><code>/models</code></td><td align="left">Select the fixed chat model.</td></tr>
    <tr><td align="left"><code>/web</code>, <code>/web stop</code></td><td align="left">Start or stop the workspace Web monitor.</td></tr>
  </tbody>
</table>

Choose a saved chat session with:

```bash
scienceflow tui --resume
```

See the [runtime guide](docs/public/RUNTIME_GUIDE.md) for workspace and execution details.

## How it works

<p align="center">
  <img src="https://raw.githubusercontent.com/science-learner/ScienceFlow/150b0a0f3aafb5eee4b7e3cc5e560dfe044a073b/docs/public/scienceflow/assets/scienceflow_system_architecture.png" alt="ScienceFlow system architecture" width="100%">
</p>

1. **Explore:** isolated workers advance executable workspaces under explicit CPU, GPU,
   model, and time budgets.
2. **Evaluate:** task evaluators convert artifacts into normalized evidence and final scores.
3. **Preserve:** Stage Gate admits valid results as immutable, recoverable stages while the
   best artifact remains available independently of submission state.
4. **Adapt:** ESTRA continues the current route or re-anchors work to a stronger archived
   stage at research boundaries.

<p align="justify">
Agent messages, reasoning, tool calls, evaluator results, resource samples, costs, and stage
lineage are retained as structured events for inspection and future training workflows.
</p>

## Task contract

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="center">You provide</th>
      <th width="500" align="center">ScienceFlow manages</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="left">Objective, constraints, and metric direction</td><td align="left">Research lifecycle and parallel workers</td></tr>
    <tr><td align="left">Executable task code and required data</td><td align="left">Isolated workspaces and resource leases</td></tr>
    <tr><td align="left">Evaluator and valid artifact contract</td><td align="left">Evidence normalization and Stage admission</td></tr>
    <tr><td align="left">Baseline or starting implementation</td><td align="left">Exploration, recovery, selection, and finalization</td></tr>
  </tbody>
</table>

<p align="justify">
A registered <code>task.yaml</code> is the source of truth for measurement and valid
artifacts. ScienceFlow does not invent a missing evaluator or silently replace required
task data.
</p>

## Models, resources, and task environments

<p>
Model settings stay local in <code>~/.config/scienceflow/models.json</code> (mode
<code>600</code>); API keys are excluded from task manifests, sessions, and reports. See the
<a href="https://github.com/science-learner/ScienceFlow/blob/master/docs/examples/models.example.json">redacted example</a>.
</p>

<p align="justify">
The base package provides the control plane and TUI. Install task-specific ML/GPU
dependencies only when needed:
</p>

```bash
python -m pip install "scienceflow[full]"==0.2.0
```

<p align="justify">
CPU and GPU allocations form task boundaries; CPU capacity is divided across workers. Model
pricing is optional—without it, the UI reports <code>Cost —</code>. Do not resume a
workspace with a different task, dataset, evaluator, prompt, or artifact contract.
</p>

## Develop from source

```bash
git clone --branch master https://github.com/science-learner/ScienceFlow.git
cd ScienceFlow
uv sync --python 3.12 --group dev
uv run scienceflow config init
uv run scienceflow tui
```

<p align="justify">
Use <code>uv sync --extra full --group dev</code> only when tests require the full ML/GPU
environment.
</p>

## Research

<p align="justify">
ScienceFlow uses the same Stage Gate and evaluator contract across machine-learning
engineering, scientific modeling, and mathematical optimization. In the reported 24-hour,
75-task MLE-bench evaluation, it reaches <strong>70.22 ± 1.18% Any-Medal</strong> over three
runs.
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/science-learner/ScienceFlow/150b0a0f3aafb5eee4b7e3cc5e560dfe044a073b/docs/public/scienceflow/assets/mlebench_top10_any_medal.png" alt="Representative full MLE-bench Any-Medal leaderboard" width="100%">
</p>

## Citation

If ScienceFlow supports your research, please cite:

```bibtex
@misc{zhao2026scienceflow,
  title         = {{ScienceFlow}: A Long-Horizon Agent for {ML} Research, Scientific Discovery and Beyond},
  author        = {Mingming Zhao and Jiqian Dong and Kangping Xu and Zadid Hasan and Chengrui Fan and Shan Jiang and Shuai Mao and Yating Ling and Linyi Zou and Tailin Zhou and Yun Hin Chan and Wenkai Zhang and Zhanhong Zhou and Guowei Huang and Hongliang Li and Wenjing Cun and Zhitang Chen and Mingxuan Yuan and Yanhui Geng},
  year          = {2026},
  eprint        = {2608.14354},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2608.14354},
  url           = {https://arxiv.org/abs/2608.14354}
}
```
