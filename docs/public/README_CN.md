<h1 align="center">ScienceFlow</h1>

<p align="center"><b>端到端自动研究 Agent 框架</b></p>

<p align="center">
  <a href="https://www.noahlab.com.hk/news/212"><b>项目报道</b></a>
  ·
  <a href="https://arxiv.org/abs/2608.14354"><b>arXiv 论文</b></a>
  ·
  <a href="../../README.md"><b>English</b></a>
</p>

> [!IMPORTANT]
> **测试预览版。** 当前代码线用于生成公开的 `preview` 分支，面向功能体验、集成测试和演示，
> 已包含 TUI 对话、多任务长程研究、续跑与恢复、资源协调和结构化遥测。稳定版发布前，接口、
> 配置结构、界面细节和工作区元数据仍可能调整。请使用仓库锁文件和独立工作区进行测试，
> 暂不建议用于无人值守的生产任务。

ScienceFlow 是一个端到端自动研究 Agent 框架，面向持续数小时或数天的高效、稳定且目标一致的研究过程。它以可恢复的可执行工作空间为核心，将持久状态、自适应探索与证据感知执行控制统一起来，使 Agent 能够在保留已验证进展的同时继续、调整或恢复研究路线。

ScienceFlow 文档使用 **iqcraft** 作为 InquiryCraft 的简称；Python 包、import 与 CLI 示例仍统一使用
正式名称 `inquirycraft`。

ScienceFlow 在机器学习、科学建模和数学优化任务上持续开展长程研究，并在完整 75 题 MLE-bench 上于 24 小时预算内达到 **70.22 ± 1.18% Any-Medal**，超过已报告最强基线 **4.92 个百分点**。

<p align="center">
  <img src="scienceflow/assets/mlebench_top10_any_medal.png" alt="完整 MLE-bench Any-Medal 排行榜" width="100%">
</p>
<p align="center"><sub><b>图 1a：完整 MLE-bench Any-Medal 排行榜。</b> ScienceFlow 结果为三次独立运行的均值 ± SEM。</sub></p>

## 最新动态

- **[2026-09]** 测试预览版已发布至 `preview` 分支，用于集成测试与反馈。
- **[2026-08]** ScienceFlow 正式开源——框架代码、任务包与文档均已发布在本仓库。
- **[2026-08]** ScienceFlow 论文已上线 [arXiv](https://arxiv.org/abs/2608.14354)。

## 核心概念

1. **可恢复的可执行状态。** 每个持续运行的 LNR worker 在隔离的可执行工作空间中推进研究；归档状态将工作空间与紧凑记忆、验证证据和资源记录绑定在一起。
2. **Stage Gate。** 任务特定的结果信号触发 `GateService`：配置的 Evaluator 生成标准化证据，Gate policy 决定是否准入。通过准入的结果被物化为不可变 Stage，并记录 ledger facts 和可恢复工作空间快照。
3. **ESTRA。** 在研究边界，Executable-State Transition through Re-Anchoring 做一次两轴决策：起点（当前工作区或已归档 Stage）与意图（`continue` 继续或 `redirect` 重定向）。选择归档起点时，系统会在下一研究分段开始前恢复对应的可执行状态。
4. **持久记忆。** Add 记录通过准入的 Stage 进展；Fold 保留 recent、best-validated 和 anchor-relevant 证据，并压缩旧记录；Unfold/restore 检索带索引的证据与状态，Assemble 为下一分段构造与锚点对应的上下文。
5. **证据感知执行控制。** 研究 worker 决定科学路线，控制器依据资源可用性、剩余预算、已验证进展和可恢复性，对物理任务进行准入、资源租约、在线监控、限时与停止；有效 worker 状态最终归档到 `merge/finals/final_*`。

## 系统架构

<p align="center">
  <img src="scienceflow/assets/scienceflow_system_architecture.png" alt="ScienceFlow 系统架构" width="100%">
</p>
<p align="center"><sub><b>图 2：ScienceFlow 系统架构。</b> Research worker 基于可恢复的可执行状态推进研究，通过边界触发的 ESTRA 调整长程轨迹；证据感知执行控制负责协调物理资源分配与任务执行。</sub></p>

## 设计边界

- LNR 默认 **no-skill**：`lnr_skill_tool_enabled: false` 且 `lnr_skill_auto_read: false`。
- `.scienceflow/skills/data_processing/` 仅保留给专用的数据准备 agent 和 validation-split 工作流。
- `auto` 是默认 evaluator backend，会将注册任务解析为 `task_package`；`artifact_command` 仍可用于通用的命令式评测。
- 并行运行在任务级绑定 CPU/GPU 资源，再在 worker 之间切分 CPU 配额；GPU 租约支持多 worker 受控共享。
- 在任务契约允许时，结果信号可以不产生 submission 而创建 Stage；Merge 只能从携带所需 artifact 的候选中产出 final。

## 仓库结构

```text
ScienceFlow/
├── scienceflow/                         # 框架运行时
│   ├── core/                            # Agent 运行时、工具、记忆与执行
│   ├── solver/                          # LNR、Stage 生命周期、ESTRA、恢复与合并
│   ├── gates/                           # Stage Gate 与 Evaluator 插件
│   ├── safety/                          # 证据感知资源与执行控制
│   ├── ui/                              # Monitor 与 trace 界面
│   ├── config/                          # 默认配置与示例 manifest
│   ├── utils/                           # 共享运行时工具
│   └── cli.py                           # 命令行入口
├── tasks/                               # 任务包与评测器
├── scripts/                             # 运行与监控 manifest
├── .scienceflow/skills/data_processing/ # 数据准备 skills
└── docs/                                # 公开文档、架构、规划、Benchmark 与验收证据
```

## 安装与启动

**前置要求：** Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

InquiryCraft `0.9.0` 已发布到公共包索引，并由 ScienceFlow 锁文件固定版本。
源码环境可直接按照已审查的依赖集合创建：

```bash
cd ScienceFlow
uv sync --python 3.12 --group dev
source .venv/bin/activate
```

不含测试依赖的 Light 环境使用 `uv sync`；完整环境使用
`uv sync --extra full --group dev`。

首次使用时创建模型注册表，填写 API 地址、模型名和 Key，然后启动 TUI：

```bash
scienceflow config init
scienceflow config path
scienceflow tui --workspace /path/to/workspace
```

默认配置位于 `~/.config/scienceflow/models.json`，权限为 `600`。`--model-config PATH`
可临时指定另一份配置；API Key 不会复制到 manifest、session 或报告中。详细字段见
[模型配置说明](LLM_CONFIGURATION.md)。

激活虚拟环境只对当前 shell 生效。如果希望之后直接输入 `scienceflow ...`，可建立用户级链接：

```bash
mkdir -p ~/.local/bin
ln -sfn "$PWD/.venv/bin/scienceflow" ~/.local/bin/scienceflow
scienceflow tui --workspace /path/to/workspace
```

ScienceFlow 在进程内使用 InquiryCraft 作为通用 Agent Runtime，不需要另起服务。常用入口包括：

```bash
scienceflow agent tools list
scienceflow repl
scienceflow parallel -m scripts/lnr.yaml -j 1
scienceflow monitor --manifest scripts/lnr.yaml --refresh 5
scienceflow web --workspace /path/to/workspace
```

普通文本进入 InquiryCraft Agent Loop；`/long-research` 依次完成任务准备、资源与模型问询、
预检和运行。
容器与 Compose 用法见 [`deploy/README.md`](../../deploy/README.md)。

测试预览包发布后，请精确固定预发布版本，避免后续 Preview 自动改变测试环境：

```bash
pip install scienceflow==0.2.0b2
pip install "scienceflow[full]"==0.2.0b2
```

## 配置要点

| 配置项 | 作用 |
|---|---|
| `lnr.num_workers` | 每个任务内部持续运行的 research worker 数量。 |
| `task.cpu_list` / `task.gpu_list` | 任务级 CPU/GPU 资源边界；LNR 会为多 worker 继续切分 CPU。 |
| `lnr.omp_threads_cap` | 每个 worker CPU slice 的线程数上限。 |
| `lnr.wall_clock_budget_sec` | LNR 总运行预算。 |
| `lnr.estra_enabled` / `estra_trigger_stage_count` | 启用 ESTRA，并设置边界审查与 context folding 的触发条件。 |
| `lnr.resource_runtime_enabled` | 启用证据感知资源与执行控制。 |
| `resume_budget_policy` | 恢复运行的预算口径；`fresh` 在历史累计时间上追加本轮 `time_limit`。 |
| `evaluator.backend` | 选择 `auto`（默认）、`task_package` 或 `artifact_command`。 |
| `evaluator.stage_source_mode` | 选择 `shadow`、`adjudicate` 或 `primary` Stage 来源模式。 |
| `evaluator.command.python_executable` | 为任务指定隔离的 Python 环境。 |
| `profile_overrides.<profile>` | 按任务类型覆盖 prompt、Evaluator 和资源行为。 |
| `tasks/**/task.yaml` | 声明任务级 artifact、metric、provider/profile、Evaluator 与 Gate policy。 |
| `metric.authoritative: true` | 将指标声明为可直接进入高可信选择的权威证据。 |

## 文档

[文档索引](../README.md) · [架构总览](scienceflow/index.html) · [可恢复状态与 LNR](scienceflow/module-lnr.html) · [证据感知执行控制](scienceflow/module-resource.html) · [新增优化任务](scienceflow/module-opt-solver-onboarding.html) · [当前规划](../plans/current/) · [SciModelingBench](../../tasks/sci_modeling_bench/README.md)

## 论文任务覆盖

论文使用同一套 ScienceFlow 工作流评测三类可执行研究任务：

- **机器学习工程**：通过 pipeline-construction 接口覆盖完整 75 题 [MLE-bench](https://github.com/openai/mle-bench)。
- **科学建模与设计**：通过 candidate-optimization 接口覆盖 Hugging Face 上的 12 个 [SciModelingBench 任务](https://huggingface.co/datasets/sci-modeling-bench/design-bench)。
- **数学与工程优化**：通过 candidate-optimization 接口覆盖 [Circle Packing](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/circle_packing)、[Ratio Minimization](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/alphaevolve_math_problems/minimizing_max_min_dist)、[Uncertainty Inequality](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples/alphaevolve_math_problems/uncertainty_ineq)，以及 [SpOC4 KTTSP](https://www.esa.int/gsp/ACT/news/spoc-2026/) 的 easy、medium 和 hard 三个 track。

所有任务类型共享 Stage Gate 与 Evaluator contract；每个 `task.yaml` 在通用 solver 之外声明 provider/profile、artifact schema、evaluator backend、authoritative status 与 Gate policy。MLE-bench 可以暂不声明指标方向，由运行期结合任务、阶段和评估证据确定。

## 运行注意

- MLE-bench 任务需要数据根、任务 `exp_id`、`submission.csv` contract 对齐。
- 不同任务 profile 的旧 workspace 不应混用 resume，否则可能继承错误 prompt、dataset 或 artifact 维度。
- `stopped_by_user` 表示人为停止的可恢复终态，不等价于失败；后续 resume 应从 `state.json` 的累计预算和 workspace stage 继续。
- `/resources` 将每个 `starting` 或 `running` task 显示为一行；`stopping` task 不显示，但在进程真正释放资源前仍会从 `Available` 中扣除。

## 验证

```bash
.venv/bin/pytest -q \
  tests/test_inquirycraft_dependency_boundary.py \
  tests/test_inquirycraft_cli_composition.py \
  tests/test_long_research_interaction.py
```

发布 workflow 仅接受与项目版本一致的 tag；它会构建并检查 wheel 与源码包，在干净环境中
安装 wheel，运行公共依赖契约，再通过 PyPI Trusted Publishing 上传。完整回归命令为
`uv run --locked pytest -q`。ML、GPU、MLE-bench 与 scientific-design 测试需要对应的可选依赖。
