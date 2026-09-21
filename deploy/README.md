# Light and Full deployment

ScienceFlow exposes two product profiles. **Light** is the default CPU control plane and uses
an external OpenAI-compatible/vLLM provider. **Full** adds the complete task and model stack.
Both profiles use the same source tree and preserve identical LNR, Resume, Resource, and
Coordinator behavior.

```bash
pip install scienceflow                  # Light
pip install "scienceflow[full]"          # Full
```

The container build retains internal layers so CI and maintainers can validate individual
dependency families without changing the two-profile product interface.

| Target | Python profile | Purpose |
| --- | --- | --- |
| `core` | default dependencies | Control plane and external OpenAI-compatible/vLLM provider |
| `ml` | `ml` extra | CPU tabular/scientific task environment |
| `gpu` | `gpu` extra | CUDA 12.8 PyTorch and pretrained model environment |
| `mlebench` | `mlebench` extra | MLEBench integration and DICOM inspection |
| `full` | `full` extra | Compatibility image matching the historical all-in-one install |

## Local installation

```bash
uv sync                                      # Light
uv sync --extra full --group dev             # Full plus tests
```

`uv sync` does not install the `dev` group by default. Use `--group dev` whenever tests are
required. MCP and scientific-design remain independent extras.

Run `scienceflow config init` to create the private
`~/.config/scienceflow/models.json`, then edit the model aliases and endpoints. Start the
full-screen Light interface with one command:

```bash
./deploy/tui.sh
```

The launcher checks the project-local `.venv`. Ordinary TUI chat and long research both
load `models.json`; code and feedback aliases remain independent, and each model can have
multiple failover endpoints. Command options are forwarded unchanged, for example
`./deploy/tui.sh --session-log session.jsonl`. By default,
conversation, event, and recovery logs live under `~/.local/state/scienceflow/tui/`
(or `$XDG_STATE_HOME/scienceflow/tui`). `SCIENCEFLOW_TUI_STATE_DIR` overrides that directory.
The CLI uses the current working directory unless `--workspace` or `SCIENCEFLOW_WORKSPACE`
is configured. PyPI installations invoke `scienceflow tui` directly without this source launcher.

## Images

```bash
docker build -f deploy/Dockerfile --target core -t scienceflow:core .
docker build -f deploy/Dockerfile --target gpu -t scienceflow:gpu .

docker run --rm scienceflow:core --help
docker run --rm scienceflow:core agent tools list
```

The Docker build context excludes datasets, workspaces, tests, logs, and the local virtual
environment. Every runtime target runs as the non-root `scienceflow` user.

## Compose with an external provider

```bash
cp deploy/env.example deploy/.env
# Copy models.example.json to models.json, edit it, and chmod 600 models.json.
# Edit deploy/.env to select the workspace and host models.json path.
docker compose --env-file deploy/.env -f deploy/compose.yaml build
docker compose --env-file deploy/.env -f deploy/compose.yaml run --rm scienceflow
```

For a vLLM server running on the Docker host, use
`http://host.docker.internal:8001/v1` as an endpoint URL in `models.json`. vLLM is deliberately not installed inside
ScienceFlow images, so provider quality/runtime differences remain separate from ScienceFlow
framework logic.

Choose an image with `SCIENCEFLOW_TARGET=ml`, `gpu`, `mlebench`, or `full`. This first deployment
layer does not introduce remote worker transport; the selected image still executes the
configured ScienceFlow command in the mounted workspace.
