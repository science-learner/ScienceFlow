# Runtime Parity Benchmark

This fixture set freezes ScienceFlow/InquiryCraft runtime behavior before the remaining ownership
migration. It compares three strict sections for each isolated case:

- `context`: memory, provider requests, prompts, tool schemas, arguments, and final answers;
- `mechanism`: state, round, retry, dispatch, replay/resume, gate, and interrupt ordering;
- `workspace`: relative paths, file kinds, normalized JSON/JSONL/text, logs, and artifacts.

Only fields listed in `normalization.yaml` may be normalized. Unknown fields, paths, ordering changes,
or values are reported as failures. Runtime event UUIDs are mapped to stable sequence IDs while their
parent relationships remain strictly compared.

Run the fast functional gate with:

```bash
.venv/bin/python tests/support/run_runtime_parity.py --suite quick --jobs 4
```

Baseline files can only be written explicitly. Normal development and CI runs never update them:

```bash
.venv/bin/python tests/support/run_runtime_parity.py --suite quick --jobs 1 \
  --record-baseline
```

Baseline updates require a clean worktree, a dedicated commit, and a reviewed reason. Candidate
workspaces are always created under a temporary root and are removed after comparison unless
`--keep-workspaces` is passed.
