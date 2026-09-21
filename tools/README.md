# Maintenance tools

`tools/` contains CI, architecture, and contract-verification utilities. These are
developer tools rather than runnable ScienceFlow experiment manifests.

- `architecture/`: dependency and V5.2 structure checks
- `benchmarks/`: reusable result comparison logic
- `contracts/`: runtime, configuration, and workspace contract checks
- `runtime_parity/`: separated scenario, comparison, and execution support

User-facing run manifests remain in `scripts/`.
