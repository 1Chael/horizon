# GPU density / noise parity track — patch snapshots

These patch files preserve, by content, the uncommitted work of the GPU
density and noise-parity investigation (Phases 26–34 of
`docs/gpu-noise-parity-progress.md`). They were produced with `git diff`
between `HEAD` (`eaa7da4fb`) and the worktree, split by modification kind.
**No history rewrite, no stash, no reverts, and nothing is committed.**

They are snapshots, not a series to apply in order; applying both to a clean
`HEAD` reproduces the tracked-content half of the track (the examples are
`#[path]`-included by `crates/horizon-client-lib/Cargo.toml`).

## Contents

- `001-hybrid-density-modify.patch` (586 lines) — modifications to tracked
  files: `crates/horizon-client-lib/Cargo.toml` (example registrations),
  `crates/horizon-core/src/density/graph/{compiler.rs,mod.rs,tests.rs}` (the
  additive `DensityGraph::rebind` and earlier export work),
  `crates/horizon-core/src/density/mod.rs`,
  `crates/horizon-world/src/worldgen/router.rs`, `docs/README.md`,
  `docs/api/shaders.md`, `docs/shaders.md`.
- `002-hybrid-density-new.patch` (6,954 lines) — new files: the density and
  Perlin probe examples (`crates/horizon-client/examples/{density_program_*,
  perlin_probe_diagnostics, radv_divergence_probe, wgpu_*}.rs`), the Phase 30
  leaf-share probe (`hybrid_leaf_probe.rs`), the Phase 31/32 GPU leaf kernel
  and hybrid probe (`gpu_leaves_{probe, hybrid, kernel, tables, gpu}.rs`),
  `crates/horizon-world/examples/vanilla_router_export.rs`,
  `crates/horizon-core/src/density/program.rs` (Phase 16–24
  density-program ladder, R1–R6), and the parity write-ups
  (`docs/gpu-noise-parity-progress.md`, `docs/gpu-noise-parity.md`,
  `docs/gpu-adapter-enablement.md`, `docs/density-ir-design.md`,
  `docs/radv-divergence.md`, `docs/session-handoff.md`).

## What is deliberately NOT in the patches

- `opencode/` and `opencode-old/` scratch directories (untracked, ~1.4 GB;
  left untouched).
- `.local/` vanilla bundles (untracked, never committed).
- Unrelated edits from the base worktree, including the two staged files
  `docs/api/shaders.md` and `docs/shaders.md` are kept in 001 as they were
  part of the track's documentation state at snapshot time.

## Verification at snapshot time

- `cargo test -p horizon-core -p horizon-world` — all test suites pass.
- rustfmt clean on the touched files; `git diff --check` silent.
- Default build unaffected: every addition is either an example (not part of
  the default build targets) or one additive method
  (`DensityGraph::rebind`); no production worldgen file was edited.
- Exact vanilla `final_density` on GPU remains **frozen** (Phase 32 gate
  G32 failed, 1.03x vs the 1.08x bar); see the final report in
  `docs/gpu-noise-parity-progress.md`.
