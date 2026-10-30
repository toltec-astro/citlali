# Completion and independent review — 2026-10-30

Disposition: reviewed local saved-evidence candidate. No integration, push,
activation, calibration, runtime/configuration changes or reductions. Noise
weighting retained. All existing products preserved; temporal research paused.

Scoped base: `ea61862a5f4bdc9c3dad0443a7ca295be1ff7e93`.
Initial evidence commit: `f5e53d8e7b7df4ce7af39b912fa305a1643a0d20`.
Final reviewed candidate: `9ca9d13335771ae2cff28a40e845a0d0196475b4`.
Tree: `7cc3f76e23b62a04127b208b063256eef78a693a`.
Direct parent is the initial evidence commit; follow-up changes only figure
labels and the figure digest. Branch: `codex/configurable-point-20261028`.
Worktree: `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor`.
Scoped inventory: status plus this directory's WORK_ORDER.md, REPORT.md,
inspect_saved.py and metrics.json. Initial and final reviewed state: clean.
This closure is a documentation-only child of the reviewed candidate.

Fresh-context read-only reviewer `/root/feedback_inspection_review` reviewed the
exact final SHA and full scope from the base. Verdict: **pass with recorded
limitations; no actionable findings**. Reviewer read AGENTS, all three governance
documents, canonical governance-effectiveness record, status and all scoped files.
Native MAP/FRUIT relevant implementation matches saved-run source
`d042083be71f76a406fb698707f7bde94c928d62`.

- Scientific/behavioral conformance: pass with limitations. Twenty authenticated
  model/map/request/feedback/next-plan associations, model FITS checksums,
  coordinates, selected values and unselected zeros reproduce. All profile,
  selection and sampling results reproduce. Masks settle at M2 with 36/50 pixels.
  Native projection and feedback use the same containing-pixel primitive.
- Architectural conformity/ownership: pass. Evidence script and documentation
  only; source selection, native FRUIT and MAP owners remain unchanged.
- Repository/evidence hygiene: pass. Exact scope and clean state verified. No
  edits, commits, Unity, pipelines, fits, reductions or source mutations by reviewer.

Independent headless saved-only execution with local TolTECA Python and warnings
as errors passes. All nonfigure JSON content and both figure digests reproduce;
figure paths differ only because reviewer outputs live in /tmp. Ruff check/format,
scoped git diff --check, visual inspection and quadrature refinement pass. Maximum
relative-L2 refinement change is 0.0002969823, below numerical gate 0.001. No
unexpected warning/error output; font-cache informational notice only.

Figure SHA256:
- masks-profiles.png: `14521cea3a47cd794693ba0c3d37a5344573591cd4d16142f6bfe10e063ddf3e`.
- sampling.png: `b3cd5c0a749c9a8f9d715b4ccb287ea61864f417422e29e28476ccb387f4d5f5`.

Spatial means do not establish source-wing significance. Gaussian omission is not
known true-flux loss. Uniform-area quadrature does not propagate actual detector
coordinates or cleaning response; neither diagnostic explains the measured peak
deficit quantitatively. No scientific harm threshold or new feedback policy is
selected. Broader build/CTest/config and Unity/Spack gates are not triggered.

Next owner decision: whether to define and prepare a controlled source-support
test, with sampling treated separately and noise weighting fixed. No new trial
has been prepared or launched.
