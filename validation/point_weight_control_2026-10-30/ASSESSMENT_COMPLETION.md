# Saved 152418 assessment — completion and independent review

2026-10-30. Disposition: reviewed local evidence candidate; no integration,
push, calibration, production activation or additional trial preparation.

- Base/direct parent: `22094011128e5110d1cb9a1dc5afab0abd6685fd`.
- Reviewed candidate: `3bef4e2b97132c4b54a42ace9e14ee690d66adb4`.
- Candidate tree: `3c9c82bc9f838abcee0145f10d31c864089d3902`.
- Branch: `codex/configurable-point-20261028`.
- Worktree: `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor`.
- Initial candidate-review state and final reviewer state: clean.
- This closure is a documentation-only child; it does not change reviewed results.

Changed paths and SHA256 at the reviewed candidate:

| Path | SHA256 |
|---|---|
| `doc/REFACTOR_STATUS.md` | `64f21ba9ce6a953952498b3fa8ed8156fbe175f942850a17b8af5165c9b6111f` |
| `validation/point_weight_control_2026-10-30/ASSESSMENT_WORK_ORDER.md` | `bea4c1b455baf0fcebe30cdd419c9fcda13e5879f274eabd04b019c15141157b` |
| `validation/point_weight_control_2026-10-30/ASSESSMENT.md` | `5a6af00d5fee81204413bd1086b4c29d02ce800a3d2f6c708b05bac6240711ea` |
| `validation/point_weight_control_2026-10-30/assess_saved.py` | `42d15f76947c31534a2bc096527cedbc826a97a43ff312b680e86d90cacecbba` |
| `validation/point_weight_control_2026-10-30/assessment-metrics.json` | `ef0a95bfb66560519894b89b548b3521bd2a81591b55605a3ebefac16e1e6f78` |

## Focused verification and boundary

The saved-only audit passed with the local TolTECA Python environment, warnings
treated as errors, Agg and dedicated temporary caches. Ruff check and format
check passed; `git diff --check` passed. Both figures were visually inspected.
Both saved runs have zero recorded unexpected error-level messages. No native
application build, CTest or config gate is triggered by assessment-only changes.
No representative Unity/Spack qualification is claimed. The paired observational
evidence was produced by Grant before this audit; no reduction or real-source
fit was launched during assessment or review.

Scientific policy, numerical implementation, calibration eligibility and runtime
configuration are unchanged. The new script is an evidence reader, not a new
pipeline stage. Original products, reference reductions and other worktrees remain
preserved. The 10% cut and current detector decisions remain unchanged; temporal
research remains paused.

## Independent fresh-context exact-SHA review

Reviewer: `/root/weight_assessment_review`, fresh context, read-only, reviewing
the exact candidate and tree above. Verdict: **pass with recorded limitations**;
no blocker, major or minor finding requiring repair.

Reviewer read AGENTS; Engineering Governance; Timestream Successor Governance
including the 2026-09-30 amendment; Review and Conformance; canonical governance
effectiveness ledger; relevant architecture, scientific conventions and bright
POINT ownership; current status; original comparison work order, handoff and
independent review; assessment work order; TolTEC skill and authority maps.

**Scientific and behavioral conformance — PASS with limitations.** Independent
saved-only reproduction yields byte-identical JSON and both figures. All 33
array-cycle checks, 726 segment pairs, six support hashes per pair, observation,
grid, rank and health associations, cleaner convergence, HITS equality, FITS
checksums and map/fit bindings reproduce. Separate arithmetic confirms peak
changes −33.157%/−16.572%/−2.097%, a1100 Gaussian brightness-area change −36.968%
and 12″ common-sky aperture change −36.996%. FRUIT gains, initial deficits,
background statistics, final increments and all six fit guard outcomes agree.

Reviewer independently inspected `ForwardMapOperator::resolve`: masks depend
on accumulated coefficients, so output-mask differences do not contradict input
sample equality. Apertures lie inside common support and background measurements
use the shared domain. Interpretation correctly identifies the combined weighting
and consequent FRUIT effect without claiming absolute flux loss, detector failure,
calibration validity or cleaning-only causation.

**Architectural conformity and ownership — PASS.** Only assessment tooling and
documentation change; existing support comparison and saved-map arithmetic are
reused. No application owners, Engine state, policies, interfaces or execution
routes change. No reduction or source refit is invoked.

**Repository, branch and evidence hygiene — PASS with limitations.** Direct
ancestry, exact candidate/tree, scoped inventory and clean state were verified.
Reviewer outputs are under `/tmp`; no edits, commits, Unity access, pipelines,
remote operations or cleanup occurred during review. No unexpected error-level
output occurred; only a Matplotlib font-cache informational message was emitted.

Reproduced digests: metrics `ef0a95bfb66560519894b89b548b3521bd2a81591b55605a3ebefac16e1e6f78`;
maps `bd6ed1590d8f83c27566592f46eb6b32de269667584c3ca3870f05e2764143fe`;
FRUIT/apertures `091e68256df1e6cec0d276656ec746286d31bb82938ec6838318dbeb7cbe07e4`.

Review limitations: support is checked from authenticated saved native publications,
not reconstructed from raw streams. Quiet-mode evidence does not identify full
weight distributions or effective detector counts. Broader application gates,
raw-input reprocessing and Unity/Spack qualification were not performed or
triggered. Both arms remain calibration-ineligible.

Next owner decision: whether to prepare a bounded separation of cleaning weights
and mapmaking weights. That preparation, any additional reductions, transfer tests,
integration and push are outside this completed assessment.
