# SCI-CAL Scientist-Facing Rationale v0.2 Change Log

Status: major revision prepared from Grant Wilson's review of the complete
v0.1 scientist-facing draft; not frozen

Date: `2026-08-16`

## Major Changes

- Replaced the formal-contract narrative with a science-team explanation.
- Moved the once-only calibration equation to the first page and made it the
  organizing principle.
- Defined the main input as the upstream-owned calibration input `d_in`, so a
  generic baseline is no longer presented as a CAL-owned operation.
- Added a schematic physical measurement model and a two-branch calibration-
  lineage figure connecting calibrator evidence, the selected APT, target
  atmosphere, and downstream response.
- Added a factor-role table distinguishing `flxscale`, embodied pointing
  correction, `responsivity`, `sens`, target atmosphere, and compatibility
  `fcf`.
- Explained why sample-dependent atmosphere correction does not generally
  commute with temporal filtering.
- Rewrote atmosphere material in physical order: attenuation, zenith opacity,
  sample airmass, per-array transmission, inverse correction, then formal
  interpolation in an appendix.
- Consolidated the missing atmosphere record into one prominent owner-
  decision box and distinguished packet-only absence from project-wide
  absence.
- Reframed the output as a point-source-equivalent, beam-peak-normalized
  amplitude and added a smoothing/filter response example.
- Added a practical uncertainty-budget table separating conditional sample
  noise from correlated calibration systematics.
- Replaced the seven-component state tuple in the main text with a compact
  science-user status table; exact machine states remain in the appendix.
- Reorganized validation into structural correctness, atmosphere-model
  representation fidelity, and observational calibration performance.
- Expanded the unresolved owner-decision register from the single numeric
  atmosphere record to nine scientific questions exposed by the owner review.
- Moved hashes and exact process authority to provenance, added conventional
  human-readable references, and retained every formal assumption,
  requirement, and edge prediction through an appendix crosswalk.

## Preserved Without Scientific Change

- the selected APT row's `flxscale` as the sole absolute detector multiplier;
- once-only application and the embodied-pointing rule;
- target-observation atmospheric correction to the top-of-atmosphere plane;
- exclusion of `responsivity`, `sens`, parent `flxscale`, and opaque `fcf` as
  additional absolute multipliers;
- conditional covariance, variance, and weight propagation;
- correlated calibration/systematic uncertainty treatment;
- point-source response qualification;
- opacity policy boundaries and fail-closed numerical behavior; and
- separation of structural, atmosphere-fidelity, and observational claims.

No implementation behavior or validation result was added.
