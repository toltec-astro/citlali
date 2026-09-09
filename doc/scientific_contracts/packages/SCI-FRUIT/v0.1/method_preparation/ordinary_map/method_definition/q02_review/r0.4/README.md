# Q02: input preflight findings and a bounded next execution

2026-09-09. Review r0.4. **Preparation complete to a scientific decision;
the combined T1/T2 experiment is not ready to run.**

## Program adherence and prior-work recovery

Continue under the [charter](../../r0.4/inputs/program/README.md),
[pilot workflow](../../r0.4/inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md),
[roadmap](../../r0.4/inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md)
and [frozen recovery](../../r0.4/PRIOR_WORK.md). Adopt Q01's freeze, Q02-A/B's
approved substance, and the [approved weighting design](../r0.2/WEIGHTING_TEST_PLAN.md).
Retain the [r0.3 numerical proposals](../r0.3/FIRST_SCREEN_BINDINGS.md) as
proposals. Cite the [replacement intake](../../input_intake/r0.2/README.md)
for file identity and single-pass state. This successor adds discovery
coordinate/flag/segment checks and returns a newly demonstrated training
problem for owner review. Implementation evidence remains manager-only;
no scientific core, upstream source, author-reference list or registry changes.

The owner authorized finishing local checks and preparing a concrete execution
proposal with “Let's go.” This does not approve the previously unbound run
settings or the scientific changes proposed below.

## What the checks resolved

- The new 123424/129081 exports remain consistent with a single pass.
- Both discovery files have a historical chunk-index defect. The actual
  contiguous lengths are recoverable without changing any data: 289 samples,
  ten chunks of 305, then 289. The reported intervals omit 160 stored rows.
  The recovery is independently supported by 152389's chunk logs and
  123424's raw indices/downsampling. Retain the original labels as evidence;
  any later experiment must explicitly adopt the recovered segment record.
- 152389 contains the inputs needed to reconstruct its legacy exported
  detector coordinates. The same formula agrees with 123424's full export
  to 4.34e-19 rad. This is representation evidence, not admission of a frozen
  AST/PTC/MAP parent chain. No new 152389 export is requested for coordinates.
- A local 123424 APT candidate matches all ten inspected identity, geometry,
  flag and calibration columns exactly. The replacement reduction log and
  executable/raw-input content identities are still missing.

## The design problem that needs a decision

The proposed first-half training rule cannot supply its minimum 64 samples
for many detector/chunk groups that have usable second-half occurrences.
Before signal-finiteness and final grid admission, the geometry/flag check
finds 2,875 such groups in 152389 and 2,001 in 123424; 507 and 227 respectively
have no training samples. These are repeated detector/chunk groups, not counts
of unique detectors. Many also have evaluation occurrences near the source.

These are input-feasibility counts, not noise estimates or weighting outcomes.
They rule out treating the present full-population recipe as ready. Reducing
the minimum, dropping groups, changing the source guard, or borrowing training
across chunks changes a scientific choice. None was done.

## Recommended owner decisions

The [execution proposal](EXECUTION_PROPOSAL.md) separates two decisions under
the existing Q02/Q05/Q06 questions:

1. **Approve T1 as a standalone synthetic screen**, using the exact r0.3
   cases, numerical limits and resources plus this revision's completion
   rules. This changes the proposed first execution unit from T1+T2 to T1
   alone. It permits a useful arithmetic/noise benchmark while T2 is held.
2. **Authorize a paper revision of T2's training rule** around one pooled
   observation-level training set per detector, with no N fallback or detector
   pruning. The proposed law and tradeoff are explicit in the execution
   proposal. This is permission to finish that design and its feasibility and
   evidence bindings, not permission to run a real-data comparison.

Both decisions are **open**. A/B need no further substance vote. C remains
open, and 129081 remains reserved. T2's noise/response uncertainty, actual
experimental input permission and final grid/support bindings still need a
completed review. No confidence interval for real data is borrowed from T1.

The [input report](INPUT_PREFLIGHT.md), [counts](DISCOVERY_PREFLIGHT.json) and
[manifest](REVIEW_MANIFEST.json) preserve the evidence. No signal matrix or
stored weight values were read; no noise estimator, weighted map, injection,
PTC fit, feedback replay or qualification ran. No Unity access or push occurred.
`FRUIT-FEEDBACK-METHOD = unavailable_pending_separate_owner_approval`.
