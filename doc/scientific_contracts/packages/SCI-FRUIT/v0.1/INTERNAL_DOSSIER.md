# SCI-FRUIT v0.1 — Internal Implementation-Informed Dossier

Status: **quarantined Stage A evidence; prohibited from implementation-blind
authorship**

Evidence revision: `7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5`

This dossier records what the repository currently does so that Stage A can
find omissions, contradictions, and ownership risks. It does not describe what
SCI-FRUIT scientifically ought to do. No item below is a default, requirement,
validation result, or conformity claim.

## Current Requested Configuration Surface

The implementation exposes enablement, diagnostic/save-all-iteration controls,
map-only and exact-restart paths, map type/mode, selection and S/N-like limits,
per-array flux limits, support radii, source center, weight feedback, injection,
interpolation, legacy-center behavior, optional post-add-back weight
recomputation, and an absolute maximum iteration. These names mix requested
policy, numerical representation, validation hooks, and historical behavior;
they must not be copied into a scientific contract without owner classification.

## Current Realized Iteration Skeleton

At a high level, current execution appears to:

1. resolve iteration geometry before the back edge;
2. start fruit, weight-validation, learning, map/coadd, pointing, and
   postprocessing lifecycles;
3. load an initial map seed, exact-restart map, or prior-iteration map;
4. apply learned masks/exclusions;
5. project and subtract the loaded map from timestreams;
6. run PTC processing and calculate source-subtracted weights/noise products;
7. add the same projected map back;
8. retain or recompute weights according to requested policy;
9. remove invalid detectors, collect learning, and produce maps/coadds;
10. write iteration outputs, summaries, and a required checkpoint; and
11. increment the absolute iteration.

The observed loop is governed by an absolute maximum and a convergence flag;
the inspected path does not establish a scientific convergence rule. This is
an implementation observation only.

## Current State And Checkpoint Inventory

The v2 checkpoint records schema/creator/type, completed and next absolute
iteration, ordered observation identity, learning and PTC policy, effective
sample-mask intervals, detector-penalty state, and accumulated/finalized PTC
weight-validation sums, counts, factors, and validity. The first checkpoint
format omitted PTC weight-validation state and was found unable to support an
exact continuation claim.

Diagnostic histories/vectors are not restored. That is compatible with exact
restart only if they cannot causally affect later outputs. Stage A must derive
the completeness criterion from scientific state ownership, not freeze the
current field list as sufficient.

## Current Map And Projection Risks

- A configurable map `type` selects a directory/product representation, but it
  does not prove a scientific parent is admissible.
- JINC or bilinear interpolation can be selected or inferred in current policy;
  numerical interpolation is not the same as a scientifically defined forward
  operator or response.
- Loading a map with compatible pixels/WCS does not establish compatible units,
  beam/response, support, validity, calibration, or sky-model meaning.
- Subtracting and adding back the same numerical map does not answer whether the
  evolving object is a cumulative model, residual estimate, increment, filtered
  amplitude field, or selection-dependent source model.

## Historical Evidence Summary

Controlled investigations indicate that projection, PTC transfer recovery,
support/masks, weights, and restart state can change the trajectory. A corrected
v2 restart reproduced uninterrupted execution for the tested state, whereas v1
did not. Population studies found that amplitude, shape, centroid, map change,
and support/learning diagnostics do not become stable together, and some
trajectories are measurement-limited. Qualified relative astrometry/effective
PSF evidence was observed at a stable endpoint, but no universal photometric
correction, science response, or stopping rule was established.

Those findings motivate contract questions; their numbers and behavior remain
validation evidence and are excluded from authorship.

## Audit And Product-Lineage Evidence

Historical audit coordination left SCI-FRUIT unstarted and required an exact
terminal iteration/pass, restart identity, parent-state relation, and delivered
map/kernel relation before downstream transfer products could be available.
This is closure evidence, not scientific authority. The current package retains
the questions while not importing audit conclusions as normative science.

## Quarantine Rule

No future implementation-blind author may inspect this dossier, the code paths,
configuration/schema, validation products, audits, debug notes, repairs, Unity
handoffs, or empirical trajectories. If the sanitized packet is insufficient,
the author must return a precise question rather than consult this evidence.
