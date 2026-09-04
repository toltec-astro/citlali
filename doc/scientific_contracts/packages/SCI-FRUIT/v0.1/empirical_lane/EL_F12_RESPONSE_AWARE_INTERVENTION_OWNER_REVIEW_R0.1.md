# SCI-FRUIT v0.1 — Response-aware intervention screen r0.1

Date: `2026-09-04`

Decision candidate: `SCI-FRUIT-EL-F12-RESPONSE-AWARE-INTERVENTION-SCREEN-R0.1`

Status: **owner-review proposal; implementation and execution await approval**

## Proposed decision

Approve one local development screen asking whether considering the expected
map consequence of a new hard exclusion can reduce leakage without sacrificing
source recovery or removing useful exclusions. The proposed rule examines every
new eligible detector/scan, using only the completed iteration that proposes
the action. It never names UID 4460 or reads a later map to make a decision.

The comparison has three arms:

| Arm | Action on a selected new hard exclusion |
| --- | --- |
| Historical hard action | Apply the existing exclusion before cleaning, unchanged |
| Hold | Withhold that exclusion through the fixed end of this short screen |
| Half map contribution | Keep the detector in cleaning and multiply its final map coefficients by 0.5 through the same fixed end |

For either alternative, an unselected exclusion follows the historical rule.
Other flagging and exclusion reasons retain their authority. The alternatives
are bounded experiments, with at most 64 selected detector/scan identities per
trajectory and no execution beyond absolute iteration 6. They are not proposed
production settings.

## What the selector would know

At a completed iteration, the existing learner has identified the detector/scans
it proposes to exclude next. For every newly eligible record, calculate the
whole-map response to deleting its current JINC contribution. A record is
selected if deletion would lose valid support or its response RMS exceeds one
robust map-scatter scale. Also check the combined deletion of the entire new
candidate set; if that combined effect crosses the same rule, select the whole
set. This catches interactions that separate deletion maps can miss.

The exact support, scatter floor, conditioning, missing-data behavior, caps,
and timing are in the [design](EL_F12_RESPONSE_AWARE_INTERVENTION_DESIGN_R0.1.md).
The numerical choices are proposed development choices. EL-F11 did not estimate
or validate them. Large influence is not evidence that an exclusion is wrong;
the paired scientific comparison must answer whether retaining that contribution
helps or harms.

## Control and input boundary

Use observation 123424, all three arrays, and the existing 100 mJy/beam source
at map-world offsets `(0, -60)` arcsec. Each arm has an uninjected and injected
trajectory. All start fresh and retain absolute iterations 0 through 6.

Use the historical complete-product recurrence, `alpha=1`. The EL-F5–EL-F11
diagnosis inherited `alpha=1.25`, which was rejected as a candidate recurrence;
those checkpoints must not become this screen's starting state. There may be
no selected opportunity under `alpha=1`. That is a legitimate result and does
not permit restoring the rejected recurrence or forcing the known event.

The same-build historical-recurrence control does not replace the still
unavailable exact `f70701ad` executable. Full Gate D and historical-superiority
claims remain blocked under the existing readiness record.

## What approval would cover

Choice A would authorize the default-disabled prototype, tests, exact input
and executable registration, and the bounded local matrix in the design:
eight fresh trajectories including a two-trajectory diagnostic-neutrality pair,
plus at most four conditional three-iteration restart checks. The ordinary
source, morphology, residual, support, convergence, runtime, and memory
measurements are required. A negative or unavailable result receives a complete
record; it does not trigger retuning or additional trials.

The design fixes the scientific method and limits now. After approval, the
implementation's exact source, binary, configuration, dependency, and analyzer
identities must be recorded before any real-data execution. Recording those
identities is mechanical preparation, not permission to change this design.
A method, gate, population, input, or scope change returns to the owner.
Routine bugs follow the existing repair direction.

## What a favorable result would mean

It could make an exact candidate eligible for a separately authorized
independent-pointing replication. It would not recommend a policy. Replication
must preserve the rule and scientific protections, include useful-exclusion
controls, and succeed before a policy recommendation. Qualification, wider
science-field transfer tests, Stage B, and production remain later decisions.

## Owner choices

- **A — Approve this bounded screen** against
  [the exact manifest](EL_F12_BUNDLE_MANIFEST_R0.1.md), including its two
  alternative actions, proposed thresholds, full metrics, and run limits.
- **B — Revise the scientific design** before implementation; specify the
  selector, action, metric, or limit to change.
- **C — Stop this intervention branch** and retain the completed diagnosis.

The current request authorizes preparation of this proposal. It does not
select Choice A.
