# EL-F14: can simultaneous measurements identify a useful retention?

Date: 2026-09-06

Decision candidate: `SCI-FRUIT-EL-F14-SKY-TIME-FEASIBILITY-R0.1`

Status: proposal for owner review; no new method approved, implemented or run.

## Recommended decision

Approve one small, local test with synthetic data and known truth. Test whether
detectors viewing different sky positions at the same instant can supply a
benefit signal that ordinary map agreement cannot. Compare deletion, half
retention and full retention of one candidate's contribution. The output is a
conditional benefit estimate or an explicit inability to decide.

This is an information test before another real-data experiment. It would use
13 fixed scientific fixtures and four integrity checks, with no Citlali runs,
real-data scoring or interventions. The exact [design and fixtures](EL_F14_SKY_TIME_SEPARATION_DESIGN_R0.1.md)
and [manifest](EL_F14_BUNDLE_MANIFEST_R0.1.md) define the proposed scope.

## Why change direction?

We can predict a map consequence in the diagnosed UID 4460 case, but that does
not tell us which action will help. EL-F12's Half and Hold alternatives both
failed the scientific protections. EL-F13 found no positive benefit evidence:
support loss prevented most comparisons, and the one available comparison
worsened agreement. Its shared-reference example also showed that maps can
agree on the same contamination. These results stay unchanged.

The proposed additional evidence is the relationship between **time and sky
position**, before map accumulation discards that relationship. If two
detectors see different positions simultaneously, subtracting their readings
removes an additive contaminant common to both. Repeated comparisons can then
constrain sky contrasts. This is a proposed model assumption, not a property
already established for Citlali's processed samples.

## What the signal would say

For each fixed retention amount, compute how much it improves map contrast
relative to deletion for **every sky consistent with the donor measurements
and a declared error bound**. A positive lower bound supplies conditional
evidence of benefit. Ambiguous sky structure, missing support or inconsistent
measurements supply no permission to act. The suspect candidate and every
other eligible detector are excluded from the sky inference.

The first test uses three artificial sky cells and exact arithmetic. Its
benefit margin is 1/4 in squared toy signal units. This makes the proposed
criterion attainable and testable; it is not a new astronomical threshold or
a relaxation of EL-F12's failed 0.1 mJy/beam endpoint. Both retention amounts
are reported. The test chooses no action, duration or policy.

Positive, harmful, neutral, ambiguous and inconsistent examples are fixed in
advance. One example favors half retention while full retention has no benefit.
A required pair has identical observations but different true skies because
one contains a sky-shaped contaminant. The signal must return identical
predictions for that pair. This exposes the remaining limitation: this method
cannot establish truth against arbitrary contamination that looks like sky.

## What a successful result would justify

At most, success would establish conditional identifiability in this small
model and justify preparing a separate applicability proposal. Before any
real-data score, that proposal would need to establish simultaneous sample
identity, compatible detector response, a justified residual-error bound and
the relationship between the inferred sky and the processed JINC response.
Those facts are currently unestablished. A synthetic pass is not empirical
benefit, an operational selector, a safeguard or independent replication.

Any later intervention must use the historical complete-product recurrence
as its control and retain source recovery, morphology, residual leakage,
support, useful-exclusion, convergence, runtime and memory protections. It
must declare an attainable benefit endpoint, paired inputs, joint-action,
duration and cap rules before execution. Independent-pointing replication
remains required before recommending a policy. Full Gate D, qualification,
Stage B, production changes and Unity remain separate decisions.

## Boundaries and owner decision

Approval would cover an isolated toy-analysis helper, fixture materialization,
prospective registration, the 17 fixed predictor evaluations and their report.
Limits: one CPU thread, 30 minutes aggregate including focused tests and
routine repairs, 1 GiB peak process RSS and 100 MiB new output. Existing code,
reduction products and both review archives are preserved. No parameter sweep,
new population or extra scientific case is included.

The request to proceed authorized preparation of this proposal. The next
significant decision is approval or revision of this exact synthetic method
and population against the manifest. The [standing owner direction](SCIENTIFIC_OWNER_ROUTINE_DEFECT_REPAIR_DIRECTION_2026-09-04.md)
reserves changes of method, gate, input and scope for owner review; routine
implementation bugs within an approved experiment can be repaired and tested
without another decision. No further permission is needed to prepare this
reviewable package.
