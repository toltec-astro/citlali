# POINT assessment foundation and design handoff

Updated November 6, 2026. Status: accepted as a bounded software increment and
integrated locally at application `ec3a447c1666fddb6be5c5823cee1b438f9c57c7`.
The [owner acceptance](../validation/point_policy_foundation_2026-11-06/OWNER_ACCEPTANCE.json)
binds the tested source and evidence. Grant executes all builds, tests, probes,
benchmarks, reductions and validation runners. No push or activation is included.

Grant authorized this split: this chat builds the first code increment that
collects existing POINT measurements into a versioned assessment record;
“Assess contract compliance” owns rule development and owner approvals.
The approval does not adopt the unresolved details in the
[broader implementation proposal](POINT_POLICY_IMPLEMENTATION_SCOPE_2026-11-05.md).

## Coordination

| Owner | Work and deliverable |
| --- | --- |
| Grant and “Assess contract compliance” | Develop the scientific and use rules; update the [decision table and acceptance record](POINT_METHOD_AND_USE_ACCEPTANCE_2026-11-04.md) and [team summary](POINT_TEAM_SUMMARY_2026-11-04.md). Record each unresolved choice, then each explicit owner disposition. |
| This implementation chat | Maintain the accepted foundation, schema and evidence; prepare the next bounded rule implementation after receiving its exact approved design revision. Report source/evidence bindings and implementation limitations. |
| Grant | Execute all checks and reductions; decide software acceptance, integration and operational use separately. Perform any GitHub push. |

The design chat should use canonical `codex/refactor-mainline`, which now
contains the accepted foundation, for its design documents. The isolated
`codex/point-policy-foundation-20261106` checkout remains preserved at the exact
tested application. Coordinate checkout ownership for each subsequent increment;
do not edit the other chat's checkout or disturb its changes. No automatic
cross-chat messages or unattended rule adoption.

Design and code can progress separately. Return a full design commit SHA with
the owner-approved rule identity, decision-table changes, unresolved items and
concrete input/expected-output cases. The implementation chat will bind that
revision in its next work order. A report-schema version identifies the shape
and meaning of data; it is not scientific policy approval.

## Available foundation

The final product publisher constructs `point-assessment.json` from the map
records and final fit guards already in memory. It runs once per observation
publication, in ordinary and detective modes, including when fits are disabled.
It does not introduce another fit, map read, per-pass calculation or file-based
handoff. The existing fits and pointing table retain their behavior.

The pure adapter is `configured_point_assessment.build_assessment`. Its inputs
are the observation number, inspected map records, final guards keyed by array,
and the existing pointing-table result. The publication wrapper writes one
required small JSON file in the existing staging directory. Its SHA-256 binds
the same serialized bytes; no reread is used to construct this binding.
Each per-array product sidecar and its `RUN.json` entry carry the filename,
schema and digest. Required write errors prevent the final observation rename.

Schema: **`citlali-configured-point-assessment-v1`**.

| Field | Meaning and boundary |
| --- | --- |
| `observation` | The map publication's integer observation number. |
| `frame`, `measurement_axes`, `measurement_units` | Source-offset AltAz, ordered tangent azimuth/elevation, arcseconds. Measurements are signed displacements, not corrections. |
| `parameter_names`, `parameter_units` | Existing Gaussian-plus-constant vector: amplitude, cx, cy, sigma_x, sigma_y, angle, constant. Amplitude/constant are mJy per nominal beam; centroid and sigmas are arcseconds; angle is radians. Sigmas are the fitted rotated-axis widths, not sorted major/minor FWHM. |
| `arrays` | Fixed order a1100, a1400, a2000. Each entry has its own array identity. |
| `arrays[].source` | Available inspected map filename/digest, native bundle/product identity, nominal beam and, for empirical products, coefficient family, cycle, health and plan identities. These are references to existing authorities. |
| `arrays[].map_status` | `published` means a required map in this staged publication; `not_produced` means this publication did not request that array. It does not establish a declared-offline reason or bypass a missing required map. |
| `arrays[].fit_status` | `accepted`, `failed`, `unavailable`, `not_requested`, or `not_produced`. Existing producer states stay distinct. |
| `arrays[].fit_guard` | Existing method identity, status, accepted/complete flags, cause, accepted parameters/centroid, and conditional sigma_A/SNR_A. No raw rejected fit is promoted into an accepted measurement. Missing/nonfinite diagnostic scalars are JSON null, matching existing fit diagnostics; malformed accepted measurements fail publication. |
| `arrays[].fit_diagnostic` | Relative filename of the full existing fit diagnostic, if produced. The report snapshots its listed guard facts; this filename is a locator, not a separate hash binding of the full diagnostic. |
| `positional_uncertainty_status` | Unavailable. Conditional amplitude sigma_A is not an endpoint, positional or calibration uncertainty. |
| `existing_table` | Snapshot of the unchanged per-array table result, including its own eligibility, withholding cause, file/digest and source products. Its eligibility is not a new rule-evaluation outcome. |
| `rule_evaluation` | Always `not_evaluated`, with cause `rule implementation pending`, null policy identity and adopted displacement, and an empty selected-array list. This is an explicit future boundary, not an active rule engine. |
| `calibration_eligible` | False. |

No consumer should treat the foundation's `rule_evaluation` as a selected
pointing solution. The installed TolTECA consumer continues its established
table route. No consumer, telescope correction or deployment change is included.

## Design work to return

Start from the owner-approved decision table. Preserve its agreed array
agreement and override cases, diagnostics, explicit unavailable errors and
independent accuracy qualification. Reconcile only unresolved definitions or
new owner decisions; do not infer approval from a proposal or implementation.

The design response should specify:

1. Which measurements define a usable array, and how unassessable a1100 shape,
   unavailable fits and declared offline arrays affect each rule.
2. Background pixel membership, support/coverage denominators, quadrant
   boundaries, RMS/robust-scatter definitions, and zero/unavailable cases.
3. Policy identity, configurable versus fixed settings, requested/effective
   reporting and any permitted Learn–Consider–Apply changes.
4. Selected/withheld output, participating arrays, reasons and warnings for
   each decision-table case; include exact boundaries and ambiguous cases.
5. The consumer agreement: measurement versus correction, authoritative
   selection, legacy products, malformed records and withholding behavior.
6. What evidence establishes each claim, including what the 1.5 arcsec goal
   means statistically and which independent data are required.

The current foundation has no annular background, adopted selection, pairwise
agreement or shape-override calculation. It does not enable offline-array
reduction. Those fields and route changes will be a subsequent approved
increment. Schema v1 can serve as the concrete available-input vocabulary;
designs may request additional fields with their scientific definitions.

## Verification and handoff status

The [work order](../validation/point_policy_foundation_2026-11-06/PLAN.md)
records authority, gates and scope. Focused tests exercise conservation of
measurements, state distinctions, immutable input ownership, malformed accepted
data, exact output binding, no overwrite, normal publication and required-output
failure through the CLI. Existing worker tests cover final-fit equivalence;
packaging checks cover the new module in bundles and wheels.

Grant's software checks passed: 372 configured Python/package tests, 1,449
enabled CTests, 207 baseline tests, 130 config unit tests, all four mode kits
and eight required compact cases. One pre-existing disabled CTest is excluded.
Independent exact-source and prepared-packet static reviews found no actionable
issue; neither review executed runtime commands.

Grant then completed the ordinary six-worker 152389 M0–M5 pair against accepted
application `fc9c1f2935b433782ae3cfa1a79e2ac0247dd8f1`. All 11 comparison checks
passed. Detector reports/decisions, map and feedback planes, final fits and the
existing table's scientific values/metadata matched. The new 7,582-byte report
matches the final facts and retains `not_evaluated` with no selected arrays or
adopted displacement. Command times were 61.06 and 54.07 seconds; sampled summed
peak RSS was 5.99 and 5.86 GiB. This pair showed no performance regression; it
does not establish a causal speedup from adding the report.

The acceptance receipt binds the preserved external packet, review records and
owner-produced evidence. Mainline contains the exact application followed by
documentation-only acceptance updates. No GitHub push occurred in this action.
The foundation is ready for design use. Return the exact owner-approved rule
revision and decision cases before a subsequent implementation increment;
accuracy, SCIENCE/coadd, Unity/V2 and deployment qualification remain separate.
