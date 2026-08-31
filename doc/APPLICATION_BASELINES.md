# Citlali Application Baselines

This is the short answer to “which Citlali?” It names durable application
landmarks without turning branch names, validation directories, scientific
contracts, or work programs into product versions.

This document is a router, not a second status ledger. Current sequencing is
in [`REFACTOR_STATUS.md`](REFACTOR_STATUS.md), concurrent workstream authority
is in [`INTEGRATION_LEDGER.md`](INTEGRATION_LEDGER.md), scientific contract
status is owned by the scientific-contract library index, and executable
validation acceptance remains in `validation/accepted_runs.json` and
`validation/validation_profiles.json`.

## Four Independent Status Axes

Every application landmark is described on four axes. A positive result on
one axis does not imply a positive result on another.

| Axis | Question |
| --- | --- |
| Application integration | Is this code assembled on the canonical application lineage? |
| Scientific conformance | Which approved scientific contract, if any, does it conform to? |
| Validation evidence | What exact builds, datasets, products, and comparisons were actually run? |
| Production authorization | What operational use has the owner explicitly authorized? |

## Named Landmarks

### Legacy Fork Comparator

- Exact commit: `376e002238b1f49aeced8a3f33e8742db141634b`
- Historical role: the original `gw_dev` snapshot from which the structural
  refactor began.
- Application integration: historical comparator only.
- Scientific conformance: carries the legacy behavior present at the fork; it
  is not labeled retroactively with later scientific contracts.
- Validation evidence: the starting side of the refactor comparison program.
- Production authorization: none created by this designation.

### Legacy Preserved `gw_dev` Head

- Exact commit: `ffc6b9070f4744f9778f3db71cdc468846d1da89`
- Preserved checkout: `/Users/gwilson/GitHub/citlali`
- Historical role: the later legacy-development comparator, including work
  that continued after the refactor fork.
- Application integration: separate legacy lineage, not canonical refactor
  ancestry.
- Scientific conformance: historical behavior only; no retroactive WP-7.1
  claim.
- Validation evidence: used by named OG/refactor comparisons where the
  relevant records explicitly cite it.
- Production authorization: unchanged by this designation.

### Structural Refactor Baseline

- Exact application commit:
  `cee74ecbdfb4187756183879163a22ca2b8518f6`
- Exact validation-record commit:
  `71b3fd3d33b5b8ff236ea5ceff616ffa199d9208`
- Application integration: structural closeout of the initial architecture
  refactor.
- Scientific conformance: behavior-preserving refactor scope; no later
  WP-7.1 conformance is implied.
- Validation evidence: the deterministic pointing closeout compared 29-file
  manifests and 21 FITS image HDUs exactly, with NaNs treated as equal. That
  bounded result must not be expanded into a claim that every mode or every
  byte was compared.
- Production authorization: historical validated baseline, not a current
  release authorization.

### Native Integration Baseline

- Exact integrated application commit:
  `f0f423827ab321640e0cbcb003f7bf015368f694`
- Annotated tag: `wp7-native-memory-integration-20260830`
- Important predecessor milestones:
  - exact Stage 7 science identity
    `3ebc2a67fc32bad69759ff45638484efabf91773`;
  - pre-build-modernization integration checkpoint
    `a36abaebfb82d503b113de0cf4c1c6e0f6dcffc3`;
  - exact Unity-tested native-memory repair
    `187df04b21e942701cf41e6d9c50883922fd65aa`.
- Application integration: complete on canonical application ancestry.
- Scientific conformance: substantial bounded scientific-contract work is
  present, but this application is explicitly **not WP-7.1-conformant**. The
  later WP-7.1 contract closure is a separate authority.
- Validation evidence: substantial local and Unity evidence exists, including
  the Stage 7 and native-memory campaigns. The completed V2 four-mode campaign
  is mixed-SHA rather than one accepted same-SHA successor matrix.
- Production authorization: `existing_use_only`; no production expansion or
  Phase 5 release promotion follows from the integration.

### WP-7.1 Timestream Contract Baseline

This is a scientific-contract landmark, not an application build.

- Exact successor source commit:
  `170ecea9de1ee810da7d7e45a489a4545ccd623d`
- Exact scientific closure commit:
  `20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa`
- Scientific status: contract-closed for the approved bounded WP-7.1 scope;
  zero regressions, recurrent findings, new successor findings, or unresolved
  contract contradictions were reported by the locked comparison.
- Implementation status: separate. Contract closure does not establish
  application conformance, observational validation, performance, readiness,
  or production use.

### WP-7.1 Timestream Successor Program

This is the active application-development program that will implement and
validate the WP-7.1 contract on canonical ancestry. It is not the name of a
completed Citlali product or release.

- Canonical reconciliation branch: `codex/wp7-governance-reconciliation`
- Exact canonical base:
  `cb3d568c701217ee0248c77f6dccd0bab7deef31`
- Preserved divergent evidence/tooling head:
  `49fe73e757daa1885cd23127e8441cba47e648d2`
- Current phase: governance, authority, and ancestry reconciliation. Further
  application implementation is held until the gates in
  [`WP7_TIMESTREAM_SUCCESSOR_PROGRAM.md`](WP7_TIMESTREAM_SUCCESSOR_PROGRAM.md)
  pass.

## Validation Corpus And Campaign Names

`citlali-validation/v2` names a validation-corpus revision. It is not an
application version and does not identify one source SHA.

The August native closeout is a campaign using that corpus. Point, OOF, and
Beammap evidence was produced at `c31a60a0b74a7149d03d542966d6e35b77b8091c`;
science evidence was produced at
`187df04b21e942701cf41e6d9c50883922fd65aa`. It is therefore intentionally
described as mixed-SHA operational evidence, not relabeled as a same-SHA
application baseline.

## Naming Rule

Use descriptive names plus exact immutable identities. Do not invent
retroactive release numbers for these landmarks. New release/version names are
assigned only when a release candidate has one source identity, its required
scientific contract disposition, its accepted validation matrix, and an
explicit production authorization.
