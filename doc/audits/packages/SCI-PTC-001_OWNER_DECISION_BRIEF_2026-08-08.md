# SCI-PTC-001 owner decision brief

- Date: 2026-08-08
- Role: independent package auditor
- Application SHA: `46ad23888a40f5102cdfd50c06e49a549bdf8a20`
- Audit branch: `codex/audit-sci-ptc-001`

## Decision requested

Retain package identity `SCI-PTC-001`, accept verdict `amend`, and preserve
`existing_use_only` while the scientific owners approve or supersede D001–D006.
No decision has been taken by this audit. Approval of this brief would not by
itself authorize repair, validation, integration, re-audit, a downstream
launch, or production expansion.

The proposed independent axes are:

| Axis | Proposed value |
| --- | --- |
| Contract status | `proposed` |
| Implementation status | `nonconformant` |
| Validation status | `in_progress` |
| Production status | `existing_use_only` |
| Verdict | `amend` |

## Why owner action is required

At the exact governing SHA, four P0 implementation defects are established by
source and deterministic formula evidence:

1. `SCI-PTC-001-F001`: masked NaN/Inf arithmetic, null signal/mask mismatch,
   and fallback admission can put invalid payloads into fitted state and
   unflagged descendants.
2. `SCI-PTC-001-F002`: second-pass, post-clean detector, and weight exclusions
   occur after cross-detector mixing without recomputation or transitive
   invalidation.
3. `SCI-PTC-001-F003`: downsampled PTC and PTC-diagnostic products publish the
   native `SAMPRATE`, not the effective rate used by cleaning and maps.
4. `SCI-PTC-001-F004`: later variable-length processed chunks receive incorrect
   inclusive scan bounds. ALIGN repair `5c630912...` is not in the governing
   ancestry and was not treated as integrated or Unity-validated.

Two further P0 dependency gaps remain: incomplete RTC causal support/response
(`F010`) and unresolved AST/ALIGN coordinate-validity/detector binding
(`F012`). P1 findings cover conditioned-only response (`F005`), nonprecision
coefficients and absent covariance (`F006`), incomplete four-state provenance
(`F007`), mutable/non-atomic products (`F008`), disabled-clean semantics
(`F009`), CAL factor/covariance authority (`F011`), and missing direct PTC
validation (`F013`).

## Decisions D001–D006

### SCI-PTC-001-D001 — disabled-clean semantics

Question: Does disabling PTC mean exact identity, or is source-mask
mean-centering a mandatory independent stage?

Recommended disposition: make all-disabled PTC exact identity. If centering is
scientifically required, expose and name it separately in requested,
effective, observation-resolved, and realized state.

### SCI-PTC-001-D002 — direct cause and transitive influence

Question: What durable representation distinguishes direct validity causes
from transitive influence, and when must a late exclusion trigger recomputation
instead of explicit descendant ineligibility?

Recommended disposition: require cause-preserving, fail-closed eligibility for
signal, kernel, coefficients, and every consumer. The owner may choose the
representation and recompute/invalidate strategy, but it must be exact and
falsifiable.

### SCI-PTC-001-D003 — response product families

Question: Which response classes are promised: fixed-state conditioned
operator, realized local Jacobian, global/extended transfer, response
uncertainty, and explicit unavailable state?

Recommended disposition: publish only the class actually computed and bound to
its exact realized state and upstream response. Mark every stronger class
explicitly unavailable. Do not infer a global or beam response from the
current partial conditioned kernel.

### SCI-PTC-001-D004 — coefficient, precision, and covariance families

Question: What are the exact factor identity, units, normalization scope,
lifecycle, marginal-precision conditions, and retained covariance for each
coefficient family?

Recommended disposition: type all current approximate, full, hybrid,
validated, constant, correlation-penalized, and busy-row values as
nonprecision coefficients unless the complete precision conditions are proved.
Do not derive significance or independent-noise authority from map
denominators or the current NetCDF unit attribute.

### SCI-PTC-001-D005 — immutable product and state bundle

Question: What are the canonical full, mini, diagnostic, simulated, and
processed identities; scan-specific APT/state representation; parent/digest
links; append atomicity/completeness rule; and four-state replay requirement?

Recommended disposition: require an immutable complete-bundle marker and
scan-specific realized state sufficient to deserialize and replay every group,
mask, selector, pass, coefficient, learned/random state, response/covariance
status, and product link.

### SCI-PTC-001-D006 — missing-data, fallback, and null policy

Question: What eligible-only arithmetic, non-finite behavior, source-mask and
fallback semantics, coupled surrogate signal/mask shift, random seed/shift
persistence, and selection uncertainty are required?

Recommended disposition: reject invalid payloads before arithmetic, shift
surrogate signal and validity together, persist realized random/selection
state, and fail closed when eligible support is insufficient.

## Required dependency order

1. Record the applicable owner choices D001–D006 as versioned scientific
   authority.
2. Obtain accepted exact successors for RTC and the CAL/AST/ALIGN interfaces
   on one integrated application line; do not treat active-task results or
   ALIGN `5c630912...` as already integrated.
3. Authorize a separate bounded PTC repair only after the relevant decisions.
4. Run focused exact-successor fixtures for validity/influence, response,
   weights/covariance, rate/extents, four-state replay, atomic products,
   sequential/OpenMP, and simulation parity. No required-data skip and no
   unexpected error-level message may count as a pass.
5. Perform a fresh independent PTC re-audit before any consumer or production
   expansion.

Broad/costly work, a local Citlali reduction, Unity execution, and external
evidence remain outside this audit and require separate authorization.

## Interim allowlist and restrictions

Allowed: only continued behavior already authorized by owning integrated
package records and narrow static/document checks that do not imply overall
conformance.

Fail closed on any new claim of precision, inverse variance, significance,
independent noise, full covariance, response/transfer/beam completeness,
response uncertainty, causal eligibility, coordinate-valid source masking,
calibration-factor authority, immutable product reconstruction, or equivalence
among RTC input/full/mini/diagnostic/simulated/processed/map/beam products.

VAL, MAP, NOI, and BEAM retain their current restrictions. The bounded outgoing
records are proposed and unintegrated:

- `SCI-VAL-001-XAUD-008` carries validity/influence and later BEAM-response
  facts; it does not launch VAL or BEAM.
- `SCI-MAP-001-XAUD-004` carries the nonprecision coefficient and retained
  covariance/response restrictions.
- `SCI-NOI-001-XAUD-001` records that signed coefficient-weighted cleaned data
  are not independent draws, a covariance model, or calibrated significance.

## Exact review artifacts

| Artifact | SHA-256 |
| --- | --- |
| `doc/audits/packages/SCI-PTC-001_INDEPENDENT_CORE.tex` | `82c0835f51ea9b1fa8a37489f289be89a8018a0b2700e84b1e25c2e4d2a013c2` |
| `doc/audits/packages/SCI-PTC-001_SCIENTIFIC_CONTRACT_AUDIT.tex` | `c46a15c142d0938baf9576d84a19332e0d46b34852b4d59c0029ba00ac62d7e6` |
| `doc/audits/evidence/SCI-PTC-001_LOCAL_EVIDENCE_2026-08-08.yaml` | `091059abd088b8bca58ca5a885e12620972c1f75f75574e33bfff8b0eb90b195` |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-008.yaml` | `fdeaa3d18909a35b3caff85257f70e7f51ae6115ec07d172cbe96fd1b5007a32` |
| `doc/audits/handoffs/SCI-MAP-001/SCI-MAP-001-XAUD-004.yaml` | `5c5221366d9fd66cffc3881cb8fad2f9b1fee990bfd581b4583cf0d1b72c53d2` |
| `doc/audits/handoffs/SCI-NOI-001/SCI-NOI-001-XAUD-001.yaml` | `eb9c8588e0c09d8a9882ed076ee0b8cc33ccad9c49319dea0b56e4419eee3c8c` |

Final commit, parent, tree, this brief's digest, and the ledger-proposal digest
are supplied in the final Git return rather than self-embedded.
