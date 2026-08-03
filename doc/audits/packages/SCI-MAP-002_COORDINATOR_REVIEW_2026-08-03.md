# SCI-MAP-002 coordinator review — 2026-08-03

Status: audit package verified and integrated; owner scientific decisions
required before repair or any exact-SHA evidence request

Package: `SCI-MAP-002` — JINC gridding interface, normalization, and response

## Verification and integration

The audit source commit `abb8d8896d6a1cbaa912b9ac181bd649588acc62` was
verified clean, against governing application SHA
`9aae0e669384c5c0c0dda93debc194d6b8dac787`. It contains only the three
declared audit artifacts; it makes no application change and records no build,
test, reduction, Unity request, Unity access, or external evidence.

The coordinator integrated that commit without modification as
`d313c879b`, then applied this review and the accepted machine-readable
proposal in the succeeding coordination commit. The verified artifacts are:

| Artifact | SHA-256 |
| --- | --- |
| `SCI-MAP-002_SCIENTIFIC_CONTRACT_AUDIT.tex` | `d77bb5c8e555b43d5303ad2ce0a81e5baef42df2beb85347f9f98cab759d5239` |
| `SCI-MAP-002_LOCAL_EVIDENCE_2026-08-03.yaml` | `3a10f410dcb681a41bb2dee42b5ef1141a93a0de1362ce829dff0afa297855b8` |
| `SCI-MAP-002_LEDGER_PROPOSAL_2026-08-03.yaml` | `93eb40e598e2ac417c0f966843aa770730658984f2a14c4249bc99403be77881` |

The task's final chat message transcribed the first digest incorrectly. The
table, the audit proposal, and this review use the independently recomputed
digest above.

The independent core, phase-two trace, and the post-core MAP handoff were
used in their intended roles. The handoff remains naive-only context and
establishes no JINC behavior.

## Coordinator disposition

The audit's proposed status is accepted as the coordination state:

- contract: proposed, not owner-approved;
- implementation: nonconformant to the frozen proposed contract;
- validation: incomplete; and
- production: existing use only.

No repair, parameter campaign, Unity request, or re-audit is authorized by
this review.

### Findings accepted into the coordination ledger

- **F001 (P0):** `coverage_bool` can mark zero or non-finite formal weight as
  valid. It is not an authoritative scientific validity mask.
- **F002--F003 (P1):** the realized support is a square dual-use `r_max`
  cache, and subpixel refinement is phase-quantized point sampling; neither
  matches the independent radial/pixel-average reference by default.
- **F004--F005 (P1):** fixed absolute cancellation/floor thresholds have
  implicit units, and shape/coefficient admission is incomplete.
- **F006--F007 (P1):** coverage is coefficient-squared time rather than a
  declared geometric exposure, and realized JINC/product provenance is
  incomplete.
- **F008 (P1):** no exact-SHA JINC response or numerical
  sequential/concurrent evidence exists. The proposed U01--U06 protocol is
  retained as a proposal only.

The audit also establishes positive constraints for any successor work:
preserve distinct raw `N`, signed `C`, and `Q`; the observed `N/C` signal and
conditional `C^2/Q` formal-weight finalization; signed-lobe treatment;
stable array identity; and the distinction between formal and empirical
working weights. The preliminary detector-parallel overlapping-write concern
is discharged for the traced one-to-one detector-to-map Beammap caller, but
not numerical reproducibility under runtime-dependent merge order.

## Decision routing

`SCI-MAP-002-D002` is accepted as an immediate restriction: do not use
`coverage_bool_*` as an authoritative validity mask for new validation or
expanded scientific claims. This changes neither historical output nor
application behavior.

`SCI-MAP-002-D001` and `D004` are retained as proposed mathematical and
positive-control constraints, pending the coherent owner contract below.
`SCI-MAP-002-D005` remains an unrequested evidence prerequisite.

The owner decision boundary is `SCI-MAP-002-D003`. Before repair can be
scoped, the owner must define a coherent JINC scientific contract for:

1. square/dual-use-`r_max` support versus a radial support definition;
2. point-phase versus pixel-integrated subpixel response;
3. unit-invariant cancellation/conditioning semantics;
4. physical parameter domain and coefficient-finiteness admission;
5. coverage estimator meaning, mask authority, and kernel-amplitude identity;
   and
6. requested/effective/resolved/realized provenance required to join those
   products to downstream noise and filtering consumers.

The coordinator will present these decisions one at a time, with the observed
behavior and a bounded recommendation. No answer in this record authorizes a
numerical parameter search.

## Validation note

YAML parsing, artifact hashes, diff/whitespace checks, and LaTeX source
structural checks passed. Local LaTeX rendering was unavailable because the
installed Tectonic client panicked before document parsing; this is a local
tool failure, not evidence for or against the audit conclusions.
