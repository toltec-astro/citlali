# SCI-FLT-001 returned-evidence coordinator review — 2026-08-01

## Identity and scope

The coordinator reviewed the locally returned, human-run point-observation
bundle for the convolve candidate built from source commit
`b294802a5e339f9ba5e0e323980cec3a4bd00249`.

- Evidence ID: `SCI-FLT-001-UNITY-001-PARTIAL`.
- Observation/case: pointing observation 152389, control and candidate.
- Local evidence root:
  `/Users/gwilson/work_toltec/local_data/citlali-validation/v1/evidence/convolve_contract_b294802`.
- Logged candidate version: `v4.0.0-3631-gb294802a`.
- Recorded executable SHA-256:
  `c199b514efbe2c19b7fd785f7a49e5fa912a5dcf95208513255671c21802fa49`.
- The executable snapshot bytes are not present in the returned local bundle,
  so the recorded executable digest could not be independently rehashed. A
  clean source-state record, compiler/CMake identity, and complete dependency
  identity were also not supplied.

The following evidence artifacts were verified locally:

| Artifact | SHA-256 |
| --- | --- |
| `candidate-audit.json` | `3f5fd759e1c98ed7f844c9abaa9a6612191c118809b69de50991b4ec4a8a7fe2` |
| `candidate-audit.md` | `0d5a9cc854ee70ad017b260c55745995857adca155fbfbd706a73e74f81d67a8` |
| `control-audit.json` | `8fd9269a330c987e4d8aa859c870b39bd1e808423ad87ff72285377e37a74dad` |
| `control-audit.md` | `a33094643d1aebef25f5628d0f75aef6df97eefd95f35bca1b32fa6687fba07d` |
| `raw-comparison.json` | `39ddb6d7fcaf976c3dd6c9ce726aa7f92f552cb979d01ee3c2f2e62025e4e556` |
| `raw-comparison.md` | `ef7827cec5001d20e9442ecdca0a02285d583bebfa29aa15e2b545b7590afa8c` |
| `citlali.sha256` | `be5968bdc19007f7337b7a5a3601738828b41dce3c040be72e3423792e75d01c` |
| `tolproj-unity.yaml` | `609934e5c81b9a9afaf17c9564c26fd07011e72bdc919a2590ff8576248b7e30` |

## Implementation-level results accepted

- Both reductions completed with no unexpected logged issue.
- Thirteen required candidate and thirteen required control provenance
  sidecars were valid and hash-matched.
- The strict raw comparison covered 15 common products and 1,960 records with
  zero changed, skipped, missing, or extra records. This is record-level
  equality, not byte-identical whole FITS files.
- Required filtered metadata was present for all three arrays.
- The convolved-signal compatibility aliases were exact.
- The stored ten-realization `N-1` variance was independently reproduced to
  floating-point roundoff; maximum relative differences were approximately
  `7.92e-16`, `9.90e-16`, and `9.30e-16` for a1100, a1400, and a2000.
- Direct uncertainty and direct S/N identities were exactly reproducible.
- Candidate metadata correctly kept feedback closed with `FLFBACK=false` and
  `FLWHY=support_contract_unresolved`.

These results support implementation/regression characterization only. They
do not approve the filtered uncertainty, significance, response, support, or
photometric contracts.

## Adverse scientific-acceptance result

The review reconstructed:

```text
science_mask = raw_weight_I >= runtime_weight_threshold
               AND raw_coverage_I >= runtime_hits_threshold
support_mask = filtered_weight_formal_I > 0
guard_mask = support_mask AND NOT science_mask
```

Using `sig2noise_point_source_I` as the direct empirical S/N statistic:

| Array | Guard pixels | Total pixels with direct abs(S/N) > 5 | Of those in guard |
| --- | ---: | ---: | ---: |
| a1100 | 7,625 | 1 | 0 |
| a1400 | 11,436 | 1 | 0 |
| a2000 | 17,296 | 9 | 8 |

The extreme a2000 guard pixel is NumPy zero-based `(row=89, col=18)`, or FITS
one-based `(x=19, y=90)`:

- direct S/N: `-165.71731086416185`;
- `coverage_bool_I`: true;
- signal: `-0.22962288858136468 mJy/beam`;
- direct uncertainty: `0.0013856300671544575 mJy/beam`;
- noise variance: `1.919970683002466e-06 (mJy/beam)^2`;
- raw weight: `0.003048213704414995`, below threshold
  `0.00319913795615698`; and
- raw coverage: `1.5272387563175867 s`, above threshold
  `1.50695758396 s`.

The result is not an arithmetic reader error: the direct uncertainty and S/N
identities close. It demonstrates that a ten-realization pixelwise empirical
variance can collapse in the a2000 numerical guard band while the filtered
`coverage_bool_I` admits that pixel. Therefore direct empirical S/N and that
coverage plane are not scientifically acceptable as filtered significance,
science confidence, support eligibility, or feedback eligibility.

## Missing closure evidence

The returned bundle does not supply:

- clean source state and complete compiler/build/dependency identity;
- executable snapshot bytes needed to verify the recorded executable digest;
- effective expected-mode and expected-label gates;
- an independent exact squared-coefficient formal-variance table;
- blank-field covariance, correlation, false-S/N, and edge-distance analyses;
- compact, beam-shaped, resolved, and extended injection/recovery;
- a median-fill covariance bound;
- approved response definition and normalization;
- an approved science-support/confidence plane and floor;
- a downstream multi-pixel covariance contract; or
- preregistered scientific acceptance tolerances.

## Coordinator disposition

This is a **reviewed partial return with failed and incomplete scientific
acceptance**. It does not replace the historical MAP cross-audit transcript or
complete `SCI-FLT-001-UNITY-001`.

`SCI-FLT-001` remains contract `proposed`, implementation
`conditionally_conformant` only under its narrow numerical assumptions,
validation `in_progress`, production `fail_closed`, verdict `amend`, and
re-audit `required`. Re-audit closure remains blocked by:

- the adverse a2000 guard-band significance/support result;
- incomplete same-SHA build and scientific return identity;
- unresolved findings F004--F009 and their owner decisions/evidence gates; and
- open `SCI-MAP-001`, `SCI-NOI-002`, and `SCI-CAL-001` dependencies.

Do not integrate the mixed-owner candidate unchanged. Keep filtered
fruit-loop amplitude, support, confidence, and gain fail-closed. The next
scientific sequence remains MAP/VAL/NOI contract closure, followed by an
owner-approved FLT response/support/photometry amendment, a bounded owner-
separated successor, complete exact-SHA evidence, and fresh re-audit.
