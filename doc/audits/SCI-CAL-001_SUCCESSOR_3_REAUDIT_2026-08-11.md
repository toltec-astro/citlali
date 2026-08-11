# SCI-CAL-001 successor-3 independent re-audit

Date: 2026-08-11

Auditor role: fresh, role-separated independent technical auditor

Candidate ref: `codex/repair-sci-cal-001-successor-3`

Candidate commit: `3af6faf996fa002b2647adca8f33991002d49ff1`

Candidate parent: `8b1534807f5abe4d80be2fbd45ed3838ed351509`

Candidate tree: `16130eb6deba3f9d8b5a8f1d1fae126084b63c95`

Parent-to-candidate binary patch SHA-256:
`4558d541a82b2a1f5c4406825c277a2f7317b6d4c788f4bbaf699a385d471bdf`

## Executive disposition

The candidate is **not a complete conforming successor-3 repair of
SCI-CAL-001**. The recommended verdict is **amend** and return for bounded
technical repair. No new scientific decision is required to resolve the
observed defects: they concern completeness of the already-authorized
recipient proof, canonical identity, product linkage, and validation-contract
synchronization.

| Axis | Proposed state | Basis |
|---|---|---|
| Contract | `approved` | The accepted successor-3 finding scope and prior owner conditions remain the authority. |
| Implementation | `nonconformant` | F005, F007, F008, and local F009 retain material implementation or evidence gaps. |
| Validation | `in_progress` | Every runnable local gate passed, but those gates do not exercise or accept the new v4/package surface completely. |
| Production | `fail_closed` | No production or downstream authorization is proposed. |

The audit positively confirms F003 and F004. F002 and F006 remain closed only
within their previously accepted bounds. F001 and F010 remain conditioned and
are not promoted.

## Entry, identity, and bounded scope

Before candidate source, diff, or test exposure, the mandatory READY
checkpoint independently established and the coordinator accepted:

- live origin and local `HEAD` both resolved to the exact candidate;
- the sole parent, tree, and binary patch digest matched the immutable values
  above;
- the worktree and index were clean and role-separated from repair task
  `019ff115-f03d-75f0-81d8-88b5f165669c`;
- the candidate changed exactly 18 authorized paths;
- the audit branch was absent locally, in remote-tracking state, and at live
  origin; and
- the two planned audit artifacts were documentation-only.

After approval, the audit branch
`codex/reaudit-sci-cal-001-successor-3-20260811` was created directly at the
candidate. The exact candidate diff, relevant source, tests, schemas,
product/profile contracts, and documentation were then inspected. Candidate
claims were treated as claims until independently reproduced.

No application, configuration, test, build-system, validation-product, or
canonical-coordination file was edited. No Citlali reduction, Unity access or
request, push, merge, external coordination, downstream launch, or production
authorization occurred. The candidate does not change the inherited RTC, PTC,
MAP, Beammap, Pointing, Lali, kernel, or filtering implementation files.

## Independent findings

### F001 — open, conditioned and unchanged

Successor-3 supplies no authorized SCI-ALIGN, SCI-AST, exact-SHA Unity, or
astronomical-standard evidence. Its local factor and response-provenance work
does not discharge those conditions. F001 remains open and conditioned without
broadening.

### F002 — retain narrow closure, unchanged

The fixed-DJF25 contract, nodes, generated header, and production operator are
byte-identical to the parent. Independent SHA-256 digests are:

| Artifact | SHA-256 |
|---|---|
| `data/calibration/sci_cal_001_fixed_djf25_full_domain_operator_contract.json` | `7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a` |
| `data/calibration/sci_cal_001_fixed_djf25_full_domain_operator_nodes.csv` | `fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f` |
| `include/citlali/core/timestream/atmosphere_operator_nodes_generated.h` | `d322bdc863ccb1292325c739865f772ef53f4e9f4101967752027ea0a2413262` |
| `include/citlali/core/timestream/atmosphere_operator.h` | `3fd4352d05e77e07c1e354b7e4124733505064667968676d8d4e94315017d584` |

The generator check reconstructed 1,368 rows and 72 series exactly. This
retains only the accepted structural closure; it makes no atmospheric-model
fidelity, uncertainty, or observational-truth claim.

### F003 — conformant

`ReductionConfig::validate` rejects an unsupported requested calibration unit
when raw flux calibration is enabled. The typed validation is invoked by
`Engine::get_citlali_config`; both CLI session paths return failure before the
output-root lease, observation/APT analysis, reduction pipeline, or product
publication. An uncalibrated request is deliberately preserved.

The two focused startup tests pass:

- `config_scaffold.rejects_unsupported_calibrated_unit_but_preserves_uncalibrated_request`;
- `cli_reduction_runtime.rejects_unsupported_calibrated_unit_before_observation_or_output_work`.

The CLI fixture uses a bounded fake engine, but static production wiring
independently confirms the same fail-before-work boundary. F003 is closed for
the accepted successor-3 requirement.

### F004 — conformant within the accepted association claim

The candidate preserves truthful legacy APT lineage and adds a separately
labelled optional modern TolAPT association. It verifies an exact sibling
`manifest.yaml`, `tolapt.run.v1`, structured design/measured input records,
safe output paths, exactly one resolved selected output, and a digest binding
the manifest, run/input records, output key/path, and selected APT digest.
Modern-looking fields do not become authority when that complete association
is absent or invalid.

The selected ECSV digest, source-row indices, raw acquisition join, retained
typed fields, eligibility, and validity are carried explicitly. The verified
association is structural and does not authenticate the producer, rehash the
manifest-declared design/measured inputs, or certify run QC or scientific
validity; successor-3 does not claim otherwise. F004 is closed within this
explicit boundary.

### F005 — partially addressed, remains open

Static tracing confirms that the inherited numerical routes are algebraically
scale-covariant and apply the response basis once:

- RTC signal calibration applies `target_unit_factor * flxscale` once and the
  per-sample extinction once;
- approximate weight uses selected-APT `sens` through the compatibility FCF
  without a second flxscale;
- full weight is inverse variance of already-calibrated samples; and
- naive map signal and weight use the inherited calibrated inputs without a
  second CAL pass.

The new deterministic tests cover approximate and hybrid weighting, full
weight, once-only signal factors, and a naive normalized map. They do not,
however, prove every existing recipient route:

1. `validated` is a distinct active production weighting mode used by the
   Pointing and OOF configurations and shares the approximate-baseline branch,
   but the new test loop includes only `approximate` and `hybrid`.
   `weight_recipient_semantics` likewise says `approximate_or_hybrid` and omits
   `validated`.
2. The required conditional `noise_variance_I` product is formed from
   normalized noise realizations and should scale as `a^2` by static algebra,
   but the new naive-map fixture sets `n_noise=0`. The direct
   `variance_prime=a^2*variance` assertion exercises only a scalar helper, not
   that production recipient.

No numerical counterexample was found and no recipient redesign is indicated.
The accepted exactly-once proof and truthful recipient inventory are still
incomplete, so F005 remains open.

### F006 — retain bounded closure, unchanged

The production calibration boundary remains `mJy/beam` only, now with the
earlier configuration rejection required by F003. The relevant production
section is unchanged from the parent (SHA-256
`ff6b3ae56dff8a4780b60c974945edf121d991f364b782c82439f476909c46bc`).
This remains only the accepted target-unit policy; it does not certify response
fidelity, total uncertainty, or astronomical truth.

### F007 — partially addressed, remains open

The package writer is a substantive improvement. It defines separate
calibration and package identities, publishes an exact required
`selected_calibration_apt.ecsv` copy plus YAML lineage, checks source and copy
digests, reopens the YAML, rolls back the copy on YAML failure, and rejects
stale or conflicting state.

The purported full canonical identity is nevertheless not collision-resistant
over the complete applied state:

- admitted factor identity binds only target-unit factor, detector flxscale,
  and per-detector minimum/maximum extinction. Different sample-elevation
  sequences with the same extrema have the same factor identity; the separate
  acquisition binding covers raw KIDs files, not the telescope elevation
  source;
- response identity omits active FIR `a_gibbs`, fixed-notch zero-phase state,
  center frequencies and widths, and applied dynamic line-audit notches; and
- configured response values are labelled `realized_*` even when the
  corresponding stage is disabled.

Moreover, the actual map FITS, TOD NetCDF, and Beammap ECSV metadata population
paths carry APT and acquisition-binding digests but neither
`calibration_identity` nor `package_identity`. Because the candidate's own
test shows that calibration identity may change while those older component
digests remain fixed, an individual product cannot uniquely resolve its
canonical package. The Beammap test manually injects a calibration identity
instead of exercising production metadata population. F007 remains open.

### F008 — partially addressed, remains open

The candidate truthfully states that empirical response fidelity and nuisance
covariance are unavailable and does not promote conditional variance or weight
to total precision or significance. That fail-closed limitation is retained.

The response identity does not bind the complete active transfer function for
the reasons listed under F007, and required products do not carry a stable link
to the canonical response/package identity. Together with the incomplete F005
recipient matrix, this prevents the claimed response-basis provenance from
being a complete, uniquely resolvable account of the applied response. F008
remains open without introducing any new covariance or uncertainty product.

### Local F009 — partially addressed, remains open

The isolated canonical writer and its failure paths are locally well tested,
and every runnable gate below passes. The schema/product/profile surface is
not synchronized with that writer:

1. The writer emits `citlali-raw-timestream-provenance-v4`, while
   `tools/baseline/audit_reduction_run.py` accepts only v1-v3. An executable v4
   probe independently returned `schema_ok=false` and `valid=false`; v4
   canonical-lineage semantics are not checked.
2. `selected_calibration_apt.ecsv` is absent from
   `validation/product_contracts.json`. The validator inventories every ECSV
   and rejects unmatched products as unclassified.
3. Strict baseline/profile comparisons inventory the new ECSV, while the
   checked-in comparison options exclude `citlali_profile.ecsv` but do not
   classify or exclude the new required copy. It therefore appears as an
   extra product against current accepted runs.
4. No checked-in v4/package-copy baseline fixture exercises the new required
   output, and the Beammap reopen test does not prove production identity
   population.

Thus the passing 174-test baseline suite verifies the unchanged v1-v3
contracts, not successful admission of successor-3 outputs. Local F009 remains
open.

### F010 — open, conditioned and unchanged

No new authorized SCI-ALIGN, SCI-AST, exact-SHA Unity, or astronomical evidence
is supplied. F010 remains open and conditioned without promotion.

## Validation performed on the exact candidate

A fresh Release build was configured with tests enabled. Public build
dependencies were retrieved after the sandboxed configure attempt encountered
the expected network restriction. The local targets `citlali_cli`,
`citlali_test`, `citlali_safety_test`, and
`citlali_science_map_fits_products_test` all built. Only third-party compiler
warnings were observed.

| Check | Independently observed result |
|---|---|
| Successor-3 focused CAL/startup/publication matrix | 24/24 passed: 21 normal-target tests plus 3 actual-writer tests |
| Corrected broad CAL fixtures | 6/6 passed across 3 suites |
| Normal test binary | 616/616 runnable passed; 1 known disabled test |
| Safety test binary | 14/14 passed |
| Full runnable CTest | 665/665 passed, 0 failed; 666 enumerated with one pre-existing disabled `MapFitterLifecycle.ExactProductSequence` |
| Baseline Python unittest discovery | 174/174 passed |
| Targeted atmosphere/product-contract Python tests | 26/26 passed |
| Checked-in validation ledger | valid, 60 records |
| Full config preflight `--require-all` | passed: 127/127 unit tests, 4 kits, 8/8 compatibility checks, 592 schema leaves, 100% coverage |
| Raw execution census | 67 records, zero review-required, digest `9603bb76740d549eeb5cd6f7364d0cde2b8f14052006e006047922eb6dc62448` |
| Raw boundary / execution-read unit tests | 12/12 and 3/3 passed |
| Atmosphere generator check | passed: 1,368 rows, 72 series |
| Parent-to-candidate `git diff --check` | passed |
| Parent-to-candidate binary patch SHA-256 | exact expected digest |
| Synthetic v4 reduction-audit consumer probe | nonconformity reproduced: `schema_ok=false`, `valid=false` |

The reported candidate facts are therefore independently reproduced: 18
authorized paths; focused 24/24; corrected broad 6/6; full 665/665 runnable
with one pre-existing disabled; baseline 174/174; preflight 127/127; schema
parity at 592 leaves; and raw census 67 records, zero review, with the exact
reported digest. Green gates do not erase the explicit contract-consumer and
coverage gaps above.

## Finding ledger

| Finding | Independent disposition |
|---|---|
| F001 | open, conditioned and unchanged |
| F002 | retain narrow structural closure, unchanged |
| F003 | closed/conformant for initial unsupported-unit rejection |
| F004 | closed/conformant for truthful legacy lineage and explicitly verified optional modern association |
| F005 | open, partially addressed: `validated` and actual noise-variance recipient proof/metadata incomplete |
| F006 | retain bounded `mJy/beam` closure, unchanged |
| F007 | open, partially addressed: incomplete/colliding identity and missing per-product canonical link |
| F008 | open, partially addressed: response identity is incomplete and not uniquely resolvable from products |
| local F009 | open, partially addressed: publication logic passes but schema/product/profile consumers are unsynchronized |
| F010 | open, conditioned and unchanged |

## Scope and architecture assessment

No scientific, numerical-recipient, covariance, uncertainty-product, RTC,
PTC, MAP, Beammap, Pointing, Lali, kernel, or filtering algorithm broadening
was found. Candidate changes are confined to the authorized 18-path repair
surface. The defects above are local technical conformance defects within the
existing decision, not authority for an auditor-designed repair.

## Owner decision brief

Reject complete SCI-CAL-001 closure for successor-3. Retain F002 and F006 only
within their accepted bounds; accept F003 and F004; keep F001 and F010
conditioned; and return F005, F007, F008, and local F009 for bounded technical
repair. Set the package to contract `approved`, implementation
`nonconformant`, validation `in_progress`, production `fail_closed`, with
verdict `amend`. Authorize no production or downstream activity from this
re-audit.
