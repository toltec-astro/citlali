# SCI-MAP-002 bounded repair and re-audit handoff — 2026-08-03

## Authority and disposition

The project owner completed the SCI-MAP-002 JINC scientific contract on
2026-08-03. This record prepares a bounded repair/re-audit handoff. It does
**not** authorize implementation, a repair task, a Unity request, a numerical
parameter campaign, a production-status change, or a re-audit.

- Governing application SHA assessed by the audit:
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Audit source commit: `abb8d8896d6a1cbaa912b9ac181bd649588acc62`.
- Coordination integration commit: `d313c879b`.
- Final audit:
  `doc/audits/packages/SCI-MAP-002_SCIENTIFIC_CONTRACT_AUDIT.tex`, SHA-256
  `d77bb5c8e555b43d5303ad2ce0a81e5baef42df2beb85347f9f98cab759d5239`.
- Frozen independent core:
  `doc/audits/packages/SCI-MAP-002_INDEPENDENT_CORE.tex`, SHA-256
  `2c1f9ff95f65422a098846f747ed165d5aeddc5bedd854678bfa7faeebba4e24`.
- The eight owner-decision records and their digests in
  `doc/audits/audit-ledger.yaml` are part of this handoff.

Until a separately authorized repair and fresh re-audit both succeed,
implementation remains `nonconformant`, validation remains `incomplete`,
and production remains `existing_use_only`. Findings F001--F008 remain open.

The repairer must use a fresh worktree and a separate branch, suggested as
`codex/repair-sci-map-002`, created from a coordinator-selected exact
application SHA. It must never be based on the audit branch. Record the
selected base SHA, its merge relationship, and exact touched files before the
first source edit.

## Approved repair sequence

`SCI-MAP-001` owns the shared map-product validity vocabulary that this JINC
successor consumes. Its accepted repair, including a fresh re-audit, is a
required upstream application state for `SCI-MAP-002` repair. Therefore the
selected MAP-002 repair base must contain the accepted MAP-001 repair commit;
the MAP-002 repair does not begin from the pre-repair audit SHA or in parallel
with changes to that shared contract. This sequencing does not close MAP-001,
authorize its repair, or expand MAP-002 scope.

## Approved contract to implement exactly

The successor preserves the observed identities

\[
N = \sum_i q_i c_i d_i, \quad C = \sum_i q_i c_i, \quad
Q = \sum_i q_i c_i^2,
\]

with finalized signal `N/C` and, when formally supported, formal mapmaker
weight `C^2/Q`. Signed finite lobes remain valid. Formal mapmaker weight and
empirical working weight remain distinct; empirical policy may exclude a
formally valid pixel but may never promote a formally invalid pixel.

The following eight owner decisions are one coherent contract, not optional
work items:

1. Retain the square fully populated cache footprint. `r_max` remains both
   second-JINC-zero parameter and square-cache half-width; corner values
   beyond radial `r_max` are defined response.
2. Retain phase-quantized point sampling. `subpixel_n` refines residual-phase
   binning; it is not pixel-area integration.
3. Remove unit-bearing absolute `C`/`Q` gates. Require finite contributors
   and accumulators, `Q > 0`, exact-cancellation failure, and a documented,
   serialized dimensionless `rho = abs(C) / sum(abs(q_i c_i))` resolution
   bound from the realized summation method and contributor count.
4. Require stable selected-array identity and finite-positive `a`, `b`, `c`,
   `r_max`, pixel size, and array scale. Non-finite evaluated coefficients or
   nonphysical/missing selected-array state fail the selected JINC product
   before deposition; they are not detector-local omissions.
5. Retain `coverage_bool` as the authoritative formal-support mask: finalized
   signal/formal weight finite, formal weight strictly positive, and all
   admission/conditioning checks passed.
6. Retain coverage as coefficient-squared effective integration time,
   `sum(c_i^2 / f_s,i)`, in seconds. It is neither geometric exposure, hit
   count, nor validity; it is consumed only with formal support.
7. Define `K/C` as the realized processing-filtered source-template response
   projected through JINC—not an unfiltered analytic JINC response or a
   measured beam.
8. Emit one compact, atomic, forward-only requested/effective/resolved/
   realized record per coherent observation or declared processing segment;
   never a per-sample, per-detector, or per-pixel payload.

The detailed owner-decision records fix the boundary cases. The implementation
must not substitute a new convention, radial predicate, pixel integration,
parameter campaign, or high-volume diagnostic product.

## Bounded repair work packages

| Work package | Findings | Required outcome |
| --- | --- | --- |
| Typed request/effective/resolved admission | F005, F007 | Validate selected array and physical parameters at the mapmaking boundary; preserve requested/effective values and one-way ownership. |
| Coefficient and accumulator conditioning | F004 | Replace absolute gates with the approved finite/`Q`/`rho` policy and compact realized summary. |
| Finalization and formal support | F001, F004 | Finalize `N/C` and conditional `C^2/Q`; make `coverage_bool` the typed formal-support authority without a new map product. |
| Coverage and kernel metadata | F006, F007 | Correctly label coefficient-squared seconds, formal/empirical distinction, and processing-filtered source-template kernel identity. |
| One-way product provenance | F006, F007 | Add four-stage provenance and immutable joins for signal, formal/empirical weight, mask, coverage, kernel, output file/HDU, and digest. Required output failure suppresses realized success. |
| Local truth suite | F001--F008 | Add direct-equation, boundary, finite-state, unit-scaling, response, product/provenance, and sequential/concurrent tests at the exact repair SHA. |
| Human external evidence | F008 | Only after local gates pass, prepare a separately approved exact-repair-SHA Unity request. Codex does not access Unity. |

The repairer must establish exact source seams on the selected repair base.
New ownership belongs in a coherent JINC mapmaking plan/result or provenance
object, not cross-cutting mutable `Engine` state.

## Prohibited scope

The repair must not alter JINC parameters, sampling scale, footprint/cache
geometry, RTC/PTC filtering or PCA/common-mode cleaning, source-template
generation, noise/jackknife or downstream-filtering behavior. It must not add
a radial-support cut, pixel-area integration, geometric-exposure product,
per-sample/per-detector/per-pixel output, a new estimator, covariance claim,
or production-profile change. It must not repair, merge, push, or update
canonical audit conclusions on the audit branch.

## Required local repair gates

At the exact repair SHA, with no required-data skip and zero unexpected
error-level messages, return all of the following:

1. One-pixel direct fixtures independently evaluating `N`, `C`, `Q`, `N/C`,
   and conditional `C^2/Q` for finite contributors, including signed lobes.
2. Support fixtures below/equal/above `r_max`, at cache edges/corners, and
   map-edge cropping, proving no radial predicate was introduced.
3. Phase fixtures at rounding/bin boundaries, `subpixel_n = 1`, and bounded
   refinement that demonstrates point-phase—not pixel-area—response.
4. Exact and near cancellation, resolved low formal weight, unit rescaling,
   extreme finite range, zero/negative/non-finite `Q`, and serialization of
   summation method, contributor count, and `rho` bound.
5. Missing array identity and every parameter-domain/non-finite boundary,
   plus signed finite/non-finite coefficients. Failure precedes selected JINC
   deposition or publication.
6. Formal-support truth for finite-positive, zero/negative/non-finite,
   failed-admission, and cancellation cases; empirical downgrade and attempted
   promotion. `coverage_bool` exactly equals the formal-support predicate.
7. Coverage truth proving coefficient-squared seconds, phase/coefficient and
   sample-frequency linkage, zero-coefficient behavior, and no promotion of
   invalid formal support.
8. Kernel truth proving signal-equivalent support/mask and compact source
   template plus enabled filters/notches/masks/flags/PTC/PCA realization
   identity, without relabeling it as analytic or measured beam.
9. Requested/effective/resolved/realized provenance round trips, one-way
   lifecycle, stable array identity, product/HDU joins, digest integrity,
   failure-path suppression, and bounded cardinality.
10. An all-valid no-broadening control: preserve support, phase selection,
    arithmetic/merge policy, and numeric results except registered metadata or
    provenance changes. Attribute every numeric delta to an approved defect.
11. Focused CTests, relevant product/provenance validation, baseline tools,
    config preflight, and isolated-header/build checks for touched interfaces.
12. Sequential/concurrent agreement under the declared numeric policy. If not
    bitwise, pre-register and justify the comparison envelope.

## Fresh re-audit and external gate

A fresh re-auditor in a new worktree (suggested branch
`codex/reaudit-sci-map-002`) must assess the exact repaired commit—not the
working tree, coordination line, or audit branch. It must independently
recompute equation fixtures; verify F001--F008 closure claims and prohibited
scope; inspect any separately approved, human-supplied same-SHA Unity evidence;
and issue a new implementation, validation, and production disposition.

No repair is accepted merely because a map resembles historical output. Until
re-audit succeeds, `existing_use_only` and all validity restrictions remain.
