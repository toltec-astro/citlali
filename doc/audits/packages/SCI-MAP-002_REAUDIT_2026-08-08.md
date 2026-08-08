# SCI-MAP-002 exact-repair-SHA independent re-audit — 2026-08-08

Status: complete documentation-only re-audit; candidate not accepted

Documentation identity correction: 2026-08-08 successor to re-audit commit
`4fa876d066a8bf7e6b971147a92f0b8b7ffd5c77`; scientific assessment unchanged

## Disposition

Exact repair commit `854a04b124e083e64706fd043e105182fee568af` is
**nonconformant** to the frozen, owner-approved SCI-MAP-002 contract.
Validation is **incomplete**, production remains **existing-use-only**, and
the proposed controlled verdict is **amend**. Passing tests do not override
the production-path findings below. No application edit, repair, push, Unity
access, reduction, external contact, parameter campaign, merge, production
authorization, or downstream launch was performed.

## Identity and authority gate

The worktree was clean before substantive inspection. The verified candidate
identity was:

| Item | Exact value |
| --- | --- |
| candidate | `854a04b124e083e64706fd043e105182fee568af` |
| sole parent | `46ad23888a40f5102cdfd50c06e49a549bdf8a20` |
| candidate tree | `188ae8535b61a3f560fa992b3bac0a5196436e5b` |
| local `origin/codex/repair-sci-map-002` | `854a04b124e083e64706fd043e105182fee568af` |
| frozen authority object | `dd5894679bf12bf4a5fb551e871b3c6010ef9b9b` |
| authority tree | `e87b507a6dc5246da0f65e563d96b94824e61ba1` |

The frozen original audit/evidence, eight owner decisions, coordinator
review, provenance correction acknowledgment, ledger proposal, and bounded
repair/re-audit handoff were read directly from the authority object. The
initial re-audit package incorrectly recorded the authority tree and asserted
that all 14 recorded artifact SHA-256 digests matched direct recomputation.
Only the original audit and original evidence digests matched; the other 12
were incorrect. This documentation-only successor recomputes every digest
directly from `dd589467...` and records the corrected table in the associated
ledger proposal. This identity correction does not change the scientific
inspection or disposition. The candidate was inspected as the complete
one-commit diff from its sole parent: 25 files, 1,813 insertions, and 143
deletions.

## Independent algebra and conforming mechanics

For the signed-lobe fixture `q=(2,3)`, `c=(1,-0.25)`, `d=(5,8)`, independent
decimal arithmetic gives:

```text
N = sum(q*c*d) = 4
C = sum(q*c)   = 1.25
Q = sum(q*c^2) = 2.1875
N/C            = 3.2
C^2/Q          = 5/7 = 0.714285714285714...
```

The candidate accumulates those three quantities and finalizes `N/C` plus
conditional formal weight `C^2/Q`. Its dimensionless conditioning variable is
`rho=abs(C)/sum(abs(q*c))`, with a conservative `2*gamma_n` bound. Independent
fixtures reproduced exact cancellation, near cancellation, unit rescaling,
and the finite extreme result `N/C=1e50`, `C^2/Q=1`.

Static inspection also confirms these approved mechanics:

- signed finite JINC lobes remain signed in `N` and `C` and are squared in
  `Q` and coefficient-squared coverage;
- the cache remains a fully populated square, including finite corner values
  beyond radial `r_max`, and map edges alone crop it;
- subpixel selection remains phase-quantized point sampling;
- selected-array cache construction validates positive finite resolved
  parameters and rejects a non-finite evaluated coefficient before swapping
  temporary caches into live state;
- the formal support plane is published as `coverage_bool`, and empirical
  policy can only downgrade formal support;
- coverage remains `sum(c^2/f_s)` seconds, not geometric exposure;
- the kernel numerator is `sum(q*c*k_processed)` and is finalized as `K/C`.

These mechanics are positive evidence but do not establish whole-candidate
conformance.

## Blocking findings

### RA-001 — iterative Beammap retains stale conditioning accumulators (P0)

`reset_beammap_mapmaking_buffers` clears active `signal`, `weight`,
`grid_weight`, coverage, kernel, and noise planes for every Beammap pass, but
does not clear the newly added `denominator_sum_abs` or `contributor_count`
planes. Those planes are allocated once with the observation buffer and are
then incremented on every JINC population. A second Beammap iteration
therefore finalizes fresh `N`, `C`, and `Q` against prior-pass
`sum(abs(q*c))` and contributor counts. This can change `rho`, its resolution
bound, formal support, and cancellation summaries.

When detector maps converge selectively, finalization also resets the entire
realized summary, processes only active maps, and clears the complete
`grid_weight` inventory. The resulting summary represents only the most recent
active subset rather than the coherent observation. No test exercises this
iterative/active-map path. F001, F004, F007, and F008 remain open.

### RA-002 — realized kernel/processing provenance contains placeholders (P1)

For enabled kernels, `kernel_template_identity` is assigned the generic JINC
output-response label rather than the actual upstream source-template type or
digest. `processing_realization_identity` is the literal
`requested-effective-jinc-digests-and-runtime-kernel-v1`; it does not bind the
enabled temporal filters, notches, mean/mask handling, flags, PTC
common-mode/PCA realization, or their actual operator state. The JINC record
also does not carry the coverage sample-frequency linkage required by the
coverage decision.

The compact four-stage shape and immutable product joins are useful, but the
realized record cannot identify the processing-filtered template that produced
`K/C`. The serialization test manually supplies a synthetic join and never
checks the production filter/template chain. F006 and F007 remain open.

### RA-003 — sequential/concurrent agreement is not tested on production paths (P1)

The test named `valid_two_level_sums_agree_under_declared_policy` hand-sums
four small arrays in two groupings and calls only the scalar finalizer. It
invokes neither `populate_maps_jinc` nor `populate_maps_jinc_parallel`, does
not select a runtime policy, and does not compare contributor sets, masks,
maps, or cancellation-sensitive pixels. No other test invokes either
production JINC population method. The direct detector-parallel path also does
not implement the scratch-then-merge organization named by the serialized
summation identity; its current caller is map-disjoint, but the provenance
label does not distinguish that path.

The required production sequential/concurrent gate and comparison envelope
are absent. F008 remains open.

### RA-004 — positivity validation broadens beyond selected JINC execution (P1)

The one-commit change replaces finite-only checks for every stored JINC shape
element with strict positivity in the unconditional `MapmakingConfig`
validator. `MapmakingConfig` defaults to the naive method, and the new test
intentionally proves that a default/naive request with an inactive zero JINC
shape is rejected. The owner decision requires fail-closed admission for every
**selected JINC** array/product; it does not authorize rejecting inactive JINC
settings for naive mapmaking. This is a configuration-surface broadening
outside the approved selected-product boundary. F005 remains open.

### RA-005 — required production-boundary evidence is incomplete (P1)

Helper-level tests cover many individual cases, but the required local matrix
does not exercise all of the following through production seams: below/equal/
above `r_max` response boundaries; a generated non-finite coefficient with
failure-before-deposition/publication; formal-mask behavior after failed
admission; actual sample-frequency coverage linkage and analytic-zero
coefficient behavior; realized filter/template identity; actual output
failure suppression; product/HDU/digest joins produced by the writer; and
production sequential/concurrent agreement. F002 and F003 are implemented
consistently with their approved conventions but remain validation-incomplete;
F005--F008 are not closed.

## F001–F008 reassessment

| Finding | Re-audit status | Reason |
| --- | --- | --- |
| F001 | open | Formal-mask helper is corrected, but stale Beammap conditioning can misclassify iterative products. |
| F002 | addressed, validation incomplete | Square/corner/map-edge mechanics conform; required below/equal/above response fixture is absent. |
| F003 | addressed, validation incomplete | Phase-point implementation conforms; production-path boundary/refinement evidence is incomplete. |
| F004 | open | Scalar conditioning conforms, but Beammap reuses stale `sumabs`/count state. |
| F005 | open | Cache admission is fail-closed, but validation broadens inactive non-JINC requests and production failure boundaries are incomplete. |
| F006 | open | Coverage and `K/C` arithmetic conform, but sample-frequency and actual template/processing identity are not bound. |
| F007 | open | Compact stages/joins exist; realized processing/template identity and coherent iterative summaries do not. |
| F008 | open | No production sequential/concurrent test and no exact-SHA human runtime evidence. |

## Scope audit

The diff changes no RTC, PTC, temporal-filter, notch, PCA/common-mode,
source-template generator, noise/jackknife, Wiener-filter, convolution,
source-fitting, covariance, or GLS implementation file. It adds no radial
predicate, pixel-area integration, geometric-exposure reinterpretation,
parameter value/campaign, high-cardinality provenance, or production-status
authorization. JINC output additions are gated on initialized JINC products.

The unconditional inactive-JINC positivity rule in RA-004 is nevertheless a
prohibited unapproved behavior broadening, so the candidate cannot receive an
unqualified no-broadening finding.

## Local gates

All passing commands ran from the exact candidate source, using a local
offline build configured from already cached dependency sources and Eigen 3.

| Gate | Result |
| --- | --- |
| focused JINC plus product test selection | 13/13 passed |
| `citlali_cli` | built |
| monolithic `citlali_test` | built |
| `citlali_safety_test` | built |
| isolated JINC contract target/header | built |
| focused science-map FITS product target | built |
| full CTest | 648/648 enabled tests passed; one unrelated pre-existing disabled test did not run |
| baseline-tool suite | 173/173 passed |
| config preflight | 127/127 passed; 4/4 mode kits; 8/8 compatibility cases; zero skips; all typed audits passed |
| science-change ledger | valid: 3 changes, 5 integration commits |
| validation-profile registry | valid |

No required-data skip occurred and no unexpected error-level message appeared
in a passing gate. Before the offline build was configured, the first build
attempt reported that `build/` did not exist; the first configure attempt
rejected Eigen 5 until the existing Eigen 3 prefix was selected; and CMake
reported a nonfatal sandbox warning while trying to write its user package
registry. Two exploratory validator invocations used an obsolete ledger path
and unsupported `--check` option; both were corrected with the checked-in
`validation/intended_science_changes.json` and the supported registry listing.
None is counted as a passing gate.

## Human evidence still required

The exact-SHA U01–U06 protocol remains the human evidence specification:
ordinary array-grouped JINC with formal weights and kernel (U01); the matched
empirical-weight case (U02); matched detector-Beammap sequential and concurrent
cases (U03/U04); concurrent repeatability (U05); and U06 only if a pre-existing
checksum-pinned independent unit-source input already exists. Every case must
bind a clean exact-SHA checkout, compiler/build/dependency/host/thread and
executable identities, checksum-pinned input/calibration/flags/sample-rate,
authored/resolved configuration, provenance, logs, product inventory, FITS
metadata, and file digests.

For `854a04b...`, that evidence was not supplied or executed. More importantly,
it cannot cure RA-001–RA-004. No human run should be launched from this report;
a coordinator/owner must first review this disposition and decide whether a
new bounded successor repair is authorized. Any later human evidence must
target the exact successor SHA and use a pre-approved comparison envelope.

## Stop point

Stop for coordinator and project-owner review. Do not integrate this proposal
into the canonical ledger automatically, do not treat the candidate as
accepted, and do not launch Unity or downstream work from it.
