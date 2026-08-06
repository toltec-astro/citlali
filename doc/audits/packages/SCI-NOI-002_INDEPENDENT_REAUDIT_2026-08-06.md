# SCI-NOI-002 independent re-audit — 2026-08-06

Status: **re-audit complete; verdict `amend`; exact repair is not ready for
application integration**. The approved scientific contract is implemented
correctly for the bounded finite-stack estimator, global nonprecision scale,
canonical FITS/source-table identities, and deterministic local fixtures, but
the exact candidate remains nonconformant at package membership/integrity,
completion truthfulness, filtered-product validity, and the active validation
boundary. Broad F004, F005, F006, and F007 closure is not supported.

This is a documentation-only independent re-audit of exact candidate
`0bc4d95d6bb2117442d0ccdb79c57e42e0b79989`. It is not an application
repair, integration, push, production decision, evidence request, Unity action,
or astronomical reduction.

## Exact object and authority binding

- Worktree: `/Users/gwilson/.codex/worktrees/f908/citlali-refactor`.
- Re-audit branch: `codex/reaudit-sci-noi-002`, created at the exact candidate
  only after a clean-state/ref/worktree check.
- Candidate: `0bc4d95d6bb2117442d0ccdb79c57e42e0b79989`.
- Required parent/application base:
  `d5015fe716971bf8ea617e8a187311bf5af05185`.
- Candidate tree: `25f39d8b1ba2527c2a154a69a527b0d8835a412a`.
- Parent tree: `5483c0647ef692439b5773744a5576b4b4dfcabd`.
- Candidate is exactly one commit after the required parent; their merge base
  is the required parent. Local and remote `codex/repair-sci-noi-002` both
  resolve to the candidate. `origin/codex/refactor-mainline` resolves to the
  required parent.
- Coordination authority:
  `ad5c288bde7c801d2e436c053e340a809a669343` on
  `codex/coordinate-sci-noi-001-dispatch`, tree
  `5a00271c729a0627cd6a3d4dacb2eae1adebc7b7`. It is neither the candidate nor
  its application ancestor; candidate/coordination merge base is
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`. It was used only to recover
  authority objects, never as the application base.
- Entry and final pre-documentation worktree state were clean.

The frozen repair authorities and manifest-bound evidence were read as exact
Git objects and independently SHA-256 checked:

| Authority/evidence | Exact Git object | SHA-256 |
| --- | --- | --- |
| repair prompt | `ad5c288:doc/audits/prompts/SCI_NOI_002_REPAIR_PROMPT.md` | `45fc19d6ccf0e55aa1c2a1189d97f72ffb9b51027c659580d7cbcc9415c4bc71` |
| repair authority manifest | `ad5c288:doc/audits/handoffs/SCI-NOI-002/SCI-NOI-002_REPAIR_AUTHORITY_MANIFEST_2026-08-06.yaml` | `6f4b84995c8cb118bdb182b9189c4b95464c4fe7e414b762debb8786e825ce79` |
| repair readiness | `ad5c288:doc/audits/packages/SCI-NOI-002_REPAIR_DISPATCH_READINESS_2026-08-06.md` | `246c751397b2d138939372c7943e8187ba36658bbb7a5b8f96bb01edca0f4804` |
| owner decision brief | `64ba81795110d89d8baf0ad7d645d16472c254c5:doc/audits/packages/SCI-NOI-002_OWNER_DECISION_BRIEF_2026-08-06.md` | `3520172cfc11e8e34f280f9ebdf147ea414c7a3a4ca6109bad55354a5ff3cf71` |
| final exact-d501 audit | `4f1fec36f7802f3b5e8ac067377679946930983c:doc/audits/packages/SCI-NOI-002_SCIENTIFIC_CONTRACT_AUDIT.tex` | `2874ffe950aed769f73277ed8f60ecab8860692d24e7c541f05a47a041a8a40d` |
| frozen independent core | `f08a6da2ceebff03f498386f374980d13c5146a6:doc/audits/packages/SCI-NOI-002_INDEPENDENT_CORE.tex` | `36781b766a2f57c9a3bd7e173ee8f1d85cba7f3d08afe2e67a403166f6b6d72d` |
| original ledger proposal, noncanonical | `4f1fec36f7802f3b5e8ac067377679946930983c:doc/audits/proposals/SCI-NOI-002_LEDGER_PROPOSAL_2026-08-06.yaml` | `5574d8e34fcfba8f4709d5848e79732ff3557a9817be25604375b9f3d4ec278d` |
| NOI-001 R3 bounded ensemble evidence | `5a027c94ef9fc9c4a6e6cadc84af1c8a550d3508:doc/audits/packages/SCI-NOI-001_INDEPENDENT_CORE_R3.tex` | `27263ab3bf29ac8f098463455e540f13e783241a688ef2bc5cb15b1f2a4319da` |

The owner brief is the approved policy authority, the independent core is the
equation authority, the final audit supplies the exact-d501 trace/findings, and
NOI-001 R3 supplies only the conditioned `source_imprinted_current` ensemble
identity. Thread summaries were not treated as authority.

## Exact candidate diff and scope resolution

The parent-relative candidate diff has 14 modified existing files, 1,449
insertions, 135 deletions, and no added/deleted/renamed path:

| Path | `+/-` | Finding/component trace |
| --- | ---: | --- |
| `include/citlali/core/mapmaking/map.h` | `21/1` | F001/F005 validity state and fixed-projection API; coordinator-authorized expanded direct path. |
| `include/citlali/core/pipeline/fits_image_hdu_names_wcs.h` | `20/0` | F004/F005 canonical and legacy HDU names; authorized expanded direct metadata path. |
| `include/citlali/core/pipeline/fits_image_metadata_keys.h` | `141/15` | F002--F005 units, aliases, package joins, validity/restriction keys; authorized expanded direct metadata path. |
| `include/citlali/core/pipeline/fits_image_units_kernels.h` | `38/13` | F001/F002/F004/F005 descriptions and units only; authorized expanded direct metadata path, no kernel arithmetic change. |
| `include/citlali/core/pipeline/map_filtering_template_noise.h` | `3/2` | F002/F004 log-label correction only; no FLT operator change. |
| `include/citlali/core/pipeline/map_image_output_helpers.h` | `188/25` | F002--F005 canonical/alias HDUs, joins, validity, and storage consequences. |
| `include/citlali/core/pipeline/map_source_table_output.h` | `31/4` | F004 distinct fitted-amplitude/full-map-RMS identity and invalid-ratio handling. |
| `include/citlali/core/pipeline/noise_config_serialization.h` | `9/0` | F007 realized completion/validity serialization; authorized expanded direct provenance path. |
| `include/citlali/core/pipeline/noise_execution_plan.h` | `118/15` | F001--F007 identities/digests, requested/effective/completed state, and planned cardinalities. |
| `include/citlali/core/pipeline/noise_provenance.h` | `138/4` | F001--F007 package semantics and recursive member digest inventory; authorized expanded direct provenance path. |
| `src/citlali/core/mapmaking/map.cpp` | `133/30` | F001/F002 estimator, calibration, invalid ratios, projection; F004 source-finder name/comment only. |
| `tests/test_config_scaffold.cpp` | `163/2` | F001--F007 exact local configuration/count/provenance/source-table fixtures. |
| `tests/test_science_map_fits_products.cpp` | `299/0` | F001--F005 exact math/FITS/projection/alias fixtures; authorized existing direct FITS test. |
| `validation/product_contracts.json` | `147/24` | F003/F004/F005 canonical/legacy product requirements and header checks. |

The coordinator explicitly authorized the seven expanded paths named above as
existing direct metadata/provenance dependencies and the existing direct FITS
test. Every expanded change stays within that direct purpose. No new file,
schema version, wrapper, verifier, framework, product class, or persistent
product type was introduced. The additions are narrow functions/constants in
existing headers, not a new helper component. This scope admission does not
approve their correctness; the defects below remain findings.

No runtime YAML, count, default, selection threshold, FRUIT/FLT/SRC/MODE/MAP/
JINC/RTC/PTC algorithm, dense covariance, sign stream, per-sample ID, or
primary-channel behavior changed. ADR 0005 and timestream paths are byte-
unchanged by the candidate.

## Independent mathematics and exact limits

For the completed stack of `R` maps at pixel `p`, the implementation computes

```text
mean_p = (1/R) sum_r Y[r,p]
V_p    = (1/R) sum_r (Y[r,p] - mean_p)^2
       = (1/R) sum_r Y[r,p]^2 - mean_p^2 .
```

Finite negative roundoff in the final expression is clamped to zero. This is
the second central moment of the completed, empirically centered,
`source_imprinted_current` stack. It is not the `1/(R-1)` iid-unbiased sample
variance and contains no joint-design dependence correction. It is conditional
on the exact completed stack, support, response, and source imprint; it is not
physical-noise variance/covariance, inverse variance, precision,
significance, calibration, or count-adequacy evidence.

The fixed complete-stack missingness rule is truthful in the estimator: a
nonfinite member makes that pixel's mean and scatter nonfinite, and Eigen's
post-subtraction maximum preserves NaN. No per-pixel reduced count is silently
used. A direct Eigen probe returned `1 nan 0` for positive/NaN/negative inputs
after `.max(0.0)`. Product-level raw scatter validity is true when at least one
pixel is finite; per-pixel nonfinite values remain unavailable.

Exact limits and independent fixtures agree with the implementation:

| Stack | Mean | `V` | Permitted meaning |
| --- | ---: | ---: | --- |
| `[5]` (`R=1`) | 5 | 0 | descriptive zero only; uncertainty/ratio invalid |
| `[-2, 2]` | 0 | 4 | complementary completed-stack scatter |
| `[3, 3]` | 3 | 0 | duplicate design; no positive denominator/scale |
| `[0, 2]` | 1 | 1 | exact `R=2` completed-stack normalization |

For formal nonprecision coefficient `q_p`, support threshold `tau`, and

```text
D = {p: q_p finite, q_p > 0, q_p >= tau, V_p finite, V_p > 0},
m = median over D of (q_p V_p),
alpha = 1/m,
q'_p = alpha q_p,
```

the scale is valid only when `D` is nonempty and `m` is finite and positive.
An unavailable diagnostic remains NaN; requested live application throws.
`q'` remains an existing-use-only nonprecision coefficient. The exact fixtures
give `(q,V)=(0.25,4) -> alpha=1` and `(0.5,4) -> alpha=0.5`, both with
`q'=0.25`. Coefficient-standardized signal is emitted only where signal and
scaled coefficient are finite and the coefficient is nonnegative. Direct
signal/scatter ratios require `R>=2` and a finite positive denominator;
otherwise the numeric value is NaN, never zero.

For a fixed projection `P` and fixed nonzero response `rho`, the admitted
scalar diagnostic is

```text
h_r = sum_p P_p Y[r,p] / rho,
V_h = (1/R) sum_r (h_r - mean(h))^2 .
```

The implementation rejects shape mismatch, nonfinite projection/realization,
and nonfinite or zero response. Two perfectly correlated pixels with
realizations `[1,1]` and `[-1,-1]` and `P=[1,1]` give `V_h=4`, while diagonal
summation gives only 2; `rho=2` gives 1. Realizations `[2,0]`, `[0,2]` with
template `[0.5,-0.5]` give 1. This preserves the selected projection's finite-
stack cross-pixel behavior without constructing dense covariance. It does not
become physical aperture uncertainty, a fitted/selected template error, or a
general covariance product.

All 11 semantic IDs independently recomputed as
`SHA256("citlali-noise-products|SCI-NOI-002-v1|" + identity)`; all five
registry-persisted digest checks matched.

## Requested, effective, and completed-count semantics

- Enabled requests with count `<=0` are rejected without changing the request
  or installing an effective plan.
- Directly or mapmaking-disabled requests preserve requested values and
  feature flags, while the effective activation is false and effective count
  is zero. No default/count selection was added.
- Disabled completion records all eight cardinalities as available zero,
  `generation_executed=false`, `outputs_completed=true`,
  `actual_completion_valid=true`, and basis
  `effective_disabled_zero_work`. This is truthful only as zero planned work.
- Enabled completion is recorded after observation/coadd output routines and
  the mapmaking/coadd sidecars return, immediately before the noise sidecar is
  written. Earlier exceptions prevent a new noise sidecar; its own write
  failure propagates.

Runtime YAML is unchanged: Science remains enabled with configured request 10,
Pointing enabled with configured request 5, OOF disabled with inert configured
request 1/effective zero, and Beammap disabled with inert configured request
10/effective zero. These are observed configuration values, not validated
scientific counts, minima, or universal defaults. The optional capacity 64 is
not selected as a target or requirement.

The enabled "completed" cardinalities are nevertheless not measured from
realized stack IDs or the publication inventory. They are calculated as the
effective requested count multiplied by completed mapmaking plan cardinality,
with output-stage factors inferred from configuration. The code then sets
`actual_completion_valid=true`, `completed_count_matches_effective=true`, and
`outputs_completed=true` with basis
`successful_pipeline_return_under_effective_plan`. That basis proves the
bounded successful-return path, not actual realization-by-realization or
member-by-member completion. Crash/interruption, partial publication, and a
reused output root with a pre-existing sidecar remain unrepresented. F007 can
therefore close no broader than enabled-zero admission, disabled-zero no-work,
and post-success planned cardinality; count adequacy/default selection and
incomplete-execution truth remain open.

The active reduction auditor encodes the old disabled representation: it
requires `outputs_completed=false` and all eight disabled counts unavailable.
A direct exact-candidate record probe produced zero errors for enabled state
and nine errors for disabled state (completed outputs plus all eight available-
zero counts). Its 95 self-tests pass because they test its historical contract.
Thus the repository cannot presently validate the exact candidate's approved
disabled semantics end to end.

## Package integrity, joins, and detached products

The sidecar carries complete estimator/design/support/missingness/restriction
semantics once. Each joined FITS HDU receives ten compact fields: package,
sidecar, product identity/version, semantic digest/kind, scope, product
validity, restriction, and missingness. Source-table metadata carries its
product-specific join. Full package metadata is not duplicated into every HDU.
Detached products are explicitly unverified/out-of-contract.

Member hashing is deterministic for the set it happens to discover: recursive
paths are sorted lexicographically; each recorded member has relative path,
SHA-256, and size; the aggregate digest covers path, digest, and size in that
order. The YAML sidecar excludes itself, avoiding a circular/self digest. The
sidecar write uses a temporary file plus rename and propagates hash/write
failures.

The discovered set is not an authoritative current-publication inventory.
`write_noise_provenance_file` recursively admits every regular `.fits`,
`.ecsv`, or `.nc` under the reduction root and nothing else. It has no realized
publication admission record, reduction-instance identity, or stale-file
filter. Arbitrary/pre-existing/unrelated matching files are included; current
members using another suffix or another package sidecar are excluded; symlink
targets may be followed. Default `use_subdir: true` reduces but does not repair
the defect, because `runtime.use_subdir: false` deliberately reuses the output
root. Files can also change between hashing and sidecar rename. Atomic sidecar
publication is not an atomic package snapshot, and an older sidecar can remain
after a failed reused-root run.

No existing validator recomputes the member inventory/digest, proves inclusion
or exclusion, checks a file digest, or verifies HDU/table joins against that
sidecar. `validation/product_contracts.json` checks selected fixed header
values, but not the package instance or inventory. Package integrity and joins
therefore remain F003-open.

Hashing uses an 8,192-byte input buffer but feeds its SHA implementation byte
by byte. Time is `O(total admitted package bytes)`, transient hash memory is
constant, and YAML memory is `O(member count)`. It adds one full read of every
admitted product after publication.

## Canonical/legacy data-plane duplication and cost

The candidate writes the same `float64` array under these canonical/legacy
names:

| Canonical HDU added by candidate | Existing legacy data plane(s) with identical values |
| --- | --- |
| `conditional_stack_scatter_*` | `noise_variance_*` |
| `coefficient_standardized_signal_*` | `sig2noise_*` and `sig2noise_pixel_*` |
| `filtered_pixel_stack_scatter_*` (filtered only) | `point_source_uncertainty_*` |
| `conditional_stack_scatter_ratio_*` (filtered only) | `sig2noise_point_source_*` |

The two legacy coefficient-standardized planes already duplicated one another;
the candidate adds one more canonical copy, not two. `point_source_flux_*`
already duplicated signal and is not candidate-added storage.

For `N` pixels, each new plane has exactly `8N` payload bytes. The candidate
adds two planes (`16N` bytes) per raw empirical map and four (`32N` bytes) per
filtered empirical map. A representative three-array Stokes-I observation
with one raw and one filtered product per array adds 18 planes, exactly `144N`
payload bytes:

| Map shape | Added payload | FITS-padded data | Minimum total including one 2,880-byte header block per added HDU |
| --- | ---: | ---: | ---: |
| `1024 x 1024` | 150,994,944 B = 144 MiB | 151,009,920 B | 151,061,760 B (144.064 MiB) |
| `2048 x 2048` | 603,979,776 B = 576 MiB | 603,987,840 B | 604,039,680 B (576.057 MiB) |

Additional header blocks, observations, and iterations multiply this cost.
Because the package hash rereads each plane, the representative incremental
write-plus-hash traffic is at least 288 MiB at `1024^2` and 1,152 MiB at
`2048^2`. The writer allocates one temporary full-plane `valarray` per HDU, so
peak transient memory grows by one plane, not by all added planes; the
`MapBuffer` does not persist canonical copies.

Legacy EXTNAMEs are plausibly required by existing name-based readers.
Separate canonical EXTNAMEs are not required by the frozen owner decision,
which requires distinct product identity and explicit compatibility metadata.
A single legacy-named physical plane can retain old readers while carrying
canonical `NOIPRID`/digest/sidecar identity for metadata-aware readers. That
would not satisfy a new reader that insists on canonical EXTNAME rather than
canonical identity. The current duplicate-plane policy is therefore material
and avoidable under one compatible contract, but the coordinator/owner must
choose whether canonical EXTNAME lookup is itself required. The writer is not
changed by this re-audit.

## Product validity and consumers

Canonical FITS units, identities, restrictions, legacy alias metadata, and
source-table ratio behavior otherwise match the approved contract. Raw and
filtered primary stack scatter uses the calculated
`noise_stack_scatter_valid`; invalid scale and ratios fail closed. The source
finder's arithmetic and threshold are unchanged: the diff only renames a local
`sig2noise` variable to `source_finder_engineering_score` and changes comments.
The source table still computes fitted amplitude divided by full-map RMS, now
returns NaN for nonfinite/nonpositive denominators, and labels the legacy name
as not significance.

One filtered-product validity path is not truthful. The specialized
`filtered_pixel_stack_scatter_*` HDU declares
`conditional_descriptive` whenever `n_noise>0`, ignoring the calculated
`noise_stack_scatter_valid`. An all-nonfinite completed stack can therefore
publish an all-NaN plane marked conditionally valid. Its ratio is labeled
`unavailable_R_lt_2` whenever `noise_uncertainty_use_valid` is false, even
when `R>=2` and the real cause is unavailable scatter. This violates the
minimum detached-product validity requirement and keeps F003/F005 open.

The untouched secondary Mapdiag NetCDF surface still emits:

- `map_noise_weight_median_ratio`: description calls the quantity
  "jackknife variance";
- `map_noise_weight_scale`: description calls it an empirical scalar applied
  to formal weights without the canonical nonprecision identity; and
- `map_noise_products_s2n_sigma`: retains the S2N name, although its
  description does say it is not calibrated significance.

Those names/descriptions do not satisfy broad canonical F004 identity and
precision/significance restrictions. Canonical FITS and source-table behavior
passes, but F004 remains `open_conditioned` for Mapdiag. No Mapdiag repair or
scope expansion occurred.

F005 remains `open_conditioned`. Static tracing confirms signal filtering uses
signal-background affine edge handling, whereas realization paths are zero-
centered and use different edge treatment. Strict signal/realization/kernel
operator-edge parity is therefore correctly persisted as
`scope_blocked_not_applicable_pending_FLT`; the exact projection fixtures do
not close it and no FLT/Wiener mathematics was changed.

F006 remains open and SCI-FRUIT-001-owned. No FRUIT algorithm, default,
threshold, iteration, add-back, stopping, or bright-source uncertainty path
changed. Passing bounded FRUIT configuration fixtures supplies no adaptive or
astronomical closure.

Only Stokes I is within the repaired science-map surface. No `r`, I/Q, phase,
or ADR 0005 boundary changed. The held future note remains exactly that
auxiliary measured channels may diagnose non-optical readout/electronics noise
but are not substitutes for primary x-derived science-map noise.

## Re-audit findings

No P0 finding was observed.

| ID | Severity | Class | Mapped finding(s) | Exact result |
| --- | --- | --- | --- | --- |
| `SCI-NOI-002-RA-B001` | P1 | implementation/provenance defect | F003 | Recursive extension scan is not a realized current-package inventory; stale/unrelated inclusion, valid-member exclusion, TOCTOU, and absent inventory/join validation make integrity claims unreliable. |
| `SCI-NOI-002-RA-B002` | P1 | required-validation defect | F003, F007 | The active reduction auditor rejects exact disabled available-zero semantics with nine errors while its own historical tests pass. |
| `SCI-NOI-002-RA-B003` | P1 | lifecycle/contract defect | F007 | Enabled completed counts are inferred from effective plan and map cardinality, not measured realization/publication completion, yet are labeled actual and matching; crash/partial/reused-root behavior is not represented. |
| `SCI-NOI-002-RA-B004` | P1 | product-validity defect | F003, F005 | Filtered stack scatter uses `n_noise>0` instead of actual scatter validity and can misstate an all-NaN `R>=2` product as conditionally descriptive with the wrong ratio-invalid reason. |
| `SCI-NOI-002-RA-R001` | P2 | storage/performance defect | F003, F004 | Canonical EXTNAME planes duplicate legacy arrays, adding exactly 16N raw and 32N filtered bytes plus a full hash reread; three-array raw+filtered products add 144N payload bytes. |
| `SCI-NOI-002-RA-R002` | P2 | residual metadata gap | F004 | Untouched Mapdiag names/descriptions retain variance/weight/S2N terminology inconsistent with broad canonical meaning. |

## F001--F008 proposed dispositions

| Finding | Proposed controlled disposition | Re-audit basis and retained boundary |
| --- | --- | --- |
| F001 | `closed_bounded_owner_accepted` | Exact centered `1/R` conditional finite-stack math, R=1/R=2 behavior, fixed complete-stack missingness, source-imprinted target, and forbidden interpretations are implemented and pass exact fixtures. No physical covariance/precision/significance claim closes. |
| F002 | `closed_bounded_owner_accepted` | Reciprocal-median diagnostic, valid region, explicit invalid state, fail-closed requested application, and existing-use-only nonprecision labeling conform. No spatial precision model is inferred. |
| F003 | `open` | Compact joins/full-once metadata are boundedly correct, but current-package membership, stale-file behavior, atomic package integrity, filtered validity, and end-to-end inventory/join validation are nonconformant. |
| F004 | `open_conditioned` | Canonical FITS/source-table identities and invalid ratios pass; broad closure is blocked by Mapdiag and the unresolved canonical-EXTNAME versus material duplicate-plane compatibility choice. SCI-SRC-001 still owns any tail/search/catalog claim. |
| F005 | `open_conditioned` | Exact status is `scope_blocked_not_applicable_pending_FLT`; strict signal/realization/kernel edge/operator parity remains unproved, and filtered validity needs bounded repair. Projection fixtures close no physical/aperture-uncertainty claim. |
| F006 | `open` | Remains SCI-FRUIT-001-owned; no adaptive algorithm/default/threshold/add-back/stopping/bright-source uncertainty change or closure occurred. |
| F007 | `open_conditioned` | Enabled-zero rejection, disabled-zero no-work, and successful-return planned cardinality are boundedly demonstrated. Actual incomplete/partial publication is not measured, active validation rejects disabled state, and count adequacy/default selection remain open. Configured counts are not scientific values. |
| F008 | `closed_bounded_owner_accepted` | Closure is only the admitted deterministic-local tranche: exact math/design/calibration/invalidity/identity/projection/count fixtures and proportional local gates. No astronomical, Unity, FRUIT, strict-FLT-parity, tolerance, or production closure is claimed. |

## Exact local gates at candidate `0bc4d95d6`

| Command/gate | Result |
| --- | --- |
| `env BUILD_DIR=/Users/gwilson/.codex/worktrees/f908/citlali-refactor/build BUILD_TYPE=Release BUILD_TESTS=ON tools/macos/configure-homebrew-build.sh` | pass after an approved dependency-network retry; initial sandbox DNS denial was a setup condition, not a candidate failure |
| `cmake --build build --target citlali_cli -j 8` | pass |
| `cmake --build build --target citlali_test -j 8` | pass |
| `cmake --build build --target citlali_science_map_fits_products_test -j 8` | pass |
| `cmake --build build --target citlali_safety_test -j 8` | pass |
| `build/bin/citlali --version` | `v4.0.0-3634-g0bc4d95d6` |
| six exact estimator/calibration/projection/alias FITS fixtures | 6/6 pass |
| complete FITS-product binary | 28/28 pass |
| twelve focused noise/count/provenance/source-table fixtures | 12/12 pass |
| unchanged bounded FRUIT configuration fixtures | 6/6 pass; no F006 claim |
| unchanged source-finder fixtures | 2/2 pass |
| complete core C++ binary | 560/560 pass; one unrelated disabled test reported |
| complete safety binary | 14/14 pass |
| `ctest --test-dir build --output-on-failure` | 602/602 executed pass of 603 registered; one intentionally disabled test |
| baseline Python unit suites | 95/95 pass, covering the historical validator contract |
| direct exact-candidate baseline compatibility probe | enabled: 0 errors; disabled: 9 errors — finding |
| product registry semantic load | pass; registry v2, 10 contracts, 1 science-map contract |
| independent math/projection/scale fixtures | all exact expectations pass |
| semantic digest recomputation | 11 identities; 5 registry checks; 0 mismatches |
| proposal structural/YAML validation | pass: parsed mapping, exact candidate, report SHA-256 binding, controlled axes, ordered F001--F008 set, F005 parity state, and F006 ownership |
| `$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all` | pass: 127 unit tests, four mode kits, 8/8 compatibility cases, 100% compact coverage, all boundary audits |
| `git diff --check d5015fe... 0bc4d95...` | pass |
| strict filter/operator/kernel parity | `scope_blocked_not_applicable_pending_FLT`, not passed or claimed |
| local/Unity astronomical reduction or external evidence | not run/requested |

No required local data was skipped. The one disabled repository test was not
counted as evidence or failure.

## Verdict, bounded successor, and production implication

Proposed controlled axes are:

- contract `approved`;
- implementation `nonconformant`;
- validation `failed` at the required package/count validation boundary;
- production `existing_use_only`; and
- verdict `amend`.

The exact candidate is not ready for integration. A bounded successor should:

1. derive member inclusion from the realized publication inventory for the
   current reduction instance, define exclusion/self/atomicity behavior, and
   add existing-tool validation of member hashes and HDU/table joins;
2. update the active baseline auditor and regressions to accept and require the
   approved disabled available-zero representation and validate new completion
   fields without treating plan arithmetic as measured partial completion;
3. make filtered stack-scatter and ratio validity use the actual stack state
   and distinguish `R<2` from unavailable/nonfinite scatter; and
4. separately correct the bounded Mapdiag labels if broad F004 closure is
   sought.

A product-policy decision is required before retaining the duplicate canonical
HDUs: either accept the quantified storage/hash-I/O cost because canonical
EXTNAME lookup is required, or use one legacy-named plane with canonical
identity/join metadata and update the product registry accordingly. This is a
compatibility/storage decision, not authorization for new numerical work.

That successor needs no count/default choice, dense covariance, sign stream,
new framework, FLT/Wiener/FRUIT algorithm work, astronomical evidence, or Unity
action. F005 remains FLT-conditioned; F006 remains SCI-FRUIT-001-owned; F007
count adequacy remains a separately admitted study. Production must not expand
and configured counts must not be treated as validated scientific values.

No delegation or subagent was used. No application code, tests, configuration,
frozen authority, canonical ledger, or canonical handoff was edited. No repair,
integration, push, Unity contact, reduction, external evidence request, or
production action occurred. Only this report and its machine-readable ledger
proposal are committed on the re-audit branch.
