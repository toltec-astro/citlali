# SCI-MAP-001 independent repair re-audit — 2026-08-05

## Audit identity and boundaries

This is a fresh, contract-first re-audit of the bounded SCI-MAP-001 repair. It
is an audit and evidence-disposition record, not a repair record. No Citlali
application source, algorithm, configuration, test, external evidence product,
coordination line, or canonical audit ledger is modified by this task.

- Required starting application branch: `codex/repair-sci-map-001`.
- Verified starting HEAD: `02b9eb303037eb3f3a7bb90838b478bb5262e346`.
- Verified starting worktree: clean, including untracked files.
- Repair candidate under audit: `ed28dafb37f9113c0d3c95297148157129a90886`.
- Candidate parent/governing implementation: `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Frozen campaign-package commit: `1b824f138754eeb1856ae5f102027db4b31598be`.
- Existing-corpus closeout commit: `02b9eb303037eb3f3a7bb90838b478bb5262e346`.
- Audit-only branch: `codex/reaudit-sci-map-001`, created from the verified
  starting commit after the state check.

The artifact snapshot at
`/private/tmp/citlali-scientific-audit-framework` was read only. Its Git
metadata was unavailable and no cleanliness or branch claim is made for that
directory. Missing original records were read directly from the immutable
shared Citlali Git object store; they were not restored into the snapshot.

Authority-integrity checks performed before contract derivation:

- the final mathematical audit at
  `b9e1e9a9b2fe492c402d8c7b0cf7e5a36c136a53` hashes to
  `6c8decef93f5607bc9e8dfc84e31aee67f45fa5c695fc80563c7e7064f78d556`,
  exactly matching the ledger;
- the frozen independent derivation at
  `c28f18ed089657dae278caba2d6d6d65c7ec72f4` hashes to
  `13dd5922bd492e381afcc3b015284216dde1ccc2199ece3d070ee577c7324381`,
  exactly matching the ledger;
- the bounded repair/re-audit handoff at the audit commit hashes to
  `c02d9ba0b8bb2d3d59c117affacb06016b34ed0b1f63c69f8e2b6f415f2019fd`,
  matching the frozen campaign authority; and
- the coordinator decision read at
  `a9ce0da5ae54164c8f7dbe6062d13649259cc76c` hashes to
  `687d2ea8fd3a5889f5457ebd7e6cc13827f53f4f7463015965b9054eea96778b`.
  The ledger records its path and commit but does not register a SHA-256 for
  that decision, so no stronger hash-comparison claim is made.

## Contract-first derivation

This section was fixed before the repair implementation and its tests were
assessed. Authority order is: the original independent derivation; the
accepted final audit; the project-owner F009/F010 decisions and bounded
handoff; the living scientific conventions, architecture, product registry,
and ADR 0009; then the frozen campaign protocol for evidence-only bounds.
Where an older record omits a later dependency, the living status and campaign
state govern. In particular, F013 is conditioned on ALIGN as well as CAL, AST,
PTC, and VAL.

### Conditioned statistical model

Let `y` be the admitted processed samples after the explicitly named upstream
selection, `X` the response/design operator, `Q` the selected weighting or
normalization operator, and `C = Cov(y)` the covariance conditioned on all
selected, fitted, or data-derived state. The general linear estimator is

```text
F = X^T Q X
A = F^+ X^T Q
m_hat = A y
Cov(m_hat | fixed selected state) = A C A^T.
```

Only when `Q = C^-1` on the identifiable subspace and the required covariance
conditions hold does the normal matrix have a precision interpretation. If
selection, weights, calibration, pointing, or response are themselves random,
the displayed covariance is conditional; an unconditional result additionally
requires their covariance and cross-covariance terms.

For the approved ordinary naive, array-grouped Stokes-I lane, hard assignment
to a pixel gives

```text
n[p] = sum_i G[p,i] * u[i] * y[i]
q[p] = sum_i G[p,i] * u[i]
k_num[p] = sum_i G[p,i] * u[i] * kernel[i]
m[p] = n[p] / q[p]             when q[p] is finite and positive
kernel_map[p] = k_num[p] / q[p] under the same support.
```

`G` is the named sample-to-pixel projection and `u` is the realized
`weight_I` coefficient at the declared lifecycle stage. The repair preserves
the established operation order `Q += u`, `N += u*signal`, and
`K += u*kernel`. `u` is a nonprecision gridding/normalization coefficient by
default. Its inverse-squared signal unit does not prove that `q` is a marginal
precision. With fractional projection or correlated samples, the formal
variance generally differs from `1/q` and can be non-diagonal.

The response is `R = A X`. A persisted kernel is a response tracer only when
its injected tracer is the declared `X tau` under the same operator. A finite
kernel-like plane alone does not establish response identity.

### Identities, shapes, WCS, and indexing

The admitted observation is one immutable ordered bundle. Its identity
contains grouping, ordered slot, array/network/detector or group identity,
Stokes identity, applicable frequency, signal unit, estimator, response and
required companions, frame, projection, epoch, orientation, scale, reference
world coordinate, reference pixel, shapes, and all versioned coefficient,
support, validity, and non-finite policies. Full-precision typed values are the
admission authority. A narrowed legacy WCS adapter is one-way and cannot prove
identity equality.

The validated component is typed Stokes index `0`, label `I`. FITS/WCS pixel
coordinates are one-based; in-memory pixels, map slots, arrays, detectors, and
Stokes component indices are zero-based unless a product explicitly says
otherwise. A FITS STOKES-axis physical-coordinate encoding is a serialization
fact and must not be conflated with the typed component index.

For coadd shape `(R_c, C_c)` and observation shape `(R_o, C_o)`, the only
approved transform is centered integer embedding:

```text
R_c >= R_o, C_c >= C_o
(R_c - R_o) and (C_c - C_o) are even
delta_row = (R_c - R_o) / 2
delta_col = (C_c - C_o) / 2.
```

The complete observation block must be in bounds. Frame, projection, units,
orientation, CRVAL, scale/matrix and other applicable WCS parameters must
match. In FITS one-based coordinates the only permitted shape-related WCS
differences are NAXIS and

```text
CRPIX1_coadd = CRPIX1_observation + delta_col
CRPIX2_coadd = CRPIX2_observation + delta_row.
```

That card relation is exact. The registered all-pixel sky-coordinate residual
is at most `1e-12 degree`. General reprojection, interpolation, fractional
shift, best-effort WCS matching, and implicit source recentering are outside
the contract. The signal-centering operator is `L = I`: constants and offsets
are neither mean-subtracted nor recentered.

Any bundle mismatch rejects the whole observation before numerical planes,
WCS/grouping, membership, observation numbers, exposure/count state, product
inventory, or provenance mutate. A failure in the last slot must leave all
coadd-owned bytes unchanged.

### Contribution, support, and validity

The stages are distinct and ordered:

1. `geometric_hits_I` counts finite, in-bounds sample/detector projections
   before upstream eligibility or mapmaker selection.
2. Upstream eligibility applies the named detector/APT and sample validity
   contract. `upstream_eligible_exposure_I` accumulates `dt = 1/d_fsmp` in
   detector-seconds for those terms.
3. An ordinary numerical contribution requires upstream eligibility, a finite
   strictly positive coefficient, finite signal, and finite declared numerical
   companions. `contributing_hits_I` counts precisely those terms.
4. An explicitly invalid term is skipped before its numerical payload is
   evaluated. An eligible term with a non-finite required payload is a
   fail-closed execution inconsistency, not a zero-weight contribution.
5. `retained_exposure_I` accumulates detector-seconds for contributing terms
   and is retained only on normalization support.

For each coefficient plane, select finite strictly positive values and sort
ascending. If their count is `N > 0`, use the zero-based index

```text
k = floor((floor(0.75*N) + N) / 2).
```

The empty-input threshold is zero. Otherwise the normalization threshold is
the selected coefficient times `coverage_cut/10`, and the science-policy
threshold is the selected coefficient times `coverage_cut`. Both masks require
an explicit finite-positive coefficient and `coefficient >= threshold`.
`!(coefficient < threshold)` is inadmissible because it accepts invalid IEEE
states.

The persisted facts are irreducible:

| Product | Dtype/unit | Contract meaning |
| --- | --- | --- |
| `geometric_hits_I` | `int64`, count | finite in-bounds projections before eligibility |
| `contributing_hits_I` | `int64`, count | terms admitted by the ordinary predicate |
| `coadd_observation_count_I` | `int64`, count | admitted observation maps contributing to a coadd pixel; coadd only |
| `upstream_eligible_exposure_I` | `float64`, detector s | eligible detector-sample dwell before contribution/retention |
| `retained_exposure_I` | `float64`, detector s | contributed dwell retained by normalization support |
| `normalization_support_I` | `uint8`, dimensionless | numerical division/population authority only |
| `science_policy_support_I` | `uint8`, dimensionless | separate full-cut policy result |
| `science_valid_I` | `uint8`, dimensionless | only authoritative raw science-validity mask |

The raw validity identity is exactly

```text
science_valid_I = normalization_support_I
                  AND science_policy_support_I
                  AND finite(signal_I and every declared companion)
                  AND admitted bundle identity.
```

`coverage_I` is a bitwise compatibility alias of `retained_exposure_I` in
detector-seconds. `coverage_bool_I` is a deprecated bitwise alias of
`science_policy_support_I`. Neither alias is a validity authority. The bundle
and aliases are explicitly unavailable for JINC, detector grouping, and the
other non-array groupings covered by the v1 absence rules; no ordinary
positive-coefficient predicate may be silently applied to JINC.

Signal and kernel are `mJy/beam` in the active profiles. `weight_I` is stored
in `1/(mJy/beam)^2` but remains nonprecision by default. Noise variance is in
`(mJy/beam)^2`; exposure is detector-seconds; counts and binary masks have
their respective count and dimensionless units.

### Observation coadd and noise operators

For centered embedding `J_o`, admitted membership `V_o`, and the realized
observation coefficient plane `u_o`, the approved coadd is

```text
N_c[p] = sum_o V_o[p] * u_o[J_o^-1(p)] * m_o[J_o^-1(p)]
Q_c[p] = sum_o V_o[p] * u_o[J_o^-1(p)]
K_c[p] = sum_o V_o[p] * u_o[J_o^-1(p)] * k_o[J_o^-1(p)]
m_c[p] = N_c[p] / Q_c[p] when Q_c[p] is finite and positive.
```

The kernel, retained exposure, observation count, and every noise realization
use the same admitted observation set, numerical-contribution mask, embedding,
and coefficient. The coadd validity states are evaluated and persisted
separately; a normalization-retained but science-policy-rejected term can
remain in the numerical mean while remaining explicitly low-confidence, but an
explicitly invalid term never contributes.

If `B` is the coadd operator applied to observation maps, then

```text
Cov(m_c) = B C_observation B^T.
```

The pixelwise weighted mean becomes an inverse-variance coadd only under the
separately proved marginal-precision and independence/covariance conditions.
Correlated GLS, negative GLS coefficients, coadd uncertainty, calibrated
significance, and covariance regularization are not authorized.

Pinned sign-randomized observation realizations must pass through the same
ordinary map operator `A`; coadd realizations must pass through the same `B`.
MAP-UNITY-PR1 permits an empirical coadd run to retain compact per-observation
empirical planes in the ordinary FITS and the separate 64-realization ensemble
only for the coadd. That storage decision is not proof that the observation
realizations used the same operator. A same-case serialized observation
ensemble is direct evidence; a byte-identical explicit non-coadd sibling is
supporting proxy evidence only unless every required identity and operator
input is independently bound.

### Provenance and downstream carriage

Requested, effective, observation-resolved, and realized state are one-way
authorities. Realized provenance must losslessly preserve algorithm/version
IDs, coefficient product and lifecycle stage, requested and realized cuts,
binary64 thresholds, positive-value count and selected index, comparison and
finite policies, per-state counts, companion inventory, full admitted identity,
observation membership and offsets, response identity, product inventory, and
the exact raw-parent/product digest. Display-rounded FITS cards do not replace
that record.

The threshold-input stages are exactly
`pre-observation-normalization-accumulated-coefficient` and
`pre-coadd-normalization-sum-of-admitted-observation-coefficients`. Published
stages are the corresponding post-normalization observation/coadd states, with
or without the named global empirical rescale.

Before filtering mutates a map, the raw F010 bundle is frozen. Filtered signal,
coefficient, F010 facts, and aliases carry `RAWSTATE=immutable_input` and the
same exact `RAWPDGST`. Downstream numerical, stencil, response, covariance, and
output-validity masks remain separate and may not promote a raw-invalid pixel.

### Numerical and evidence bounds

- Within a scan, requested sequential and parallel ordinary accumulation use
  one detector/sample-ordered primitive and are exact.
- Mutex-protected scan-farm commits may arrive in a different order. For each
  binary64 plane, comparison with the long-double sum of per-scan planes is
  bounded by `2*gamma_n*sum(abs(per_scan_value))`, where
  `gamma_n = n*epsilon/(1-n*epsilon)`. Integer fact planes are exact.
- The local truth suite must include direct well-conditioned small-matrix
  `A`, mean, and `A C A^T` comparisons at the registered `1e-12` scale while
  retaining the PTC/covariance condition.
- The frozen external campaign registers exact inventory, identity, topology,
  support, aliases, and WCS-card relationships; a `1e-12 degree` sky bound;
  and seq/OpenMP numerical regression bounds `atol=2e-8`, `rtol=1e-10`.
  Those broad regression tolerances do not relax an exact contract or the WCS
  bound.
- Normalized aggregate products cannot prove the scan-farm gamma bound. That
  lane needs run-produced pre-normalization binary64 per-scan planes and commit
  order; the exact local F011 gate remains authoritative if that external lane
  is unavailable.

### Finding-specific acceptance contract

| Finding | Required re-audit result |
| --- | --- |
| F001 | No shared-pixel race; sequential/parallel policy proved, including TSan or equivalent race evidence for the primitive. |
| F002 | Finite-positive support predicates and one persisted authoritative raw-valid mask; invalid payloads skipped before evaluation. |
| F003 | Full-precision identity/WCS and whole-bundle two-phase coadd admission; every mismatch fails before mutation. |
| F004 | Signal, kernel, and realization products use the same admitted map operator and coadd membership/support boundary. |
| F005 | Unexpected non-finite required values fail atomically; explicitly invalid hidden payloads cannot poison output. |
| F006 | Coefficients and products make no unconditional precision, inverse-variance, uncertainty, S/N, or significance claim. |
| F007 | Identity, response, units, hits/exposure/support/validity hierarchy, dtypes, names, and absence rules are complete and persisted. |
| F008 | Lossless one-way realized provenance and raw-parent digest are complete and tamper-rejected. |
| F009 | The approved atomic bundle, centered integer embedding, `L=I`, and normalized weighted-mean contract are implemented exactly. |
| F010 | All eight facts, two threshold rules, aliases, availability rules, and downstream raw-validity carriage are implemented exactly. |
| F011 | Exact-repair-SHA local truth, failure, product, provenance, deterministic/OpenMP, baseline, config, and no-broadening gates pass with no required-data skip or unexpected error. |
| F012 | Exact-candidate external evidence is authentic and sufficient for every claim assigned to it; unavailable lanes remain named limitations, not inferred passes. |
| F013 | Every conclusion remains conditioned on SCI-ALIGN-001, SCI-CAL-001, SCI-AST-001, SCI-PTC-001, and SCI-VAL-001. MAP evidence closes none of them. |

### Contract-first checkpoint: exact evidence questions

Before implementation inspection, the evidence questions are fixed as follows:

1. Does the exact candidate implement every F001-F011 contract without
   broadening arithmetic, registration, JINC, covariance, or empirical-noise
   policy?
2. Do local tests independently exercise the equations, atomic failure,
   non-finite state space, eight products, thresholds, aliases, provenance,
   raw-parent carriage, centered WCS, no-centering, and registered parallel
   policy at the exact candidate SHA?
3. Can the owner corpus be bound to the candidate binary and all seven case
   identities without the prepared raw/sample-ledger and wrapper/Slurm lanes?
   If not, which claims remain unsupported rather than failed?
4. Can S-X-SEQ's observation realization path be demonstrated from same-case
   serialization? If it is absent, exactly what does the S-E sibling establish,
   and what same-operator claim remains unproved?
5. Are every typed-to-FITS WCS sky residual and centered observation/coadd
   residual within `1e-12 degree`? Values above the registered bound are not
   eligible for tolerance reinterpretation.
6. Is typed `stokes_identity=0` correctly preserved as the zero-based Stokes-I
   identity, and does the frozen analyzer incorrectly require `1` by conflating
   typed identity with a FITS-axis convention? Conversely, is any persisted
   FITS STOKES-axis value independently nonconformant under an explicit product
   rule?
7. After separating authentic supporting evidence from unavailable direct
   lanes and protocol defects, is F012 sufficient for closure? Regardless of
   that answer, F013 remains conditioned on ALIGN, CAL, AST, PTC, and VAL.

## Code re-audit

The source tree at `02b9eb303037eb3f3a7bb90838b478bb5262e346` has no
application-source or test diff from the exact repair candidate; the subsequent
changes are documentation plus the frozen campaign package and its
existing-corpus closeout. The application source and tests are therefore the
exact `ed28dafb37f9113c0d3c95297148157129a90886` implementation. The
conclusions below were derived from that source rather than from the repair
commit message or repair handoff.

### Implemented contract that survives independent inspection

- Both ordinary nonpolarized entry points dispatch to one detector/sample-
  ordered primitive. It stages signal, coefficient, kernel, realization,
  hit, and exposure contributions locally and commits them under one shared
  mutex. Requested sequential and requested parallel execution therefore use
  the same within-scan arithmetic. The same critical section also contains the
  realization-cube merge.
- Detector/sample invalidity is tested before the hidden numerical payload.
  Eligible projections, coefficients, signals, kernels, realization signs,
  products, shapes, and sample rate have explicit finite/shape checks for the
  ordinary supported path. Coefficients must be strictly positive to
  contribute.
- Observation noise is observation-owned even when coaddition is enabled.
  Observation normalization applies the same support and division to signal,
  kernel, and each realization. Coadd preflight then admits the normalized
  signal, kernel, and realization bundle together, and coadd accumulation uses
  the same observation coefficient and centered embedding for each.
- Coadd admission compares the complete typed bundle, including binary64 WCS
  values and response digest, derives only the approved centered reference-
  pixel shift, validates all slots and pixels, checks proposed floating and
  integer additions, and only then commits. `L=I` is preserved. The staged
  execution plan is nothrow-move-committed after numerical admission.
- Normalization uses finite-positive coefficient selection, the two approved
  thresholds, separate normalization and science-policy masks, the exact
  eight F010 facts, bitwise aliases, and the required finite-companion
  conjunction. Coadd significance products are suppressed and coefficient
  metadata states nonprecision/default covariance-unavailable semantics.
- The canonical identity/digest and YAML provenance include full response,
  WCS, slot, policy, threshold, product, raw-parent, and coadd-membership
  state. Binary64 values are serialized with both round-trippable decimal and
  hexadecimal representations. Unsupported profiles take an explicit legacy
  lane with product-absence reasons rather than inheriting the v1 claim.

### Blocking implementation discrepancy: typed WCS is narrowed on output

`Engine::write_maps` explicitly converts the admitted binary64 pixel scale,
reference world coordinate, and reference pixel to the legacy `float` WCS
adapter before every FITS HDU is written
(`map_image_output_impl.h:249-260`). `fitsIO::add_wcs` then writes those
already-narrowed values (`fits_io.h:176-182`). Consequently, the lossless
typed sidecar is an admission/provenance authority, but the physical FITS WCS
is not a `1e-12`-degree representation of it for ordinary non-binary32-exact
sky coordinates and scales. The one-way nature of the adapter prevents it
from corrupting later admission, but does not satisfy the persisted-product
bound.

The local FITS test does not exercise this boundary. It calls the lower-level
product-HDU writer with an already-float adapter and uses exactly representable
`CRVAL1=123.25` and `CRPIX1=1.5`; it neither calls `Engine::write_maps` nor
compares typed and FITS all-pixel sky coordinates. This is a direct F007
persisted-identity nonconformance and an F011 gate omission. It does not undo
the otherwise exact F003/F009 in-memory coadd admission.

### Blocking implementation discrepancy: finite terms can overflow at merge

The ordinary primitive rejects non-finite inputs and checks each individual
`coefficient*signal` and `coefficient*kernel` product, but it does not check
the sum of staged finite terms or the subsequent addition to the live map.
`add_sparse_to_dense` lets Eigen combine duplicate triplets and then performs
an unchecked `dense += sparse`; the same helper is used for binary64 planes
and `int64` hit counts. A finite sequence with a sufficiently large magnitude
can therefore overflow a required `Q`, `N`, `K`, exposure, realization, or
count accumulator. No accepted numerical-domain upper bound excludes that
state. A non-finite coefficient accumulator is later converted to zero support,
while other non-finite companions can be detected only after normalization has
already mutated map planes. That is neither fail-closed nor atomic under the
approved finite policy, so F005 remains open; exact-count F010 claims also
remain conditioned on the absence of accumulator overflow.

The projection path has a related unbounded edge: it checks that a projected
coordinate is finite but calls `llround` before establishing that the value is
representable by `Eigen::Index`. The tests cover NaN and infinities, not a
finite out-of-range coordinate.

### Test-surface limitations found before evidence inspection

- The concurrent scan-farm/TSan fixture constructs maps with
  `with_noise=false` and calls the primitive with `run_noise=false`. Static
  inspection shows that realization commit is under the same mutex, and the
  sequential/requested-parallel equality test covers noise, but the actual
  concurrent race fixture does not exercise realization writes.
- The scan-farm numerical-bound fixture checks signal, coefficient, kernel,
  coverage/exposure, and counts, but not realization planes.
- The coadd atomicity helper hashes numerical planes, products, membership,
  exposure, and typed bundle identity, but omits the legacy `cmb.wcs` object.
  Source inspection shows WCS mutation is deferred until commit, yet the test
  does not itself prove the required unchanged-WCS assertion.
- No local test supplies multiple finite contributions whose staged sum
  overflows, a finite coordinate outside the integer conversion range, or a
  non-binary32-exact typed WCS passed through the production FITS writer.

### Stokes identity in the implementation

The typed map component is consistently the governing project convention:
zero-based `stokes_identity=0`, label `I`. The provenance serializer preserves
that integer losslessly. The production FITS adapter currently copies the same
zero directly to `CRVAL4`; the frozen analyzer derives its expected card the
same way, but separate analyzer assertions require the typed value to be `1`.
No governing project document authorizes silently changing the typed component
index to obtain that assertion. The evidence section adjudicates the analyzer
conflict separately from the WCS precision defect.

## Evidence re-audit

The owner-supplied corpus at
`/Users/gwilson/work_toltec/local_data/2026-ENG-citlali-MAP` was inspected
read only. No reduction was launched, no product or metadata file was changed,
and no multi-gigabyte artifact was copied into Git. The checks below are
bounded product/integrity tests, not a replacement campaign or generalized
corpus analyzer.

### Local gates on the candidate application tree

The local build identifies repository HEAD `02b9eb303...`; a path-by-path Git
comparison establishes that its application source and tests are byte-for-byte
the repair candidate while the later commits add only the campaign/closeout
records. The resulting exact-application-tree gates were:

- standard local Release test build and `citlali_cli`: pass;
- focused science-map truth binary: 29/29 pass;
- ThreadSanitizer-focused science-map truth binary: 7/7 pass;
- CTest: 589 registered, 588 enabled pass, zero failures; the one disabled
  test is the pre-existing unrelated
  `MapFitterLifecycle.ExactProductSequence`, not a skipped SCI-MAP required
  data lane;
- baseline-tool unit tests: 147/147 pass;
- full config preflight: 127/127 unit tests, four mode kits, eight compact-
  compatibility cases, complete surface coverage, and every authority/
  boundary audit pass;
- current frozen-package verifier: all 21 checksum members, driver self-check,
  and package checks pass.

These successes confirm the positive local behaviors listed in the code
review. They do not cover the production typed-to-FITS path, aggregate
overflow, finite-but-unrepresentable projection coordinates, concurrent noise
merges, or the omitted legacy-WCS atomicity byte, so they are not sufficient
to close F011.

### Corpus identity, integrity, and inventory

All seven accepted case roots contain an executable snapshot with SHA-256
`693c14898faa1d41a854030b86cdde2729bf121442eb8427feffb4d4e57686c5`.
Their accepted logs identify `v4.0.0-3628-ged28dafb`, the same candidate, the
declared sequential/16-thread selection, successful Citlali completion, and no
accepted-interval error or critical record. The 13-member minimal transfer
package verifies exactly; its `SHA256SUMS` file hashes to
`0909a558491301aea8ef56f53e6555aa3a43c1e1de872015774e2a316469f587`.
All installed case overlays match their transferred members. The later
canceled S-E-SEQ attempt is outside the accepted job interval and did not
change its accepted `redu00`; no result from the later transient root files is
used here.

The independently counted products are:

| Case | Observation map/noise | Coadd map/noise |
| --- | ---: | ---: |
| P-SEQ | 3/3 | 0/0 |
| P-OMP | 3/3 | 0/0 |
| S-C-SEQ | 6/0 | 3/3 |
| S-C-OMP | 6/0 | 3/3 |
| S-E-SEQ | 6/6 | 0/0 |
| S-E-OMP | 6/6 | 0/0 |
| S-X-SEQ | 6/0 | 3/3 |

No `.npz` sample ledger, completed raw-input manifest, owner-values record,
result collection, or recognizable case-wrapper, analysis-wrapper, Slurm
accounting, environment, or retrieval-integrity record exists under the seven
accepted case roots. This independently confirms that the returned trees are
authentic bounded run products, not the complete later frozen campaign
bundle.

### Product and numerical checks that pass

- All 45 map FITS files (36 observation and nine coadd) have the required F010
  planes for their scope. The four masks are binary; both compatibility aliases
  are bitwise exact; contributing hits never exceed geometric hits; retained
  exposure never exceeds upstream-eligible exposure; normalization support has
  finite positive coefficient; and the persisted science-valid mask equals
  both supports plus every directly serialized finite signal/kernel/
  realization companion. Twenty-seven maps have a direct 64-realization
  companion under the returned product-retention policy. No check failed.
- The P, S-C, and S-E sequential/OpenMP pairs comprise 18 common map files and
  261 image planes. Integer planes and finite masks are exact; 162 planes are
  byte/value exact; the largest floating absolute difference is
  `1.7053025658242404e-13` and the largest relative difference is
  `2.894365138691133e-13`. Every plane passes the frozen
  `atol=2e-8, rtol=1e-10` regression bounds.
- The six S-E-SEQ and S-X-SEQ observation maps have 96 common image planes.
  Every one is byte-identical. This is strong supporting identity evidence for
  the sibling realization lane, but it is not same-case S-X serialization.
- The six sequential S-C/S-X coadd maps were independently reconstructed from
  their two serialized observation maps using centered embedding, observation
  normalization support, `Q`, `N`, `K`, both exposure facts and all three
  count facts. Seventy-two plane comparisons are exact, including the two
  threshold-selected masks and both aliases. This product-level recombination
  does not reconstruct pre-normalization samples or prove the observation
  realization operator.
- S-X provenance records 384 generated observation realizations and 192
  generated coadd realizations, but only 192 realization images written. That
  cardinality is internally consistent with the three coadd-noise files and
  six absent observation-noise files.

### Evidence discrepancies and unavailable lanes

**Independent raw/sample authority.** MAP-UNITY-ED1 permitted a later compact
successor protocol instead of exhaustive actual-data term retention, but it
preserved automatically generated raw manifests, hashes/statistics, and
deterministic actual-data traces. The full/all-PTC ED2 successor package was
prepared locally and never dispatched. The seven returned earlier cases
contain neither the original raw/sample-ledger authority nor the approved
compact successor authority. Consequently they cannot independently
reconstruct ordinary contributions, required realization companions,
raw-parent digests, or the scan-farm `2*gamma_n*sum(abs(...))` lane. This is an
unavailable evidence lane, not a reason to infer either pass or numerical
failure.

**Wrapper and Slurm authority.** Accepted logs and immutable executable/config
snapshots are adequate to establish bounded successful-run and candidate
identity. They do not establish the later frozen campaign's completed owner
values, preflight/submission exit, wrapper exit, Slurm state/accounting and
MaxRSS, environment snapshot, collection, or retrieval-integrity chain. Those
lanes remain unavailable and cannot be converted into passes from fallback
log lines.

**S-X observation realizations.** MAP-UNITY-PR1 validly explains the storage
policy: coadd empirical mode retains compact observation empirical planes and
the separate ensemble only for the coadd. It does not prove that the 384
same-case observation realizations passed through the same `A` later used by
the coadd `B`. The S-E sibling's six noise files, 96 byte-identical common
planes, and S-X realized cardinality are supporting evidence only. Because no
same-case serialized observation ensemble or independent sample/trace
authority exists, the F004 external same-operator question remains open.

**Typed-to-FITS WCS.** For S-X observation 152390/a1100, all 183,365 spatial
pixels were compared between the binary64 typed TAN WCS and the persisted
FITS WCS. The separation ranges from `1.7988824554319328e-5` to
`1.8081951134495923e-5 degree`; the reference-pixel value is
`1.8035411997018352e-5 degree`. The registered bound is `1e-12 degree`, so a
single such product fails by more than seven orders of magnitude. By contrast,
the persisted observation and coadd WCS have the exact centered CRPIX relation
and zero sky residual after applying `delta_col=3`. Thus internal persisted
centering agrees while both persisted WCSes share the same unacceptable loss
relative to typed authority. No tolerance is changed or reinterpreted.

**Threshold FITS cards.** The six independently recombined S-C-SEQ/S-X-SEQ
coadd products expose a second serialization discrepancy. All 18 checked
`WTTHRESH` cards (normalization, policy, and policy alias) differ in binary64
value from the exact realized threshold; for example
`0.014534135095329` is persisted instead of
`0.01453413509532904`. Policy and alias cards agree with one another, and the
YAML sidecar retains the exact decimal/hex value, but the frozen analyzer
requires exact card equality. That registered check fails; it is not waived
as display formatting.

**Stokes identity.** The governing convention explicitly defines the validated
component as typed index `0`, label `I`. The candidate and returned provenance
conform. The frozen analyzer also computes its expected `CRVAL4` from that
recorded zero, but two independent assertions in the same analyzer require
typed `stokes_identity==1`. Those assertions contradict the governing typed
contract and the analyzer's own adapter calculation; they are verifier defects
and cannot justify changing the typed identity. The product registry requires
a STOKES axis but registers no numeric `CRVAL4` rule, so this re-audit has no
local authority to declare the persisted zero independently conformant or
nonconformant as a FITS physical-code convention. That separate unregistered
question is unnecessary to this disposition: the analyzer's typed-index
assertions are invalid, while the WCS and threshold failures remain real.

### F012 sufficiency decision

F012 evidence is **insufficient**. The corpus authentically establishes exact-
candidate successful execution, product cardinality, strong sequential/OpenMP
agreement, internally coherent F010 planes, exact serialized-map coaddition,
and supporting empirical-product identity. It does not supply either approved
independent raw/trace authority, the frozen wrapper/Slurm chain, direct
same-case S-X observation realizations, a passing typed/FITS WCS bound, or
exact threshold cards. Correctly rejecting the analyzer's Stokes-index
assertions removes a verifier false failure; it does not manufacture any
missing lane or cure the independent serialization failures.

## Finding dispositions and coordinator decision

The following are proposed dispositions for coordinator review. “Close” means
the original finding's scoped defect/decision is accepted at the candidate
application tree; it does not close a dependency or authorize a consumer.

| Finding | Re-audit result | Basis |
| --- | --- | --- |
| F001 | **propose close** | One deterministic ordinary primitive covers array and detector grouping; sequential/requested-parallel results including realization planes are exact; the TSan target passes 7/7. Concurrent scan-farm realization writes are not separately stressed, but source places them under the same lock. |
| F002 | **propose close** | Finite-positive support, pre-payload invalid skip, distinct support masks, and authoritative science validity are implemented and pass local plus 45-product checks. Downstream authorization remains separate. |
| F003 | **propose close** | Full binary64 bundle identity and two-phase centered coadd admission reject mismatch before mutation. Persisted WCS loss is assigned to F007, not used to weaken typed admission. |
| F004 | **remain open** | Source/local fixtures now use one signal/kernel/realization operator and publishable policy, but direct same-case S-X observation-realization evidence and the NOI empirical/covariance boundary are not closed. |
| F005 | **remain open** | Finite individual inputs can overflow staged/live floating or integer aggregates, and finite projected coordinates can reach `llround` outside representable index range. Required fail-closed atomicity is incomplete. |
| F006 | **propose close** | Products explicitly call `weight_I` a nonprecision normalization coefficient, suppress coadd significance/uncertainty, and retain covariance-unavailable/PTC conditions. This does not close PTC. |
| F007 | **remain open** | The new product hierarchy is otherwise complete, but production FITS WCS does not preserve admitted identity within `1e-12 degree`; exact threshold-card serialization also fails the frozen gate. |
| F008 | **propose close** | Versioned one-way sidecars preserve exact identity, membership, policies, thresholds, products, and raw-parent digest; local cardinality/tamper tests pass and same-candidate sidecars returned. Missing independent raw authority remains F012. |
| F009 | **propose close** | The approved strict admission, centered integer `J`, `L=I`, nonprecision normalized mean, invalid exclusion, and coadd atomicity are implemented; six returned coadds recombine exactly. |
| F010 | **remain `addressed_pending_reaudit`** | All returned facts/masks/aliases and practical thresholds agree, but unchecked ordinary count/value overflow prevents an all-admissible-input exact claim, and the frozen exact `WTTHRESH` round trip fails. |
| F011 | **remain open** | Every executed local gate passes, but required tests for production typed/FITS WCS, aggregate overflow/index range, concurrent realization merge, realization gamma bound, and complete unchanged legacy WCS are absent. |
| F012 | **remain open; insufficient** | The bounded seven-case evidence is authentic but the raw/trace and wrapper/Slurm lanes are unavailable, S-X direct observation realizations are absent, and registered WCS/threshold checks fail. |
| F013 | **remain open** | SCI-ALIGN-001, SCI-CAL-001, SCI-AST-001, SCI-PTC-001, and SCI-VAL-001 remain governing dependencies. MAP evidence closes none of them. |

No unresolved scientific or operational owner choice is needed to issue this
result. In particular, typed Stokes index zero is already governed; a future
physical FITS-axis-code rule may be registered separately without changing the
present rejection.

### Package-level decision

- Contract status: `approved` (unchanged).
- Implementation status: `nonconformant`.
- Validation status: `in_progress`.
- Production status: `existing_use_only`.
- Verdict: `amend`.
- F012: `insufficient`.
- Re-audit: completed for this candidate with blocking findings retained; no
  integration or production expansion is authorized.

The minimum coordinator-facing blockers are the F005 aggregate/index failure
surface, the F007 typed/FITS and exact-card persistence failures, missing F011
tests for those states, incomplete F004 direct/NOI evidence, F012 bundle
insufficiency, and all five F013 dependencies. The attached machine-readable
proposal is for later coordinator review only. This task does not edit the
canonical ledger or coordination line, integrate a branch, dispatch another
task, request Unity work, or authorize a repair.
