# SCI-CAL-001 exact-repair re-audit

Date: 2026-08-08

Audit role: fresh independent re-audit of the pushed one-commit repair; the
repair narrative and test summary were treated only as claims to falsify.

## Disposition

- Verdict: `amend`.
- Contract: `approved`, subject to the authority chronology below.
- Implementation: `nonconformant` as a complete SCI-CAL-001 repair.
- Validation: `in_progress`.
- Production: `fail_closed`; no production-state change is proposed.
- Re-audit result: the fixed atmosphere operator is structurally faithful, but
  the package is not ready for coordinator acceptance, Unity execution, merge,
  or downstream launch.

The core F001 missing-airmass equation is addressed and the specific F002
positive-opacity unity plateau is removed. F002 is eligible for narrow
implementation closure after coordinator review. F001 remains open because its
closure gate also requires kernel/weight propagation and exact-SHA Unity
evidence. F003--F005 and F007--F010 remain open; F006 has an owner decision but
the implementation violates it.

## Identity and authority

All identity checks passed before substantive review.

| Item | Verified identity |
| --- | --- |
| Repair commit / pushed branch | `7894346a91fa78ceb2a8b3d625335f466e5e1756`; live `origin/codex/repair-sci-cal-001-successor` returned the same SHA |
| Sole parent | `46ad23888a40f5102cdfd50c06e49a549bdf8a20` |
| Repair tree | `991f96c64e4d2d973ed5fc02630bfe29149109d9` |
| Frozen coordination object | commit `dd5894679bf12bf4a5fb551e871b3c6010ef9b9b`, tree `e87b507a6dc5246da0f65e563d96b94824e61ba1`; object rehash exact |
| Frozen atmosphere object | commit `7156881bd1a47e8cece97b8c541a013c93ac03e1`, tree `316c5c5a0188ead742f55e21ae1bd62a89e02677`; object rehash exact |
| Initial worktree | clean, detached at the repair commit; audit branch was then created without changing source |

The applicable audit, independent core, coordinator decisions and amendments,
bounded repair/re-audit handoff, canonical ledger entry, SCI-CAL handoffs, late
ALIGN XAUD, atmosphere owner decision, machine contract, manifests, and
engineering-evidence records were read from the two frozen objects. Principal
frozen identities include:

- audit blob `cde8e23a36af164c0e3de0ac1f06cd4e27dbbf22`, SHA-256
  `957ed71d1432ad67fe582d6137fbe72c52e82a31f3199331a94ab7b39490d376`;
- independent-core blob `c8722b0888374871c9114304ba97af928435498f`,
  SHA-256
  `106755520b048f601bc60fd04e7b6020e6fa470480ac3105fa7ba269c730a4fe`;
- canonical-ledger blob `6330df5915bae0d0d345d747d6d6334eeb5fbf6c`,
  SHA-256
  `9f120b1b6ae29d0d3ee72ab6f220c130dbe32a7321bb5944f1899a2ab7918ba7`;
- bounded-handoff blob `4b0fbf41b2a1f35bd4f4c7f2781c31b93240229f`,
  SHA-256
  `9d2c0ae89244d355070d6b300f431ac1799787b835c7e4cb76c8d7fc51cde106`;
- fixed-owner-decision blob `a9f70b813c370ad3fa505fffa64aec8cd0689c66`,
  SHA-256
  `c43aa932c633e336497547730f73278d3a5cf70d2a5fcfb19049d967c79dd469`.

Authority chronology matters. The later fixed-owner decision and machine
contract in `7156881...` supersede the earlier q0-only repair model and the
coordination ledger's then-unresolved operator selection. They do not supersede
the fail-closed production state, observational-evidence gates, identity,
uncertainty, provenance, or ALIGN dependencies. The `7156881...` package README
still says no operator/domain is selected; that narrative is stale relative to
the later dated owner decision and contract and should be corrected by the
authority owner, not by this audit.

## Re-derived approved operator

For TolTEC array `b`, admitted spectral index `alpha`, and nonzero opacity
anchor `q_j`, let `L[b,alpha,j](e)` be the shape-preserving elevation PCHIP of
the frozen line-of-sight optical-depth nodes, with `L_0(e) = 0`. For
`q_(j-1) <= q <= q_j`,

```text
w = (q - q_(j-1)) / (q_j - q_(j-1))
L[b,alpha](q,e) = (1-w) L[b,alpha,j-1](e) + w L[b,alpha,j](e)
T[b,alpha](q,e) = exp(-L[b,alpha](q,e))
C[b,alpha](q,e) = exp( L[b,alpha](q,e))
```

The realized code multiplier is

```text
y[i,t] = x[i,t] * U[i] * flxscale[i]
                     * C[array(i),alpha](tau225, elevation[t])
```

where the admitted successor requires `U[i] = 1` for top-of-atmosphere,
point-source-peak `mJy/beam`. The nodes already contain full sample-airmass LOS
depth, so `X_ref = 0` and no second airmass factor is applied.

The exact opacity anchors are `0`, `0.0504874104674104401`,
`0.0883393725904400573`, `0.15`, `0.158313198574890929`, `0.20`, and `0.25`.
Support is closed at `0 <= tau225 <= 0.25` and
`25 <= elevation_deg <= 80`. The quality split is metadata only:
`[0,0.15]` is `science_qualification_regime` and `(0.15,0.25]` is
`engineering_availability_regime`; one operator is used on both sides of
`0.15`. Supported alpha is exactly `{-1,0,2,4}`; omission selects zero and no
alpha interpolation/extrapolation is permitted.

Independent dense evaluation of all 12 selected surfaces found positive finite
transmission/correction, exact zero and source nodes, no opacity/elevation
monotonicity violation, no `0.15` switch, transmission range
`[0.378780089589124,1]`, and correction range
`[1,2.6400542887160063]`. Value continuity follows from shared anchors;
opacity-derivative continuity is not a gate.

This verifies representation of the selected model, not atmospheric truth.
The contract itself records `primary_holdout_fidelity_pass=false` for the
inherited low-opacity study, labels the engineering `0.532005%` result as
representation evidence only, and contains no observational calibration
validation. Neither the quality label nor the implementation's `CAL.VALID`
establishes scientific accuracy.

## Artifact and reproducibility audit

The committed contract and node table are byte-identical to their frozen
`7156881...` versions.

| Artifact | Git blob | SHA-256 / result |
| --- | --- | --- |
| Fixed operator contract | `e182fec875227eb63e5dc72ed9ee390fb66ba441` | `7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a` |
| Node CSV | `e939ee4a66e71a18b9bece7c3daf8c4a9d374ebd` | `fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f` |
| Generated C++ header | `68482584d94d3cbd8714a9d93da2b2245039cb77` | `d322bdc863ccb1292325c739865f772ef53f4e9f4101967752027ea0a2413262` |

The CSV has the exact ten-column schema, 1,368 rows, and 72
`(anchor,array,alpha)` series: 36 31-node low-anchor series and 36 7-node
TAU025 series. It covers three arrays, four alphas, and six nonzero anchors;
the analytic zero anchor is not duplicated as CSV rows. Stored corrections are
exactly `exp(LOS tau)` to the tested binary64 calculation.

The checked-in generator deterministically validates final contract/node bytes
and regenerates the C++ header. It does not regenerate the scientific
contract/nodes from primary evidence. That assembly script exists only in the
frozen atmosphere object and depends on external TAU025 and sibling TolTECA
evidence, including
`/Users/gwilson/work_toltec/local_data/sci_cal_001_tau025_engineering_extension_002_root`.
No primary-evidence assembly replay was performed or is claimed.

## One-commit diff and intentional numerical change inventory

The entire `46ad2388...7894346a` diff was inspected: 42 files, 6,359
insertions, and 187 deletions. `git diff --check` passed.

The intentional numerical/scientific changes are:

1. replace q0/q25/q50/q75/q95 selection and degree-six legacy transmission
   polynomials with the 1,368-node fixed-DJF25, TolTECA-v1-passband,
   alpha-specific surface;
2. remove the finite-positive low-opacity unity plateau;
3. replace `exp(zenith-equivalent band tau)` with sample-elevation
   `exp(LOS tau)` exactly once at `X_ref=0`;
4. make opacity value-continuous through every exact anchor, adding explicit
   `0.15`, `0.20`, and `0.25` anchors and eliminating an operator switch at
   `0.15`;
5. add separately precomputed alpha surfaces for `-1`, `0`, `2`, and `4`, with
   alpha zero as the omission default;
6. replace legacy silent selection/extrapolation with hard failure outside
   tau/elevation/alpha support;
7. consequently change all extinction-enabled TOD, maps, Beammap fits and
   derived `flxscale`, and downstream factor-dependent values;
8. change compatibility `fcf` to contain target-unit transfer times the mean
   new LOS correction while still excluding `flxscale`;
9. change `MEAN_TAU`, `MEAN_TAU_a*`, and Beammap `a*_tau` values and meaning
   from zenith-equivalent band opacity to LOS depth evaluated at mean
   elevation; and
10. add numeric alpha, tau, `X_ref`, maximum-tau, quality-boundary, and validity
    values to raw/FITS/NetCDF/APT metadata.

The change is missing an entry in `validation/intended_science_changes.json`
and has no successor validation epoch or accepted run. The existing
science-change ledger validates structurally but therefore does not register
these intentional changes.

No prohibited mature ALIGN/AST/PTC/MAP/NOI/JINC/Wiener/fruit-loop scientific
algorithm was modified. RTC changes are confined to CAL evaluation ordering;
map, Beammap, and TOD changes are metadata consumers. No Unity access,
reduction, human/scientific external coordination, external write, application
repair, push, merge, or downstream launch occurred. The required live-origin
identity check and the approved build-dependency retrieval did make read-only
network requests to origin/GitHub; this disclosed network-read scope is not an
external message or mutation. Application work nevertheless crossed the repair
handoff's stop condition because APT identity was not proven before
implementation proceeded.

## Findings F001--F010

| Finding | Implementation result at `7894346...` | Formal disposition / dependency |
| --- | --- | --- |
| F001 | Core LOS/sample-airmass equation and fixed pivot are addressed. Signal is multiplied by `exp(LOS)` once. No kernel or conditional-weight propagation fixture/implementation closes the full gate. | `open`; exact-SHA Unity and ALIGN-conditioned sample eligibility also remain. |
| F002 | Exact zero, positive low-opacity behavior, every anchor/end point, continuity, and monotonicity conform for the selected operator. | Recommend narrow implementation closure after coordinator review; do not infer model fidelity or observational validity. |
| F003 | Tau/elevation/alpha/LOS/array checks improve, but `flxscale`, unit factor, `sens`, responsivity, and beam validity remain incomplete; whole-observation pre-publication atomicity is absent. | `open`, P0; local and Unity invalid-state gates incomplete. |
| F004 | No verified-row-order proof or explicit acquisition-key join exists. Raw `RoachIndex` is read then unused; APT remains positional and `kids_tone` is discarded. | `open`, P0; UID/order/permutation and real-artifact evidence absent. |
| F005 | No conditional-variance transfer, nuisance availability/value/uncertainty/provenance, or correlation/covariance model was added. Zero source uncertainty remains admitted. | `open`; analytic/Monte Carlo and real nuisance evidence absent. |
| F006 | Owner policy is resolved, but legacy `MJy/sr`, `uK`, and `Jy/pixel` modes are still accepted and calculated. | `addressed_owner_decision_implementation_nonconformant`; nondefault units must fail closed. |
| F007 | Alpha plus atmosphere operator/node/passband/profile/tau/regime fields have a partial requested/effective/observation/realized chain. APT/join lineage, multiplier reconstruction, factor units/exclusions, response, uncertainty, and nuisances are absent. | `open`; exact-SHA real-product round trip absent. |
| F008 | Actual signal composition is visible in code, but `flxscale`, responsivity, `sens`, and compatibility `fcf` still lack the complete approved persisted factor table and propagation tests. | `open`; RTC/PTC recipient disposition remains required. |
| F009 | Deterministic operator/schema coverage is much better, but the mandatory falsification, synthetic/covariance, exact-SHA Unity, and astronomical standard-source matrix is incomplete. | `open`; local test success cannot close it. |
| F010 | CAL consumes sample `TelElAct` but does not prove timestamp, duration, gap/interpolation origin, original/synthesized eligibility, or exact detector/sample identity. | `open_conditioned`; late ALIGN XAUD remains held for re-audit and recipient disposition. |

### Closure-gate reassessment

| Gate | Re-audit result |
| --- | --- |
| A. Analytic identities and interfaces | Partial. Zero/pivot and scalar signal behavior pass for the fixed operator. Kernel, conditional variance/weight, rank-one common-gain, approved-unit, and APT row/key semantics are not closed. |
| B. Deterministic CAL-T1--T6 | Partial. Operator/config/provenance fixtures pass, but the complete 64-epsilon scalar matrix, covariance tolerances (relative Frobenius `1e-12`, absolute `1e-14`), two-observation order/state test, lossless four-state round trip, and sequential/OpenMP plus two-iteration bitwise matrix are absent. |
| C. Synthetic calibrated sources | Missing. No point and uniform templates over at least three airmasses, deterministic recovery, or 1,000-draw mean/covariance checks (`<=3` and `<=5` Monte Carlo standard errors) bind this repair. |
| D. Blank and nuisance controls | Missing. Zero/Gaussian blank mean and airmass-slope checks (`<=3` predicted standard errors) and dense common-nuisance covariance are absent. |
| E. Exact-SHA Unity | Missing; no human-run bundle binds this repair, binary, dependencies, inputs, products, and logs. |
| F. Astronomical standard | Missing; no independent record demonstrates per-measurement residual `<=3`, array weighted mean `<=2` standard errors, or ratio-airmass slope `<=2` standard errors. |
| G. Recipient dispositions | Missing for the late ALIGN XAUD and CAL-to-RTC/PTC/VAL/MAP/FLT/MODE/BEAM handoffs. |
| H. Fresh exact-repair re-audit | Performed here with a nonconformant `amend` result; coordinator disposition remains pending. |
| I. Production-state authority | Not exercised. Only the coordinator/scientific owner may change production state, and this report recommends no change. |

### Material implementation failures

- Observation setup checks only mean elevation. Actual sample elevations are
  validated per scan, after earlier valid scans can already publish RTC
  outputs. A late invalid scan can therefore leave partial calibrated products.
- `CAL.VALID` is mis-scoped: enabling extinction sets it true from tau support
  alone; disabling extinction sets it false even when flux calibration is
  otherwise valid. Product metadata labels it as generic calibration validity.
- `calibrate_tod` checks vector length but not finite/valid unit factors or
  `flxscale` before mutation.
- APT consumption is positional. The retained `uid` is not joined, raw
  `Header.Toltec.RoachIndex` is unused, and `kids_tone` is not retained by the
  CAL APT column set. No row-mode rejection or keyed permutation-invariance
  path exists.
- Conditional uncertainty should transform as `v' = a^2 v` and
  `w' = w/a^2`; no such explicit CAL transfer or nuisance separation is
  implemented. PTC instead combines scalar compatibility `fcf` with `sens`.
- Beammap `uncertainty_mJy` is serialized but not propagated; zero is accepted,
  contrary to the decision that missing uncertainty is unavailable, never
  zero.
- The Beammap "apply once" test is algebraic rather than a production Beammap
  fit. The only pre-mutation invalid-factor test injects NaN LOS, not NaN
  `flxscale`, unit factor, `sens`, beam state, late-scan elevation, or APT
  identity mismatch.
- Raw-v3 tests cover operator/contract/node/passband/profile identities plus
  alpha, regime, and validity. FITS assertions cover only operator ID,
  effective/realized alpha, tau, quality regime, `X_REF`, and tau-only
  `CAL.VALID`; they omit contract/node/passband/profile digests,
  requested/default alpha, reduction regime/maximum tau, tau frame, and
  validity reason. The new Beammap APT metadata has no direct test.

## Local validation evidence

| Gate | Fresh result | Qualification |
| --- | --- | --- |
| Generator `--check` | pass; exact contract/node digests, 1,368 rows, 72 series | Final-artifact/header reproducibility only |
| Generator unit tests | 3/3 pass | No skips or unexpected error-level output |
| Focused C++ CAL/config/APT | 16/16 pass | Required UID/order, unit, uncertainty, late-domain cases do not exist |
| Selected FITS writer test | 1/1 pass | The CAL-specific split-Beammap metadata test ran only in full CTest; its asserted field set is incomplete and codifies tau-only `CAL.VALID` |
| `citlali_cli` Release build | pass | Local binary SHA-256 `8425530a6c0407a59f85fd17cb5964adc0d0e5f0efe4170f3ecb1ff4c5d84d61` |
| Full CTest | 635 runnable pass, 0 fail; 1 disabled | Disabled `MapFitterLifecycle.ExactProductSequence` did not run and is not counted as a pass |
| Config preflight `--require-all` | pass; 127/127 units, 4/4 kits, 8/8 compatibility, zero skips, 100% coverage | No required-data skips |
| Focused raw-provenance audit | 90/90 pass | Includes all-valid and selected invalid semantic controls, not a produced-product round trip |
| Baseline tools | 174/174 pass | Tool/schema tests, not same-SHA reductions |
| Product-contract units | 23/23 pass | No exact-SHA reduction product was available for the actual product validator |
| Validation/profile ledgers | pass; 60 records, 4 active profiles | None binds this repair SHA; active epoch remains historical Phase 4 |
| Science-change ledger validator | pass; 3 changes, 5 integration commits | This numerical change is absent from the ledger |

Focused deterministic gates were run before the broader build, CTest, config,
baseline, and product gates.

The first sandboxed bootstrap attempt failed because GitHub DNS was blocked;
the approved network retry configured successfully. The successful build used
AppleClang 21.0.0, CMake 4.3.0, fetched Tula commit
`f30f81d97c44bd79618273bb842302ef839c6ab1` and KIDs commit
`04088da182622c3e879f04314974a7c0d60ee2d6`, plus build-local dependency
patches. Third-party deprecation/compiler warnings were present. No Citlali
test failure, required-data skip, or unexpected runtime error-level message
occurred. This local dependency state is not Unity dependency evidence.

Synthetic all-valid operator/provenance controls pass, as do local invalid
tau/elevation/alpha and NaN-LOS controls. There is no end-to-end all-valid
exact-SHA reduction control, UID mismatch/permutation/reorder control,
nondefault-unit rejection, full coefficient/beam invalidity control, or
conditional/nuisance covariance control. Product writers emit many atmosphere
identities, but tests do not establish complete product cardinality/digests or
lossless multiplier/APT/uncertainty reconstruction.

## Exact missing external/dependency evidence

Do not launch these gates on this nonconformant candidate without coordinator
direction. The missing evidence is nevertheless explicit:

1. A human-run `SCI-CAL-001-UNITY-001` bundle bound to exact source
   `7894346...`, executable digest, compiler, complete dependencies and
   build/runtime policy, ordered numbered configs and merged digest, raw input,
   APT, tau/passband identities, product inventories/digests, and complete logs.
2. Updated Unity cases: sequential/OpenMP equivalence; extinction off/on
   per-sample ratios for every array; two OOF airmasses with no state leak;
   production Beammap recovery; tau/elevation/alpha domain failures; NaN
   coefficient/beam failures before mutation; verified-row-order reorder
   rejection or explicit-key permutation invariance; and no partial required
   products. The old successful `MJy/sr`, `uK`, and `Jy/pixel` cases must be
   replaced by fail-closed controls under CAL-D003.
3. An independently modeled standard-source recovery record binding epoch,
   bandpass/color convention, beam/template, and full shared covariance, with
   the predeclared residual, weighted-mean, and airmass-slope thresholds.
4. Approved `SCI-ALIGN-001` implementation/evidence for exact detector and
   sample identity, trustworthy timestamps and elevation validity, duration,
   gap/interpolation origin, and original/synthesized eligibility, followed by
   proof that CAL consumes them timestamp by timestamp.
5. Recipient disposition for CAL handoffs to RTC, PTC, VAL, MAP, FLT, MODE, and
   BEAM, including the late held ALIGN XAUD.
6. A successor validation epoch, accepted same-SHA Point/OOF/Science/Beammap
   records, actual product-contract audit, and intended-science-change entry.

## Coordinator return

Stop for coordinator and scientific-owner review. This report proposes no
application repair, production authorization, canonical-ledger mutation,
external campaign, Unity action, merge, or downstream launch. The fixed
operator component may be retained for a bounded successor, but accepting
`7894346...` as SCI-CAL-001 closure would be unsupported.
