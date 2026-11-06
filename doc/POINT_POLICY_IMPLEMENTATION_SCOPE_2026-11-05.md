# Proposed POINT policy implementation scope

Prepared November 5, 2026 for Grant Wilson's review.
Disposition — proposal awaiting owner approval.

The [agreed decision table](POINT_METHOD_AND_USE_ACCEPTANCE_2026-11-04.md#agreed-decision-table)
is the policy input. Preparation is bound to clean mainline
`185ed7d9098f09f2444e6cfcaa7dc9f24050315a`, containing tested application
`fc9c1f2935b433782ae3cfa1a79e2ac0247dd8f1`.
Grant executes all builds, tests, probes, benchmarks, reductions and validation
runners on Mac and Unity. This proposal was prepared by static inspection.

## Proposed result

One bounded Citlali increment would publish an adopted POINT displacement
with its selection reason, contributing arrays and per-array quality records.
A separately bound TolTECA follow-up would consume that displacement.
Both parts need owner evidence before operational use.

The available measurements, eligibility and observation outcome have distinct
owners. A completed reduction can publish valid maps and a withheld automatic
pointing result. A required output failure propagates to the CLI.

## Citlali work

| Area | Proposed change and boundary |
| --- | --- |
| Setup and policy identity | Resolve the agreed policy identity, settings and required outputs with YAML setup. Record requested and effective values before reduction. Bind the existing fitter and final-map recipe by source/configuration identity. |
| Selection owner | Add one observation-scoped policy function taking per-array fit/shape assessments and returning the adopted displacement, participating arrays, pair distances and reason. Its inputs carry array identity, arcsecond units, common frame and explicit availability. Implement the table as the reviewable authority for branch behavior. |
| Background assessment | Consume the final map data and measured centroid already used for publication. Compute the fixed 30–60 arcsec annulus, RMS, robust scatter, map contrast and quadrant coverage once per array. Reuse the existing array worker boundary where appropriate. |
| Array availability | Trace declared offline arrays and failed/unusable fits through preparation, native MAP and feedback, and publication. Represent planned absence explicitly. All available arrays still receive reduction and diagnostics. Missing or corrupt required products fail through their existing error path. |
| Observer product | Version the existing POINT diagnostic/selection envelope and publish each array's state, widths, offsets, background values, policy settings, selected arrays and adopted measured displacement or withholding reason. Retain links to map and fit identities. |
| Per-array table | Reconcile the all-three-fit gate with the agreed one/two-array cases. Publish available per-array measurements and explicit array identifiers. Carry the adopted solution separately so all measured rows remain inspectable. Preserve unavailable errors explicitly. |
| Publication | Keep the observation's staged publication boundary. Bind policy results, table and fit/map identities before making the observation discoverable. A valid fit whose estimate is excluded remains available with its reason. |

The current implementation constrains the scope in three concrete places.

- `configured_point_weighted.py` requires all three array bundles for each
  weighted cycle. `configured_point_products.py` enforces the all-three
  empirical publication domain. Genuine offline-array support requires
  tracing that domain through setup and feedback as well as final reporting.
- `configured_point_table.py` currently requires all three accepted fits.
  The new policy can select fewer usable estimates even when all maps exist.
- The recorded operational TolTECA resolver averages the ECSV rows. A new
  report alone cannot make that resolver honor an excluded array.

Implementation should distinguish planned array absence, a scientifically
unusable fit and an unexpected file failure. The review should stop a broader
acquisition-policy or algorithm change for separate scope.

## Proposed details requiring approval with this scope

These are implementation proposals extending the agreed table.

| Detail | Proposal |
| --- | --- |
| Missing a1100 shape | Evaluate the beam trigger only from an accepted a1100 fit with finite principal-axis widths. Otherwise report shape assessment unavailable and apply the usable-array table, including its single-array rule. Never label unavailable shape nominal. |
| Policy provenance | Give the initial selection policy and report schema explicit versions, with the 5 arcsec, provisional 8 arcsec and coverage settings recorded in the effective plan. Policy changes produce a new bound realization. |
| Region sampling | Reuse the saved assessment's pixel-center geometry on the full annulus, including its part outside the image in the coverage denominator. Use four sign-defined tangent-azimuth/elevation quadrants and a deterministic half-open boundary assignment. |
| Background estimators | Reuse the saved assessment's unweighted RMS about the sample mean (`ddof=0`) and `1.4826 * MAD` about the median. Validity follows finite signal and the realized scientific-support plane. Existing reduction support cuts apply before region coverage is assessed. |
| Zero or invalid denominator | Report contrast unavailable with its cause when background RMS is zero or nonfinite. Such a diagnostic state supplies no new automatic contrast threshold. |
| Missing centroid | Report the background region unavailable with a missing-centroid cause. Existing fit validity determines whether that array supplies an offset. |
| Consumer safety | Make the versioned selection envelope authoritative for the new consumer. Preserve explicit handling of legacy products. A selection withholding or malformed new envelope must be visible to the consumer, with no fallback to averaging its rows. |

The 75% and 50% coverage requirements use inclusive comparisons. The radius
includes 30 arcsec and excludes 60 arcsec. Finite background structure remains
inside the RMS population. No data-driven clipping or ring expansion is
proposed.

## TolTECA follow-up

Bind an exact operational TolTECA v2 source before edits. The historical
consumer evidence identifies
`2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`; the implementation task must
resolve the intended current operational source. The local v3 checkout has a
different role. [Existing consumer evidence](../validation/point_handoff_2026-10-30/consumer_results.json)

The follow-up would read the selected measurement, honor accepted/withheld
status and surface array warnings. Its tests must exercise ordinary discovery,
source/date association, before/after pointings, User/Paddle composition and
one application of the established sign convention. AST owns applying the
resulting correction. The existing hold on nonzero applied POINT offsets
requires its own residual-to-measurement qualification.

The Citlali publication schema and TolTECA adaptation need one reviewed
interface agreement before the first application edit. Changes to telescope
control or installed runtimes require their own deployment work.

## Owner-run verification proposed for the implementation

| Gate | Required evidence |
| --- | --- |
| Decision table | All array-availability combinations and all three-array agreement patterns, including exact 5 arcsec, rotated 3–4–5 offsets, exactly/above 8 arcsec, ambiguous chains, a2000 outlier, unavailable preferred fit and lone shorter array. |
| Background | Exact annulus boundaries, translated centroids, valid structure and nonfinite pixels, zero RMS, off-image area, equality at 75%/50%, one deficient quadrant and insufficient coverage with an otherwise valid selected solution. |
| Route and publication | Planned offline array, failed fit with intact maps, required missing/corrupt map, selected and withheld outcomes, and required output write failure through the normal entry point. |
| Actual consumer | Selected offset used exactly once, all rows retained for inspection, withhold honored, legacy behavior explicit, malformed new records rejected and warnings exposed. |
| Software | Focused checks, enabled CTest, baseline-tool tests and full required config preflight under the representative Spack realization. |
| Representative POINT | Grant's matched 152389 M0–M5 recipe against the accepted performance baseline. Per-array maps and fits must agree while the new adopted-result envelope follows the table. Record full time and memory to detect reintroduced per-pass work. |
| Review | Independent read-only review of the exact implementation candidate, followed by Grant's acceptance decision. |

The qualification plan owns the eventual independent accuracy evidence.
One successful 152389 comparison supplies regression evidence. The 1.5 arcsec
goal requires defined statistical criteria and independent reference data.

## Deferred work

New contrast rejection/tuning-warning thresholds, formal positional errors,
gain/photometric acceptance, broader beam-shape criteria, threshold
qualification, SCIENCE/RA-Dec/coadds, CFITSIO upgrade and runtime deployment
retain their separate workstreams. The current fit method, native reduction
algorithms and feedback recipe remain the reference for the proposed
selection/reporting increment.

## Next owner decision

Approve or amend the proposed implementation details and the two-part
Citlali/operational-consumer scope. The current authorization covers these
documents. Application work starts after that scope decision and exact
consumer source binding.
