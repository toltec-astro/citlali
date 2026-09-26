# Final bounded raw-inventory and corpus review — 2026-09-26

**Disposition: PASS for the qualified inventory and future-workspace proposal.** No scope-blocking factual error found. This reviews only the observation/support inventory and corpus/budget proposal; it does not independently approve the purge target manifest or verify deletion completion.

Reviewed artifacts (exact content):

- `OBSERVATIONS.json.gz`: `d0b80e5d13163c948d3a0078f80d8a6075676097233874de0d28ffaf90902de2`
- `OBSERVATIONS.md`: `4bdbe4044915a0b178fb0bf1c16f7ae5064c3ff2bbb26326d4c1e52745d07cb9`
- `SUPPORT_AND_PROCESSED_GROUPS.json.gz`: `9fc380df77bf67eff2e8416ddd108822672b0f0e4f4bf91eadbdd344eb7d9ff0`
- `CORPUS_AND_BUDGET_PROPOSAL.json`: `36d1e9fe845e671be5c9503062fb790cfc2c9511163ef33265a1dd80384b5053`

## Verified facts

- `OBSERVATIONS.json.gz` contains188 distinct header-identified raw acquisition groupings, including27 sky-stream groups. All1,513 files containing native acquisition Data.Toltec.Is/Qs in the saved header census are represented. Additional exported raw-complex-grid products in the VNA lab are not native TolTEC acquisition files and should remain classified among protected lab products, not added to the21 complete sky packages.
- The27 sky identity/network/tune-association sets exactly match my earlier independent header aggregation. Counts are21 complete against saved reduction input inventories,5 structurally present with expectation/calibration readiness unverified, and123424 partial. No fixed13-network requirement is imposed.
- Each of the21 complete classifications has the saved configured network set, original telescope paths, and raw tune/processed tune/text-fit paths for every actual network. APT dependencies resolve to exact project-relative paths with digest identities; no complete row silently substitutes an arbitrary same-basename APT. APT variants remain explicit and are not represented as interchangeable.
- OOF152385–7 correctly share exact tune152384/sub0/scan1.123424 correctly remains partial; both original telescope acquisition and all12 raw tune acquisitions are absent, while processed tune/text fits/recomputed telescope and alternative APT candidates exist. The compact table's “and/or” wording is conservative; the owner report can state both known gaps directly.
- Processed/support-only material is separately represented as157 processed identity groups and775 support identity groups. The775 support-only groups must not be confused with775 complete raw packages. All21 complete labels remain explicitly relative to saved local reduction input inventories, with current Unity availability and full sample quality unverified.
- Every one of188 package selected-path lists has no duplicated path and sums exactly to its recorded byte total using the already captured full stat inventory. No current target NetCDF was reopened.

## Corpus and budget arithmetic

Each corpus path list exactly equals the union of selected observation input paths and counts shared paths once:

| Proposal | Unique selected paths | Exact bytes | Decimal GB |
|---|---:|---:|---:|
| POINT152389 + OOF152385–7 |132|1,830,369,481|1.830369481|
| Add SCIENCE152390 + POINT152391 |230|9,409,732,148|9.409732148|
| Add beammap148670 + POINT148669/148671 |371|26,963,413,256|26.963413256|

These are conservative exact-path unions, not byte-deduplicated minimum packages: multiple recorded APT/support variants can remain selected. The qualification makes that distinction. Raw tune152384 is shared by theOOF3 and counted once per chosen path. All chosen inputs are in the initial stat inventory; no new root, raw transfer or experiment has been selected by this proposal.

The35+25+20+40+80GB components total200GB. The proposal clearly scopes this to a possible future working root, pending the owner decision; it does not impose a quota on all current local_data or authorize historical/raw deletion. The40GB active-output and80GB scratch allocations are planning headroom, not a measured peak or proof that arbitrary concurrent reductions fit. Stating that limitation is sufficient for this inventory task. A future selected job must still have an output/scratch plan appropriate to its actual processing route.

## Qualifications to preserve in owner closeout

Keep the twenty-one count tied to “complete against saved reduction input inventories,” not “verified complete on Unity.” Preserve the five unverified POINT classifications and123424's explicit missing originals. The current evidence supports the resident-corpus proposals and measured input sizes, but does not select a new scientific campaign, approve a200GB quota, prove every historical reduction remains immediately runnable after purge, or claim all detector samples are good.

Read-only verification source: `/private/tmp/final_readiness_checks.py`, plus checks against the closed saved header census. No Unity connection, current timestream-target read, source/archive modification, reduction, or new science was performed.
