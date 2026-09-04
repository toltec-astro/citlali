# SCI-FRUIT EL-F11 — Prospective Influence Persistence Design r0.1

Status: **development-study design; not scientific authority, execution
authorization, a safeguard selection, or a production interface**

## Plain-language purpose

EL-F10 tells us, after the fact, exactly why deleting UID 4460 scan 5 from the
iteration-5 a1400 JINC map makes large arcs. It does not tell us whether that
consequence was predictable before the hard deletion was applied.

EL-F11 asks one prerequisite question before comparing new safeguards:

> Did the same detector/scan already have a similar whole-map influence in
> iteration 4, when the pipeline decided to exclude it from iteration 5?

If the iteration-4 influence resembles the iteration-5 consequence, it may be
reasonable to design a response-aware decision test. If it does not, then the
present case does not support such a rule, however compelling the retrospective
EL-F10 explanation may look.

## Why this is prospective only in a bounded sense

The proposed quantity uses only iteration-4 mapmaking values and can be
formed before a newly learned penalty affects iteration 5. In that timing
sense it is prospective.

This development replay nevertheless names UID 4460 and scan 5 because their
later behavior is already known. It is therefore an oracle-targeted
feasibility test, not yet a deployable detector-selection method. A future
method would need causal machinery that decides which candidates to account
for without knowledge of their later response. EL-F11 may not erase that
distinction or claim an unbiased performance estimate.

## Exact maps and temporal boundary

Let `t` be the already registered target subset: observation 123424, array
a1400, detector UID 4460, zero-based scan 5, and only occurrences admitted to
the relevant iteration's JINC accumulation after unchanged RTC, PTC, flags,
weights, masks, projection, and support decisions.

For absolute FRUIT iteration `k`, retain the exact total and target JINC
accumulators `N_k`, `C_k`, `Q_k` and `N_t,k`, `C_t,k`, `Q_t,k`. On the
registered conditioned support define

\[
M_k=N_k/C_k,
\qquad
M_{k,-t}=(N_k-N_{t,k})/(C_k-C_{t,k}),
\]

and the direct target-deletion response

\[
D_k=M_{k,-t}-M_k.
\]

EL-F10 already measured `D_5`. EL-F11 would measure `D_4` by replaying only
the transition from the completed injected iteration-3 checkpoint to
completed iteration 4. The new diagnostic must not alter the iteration-4
science map, learning, or checkpoint. `D_4` is evaluated only after the
iteration-4 map exists, but it uses no iteration-5 result or state and is not
fed back into the algorithm.

The study compares `D_4` with `D_5`. It does not assert that the two maps
should be equal: feedback, cleaning, learned state, weights, masks, support,
processed signal, and JINC placement can all evolve between iterations.

## Exact replay and compatibility control

Use the frozen EL-F10-R4 executable:

`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.2/setup/citlali-el-f10-r4`

It is 14,858,136 bytes with SHA-256
`71911a6768b7ecfff0d165a17d498adf5c0e8e0219e733e72d633d8b545c7636`.

Start from an exact copy of the original EL-F5 off-source-injected completed
iteration-3 checkpoint:

`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f5-off-source-injection-r0.1/point-123424/off-source-injected/reduced/redu03/citlali_restart_checkpoint.nc`

It is 6,508,010 bytes with SHA-256
`a20558aaed4ddf1c34ab343770002d15882964c1fa22b616277146bb54e5c00e`.

Advance exactly once to completed absolute iteration 4. Reuse the frozen
EL-F5/EL-F10 configuration stack, including the 100 mJy/beam off-source
injection at FITS map-world `(AZOFFSET, ELOFFSET) = (0, -60)` arcsec. The
only permitted configuration changes are:

- isolated output and restart paths under
  `/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f11-prospective-influence-r0.1`;
- the exclusive iteration bound needed to stop after iteration 4; and
- enabling the existing JINC accounting diagnostic for a1400, UID 4460,
  zero-based scan 5.

Use one configured thread and `--grppiex seq`. Preserve all EL-F5 through
EL-F10 inputs and products byte-for-byte.

Before interpreting the new accounting, all nine a1100/a1400/a2000 science
planes must match the existing EL-F5 injected iteration-4 FITS products
bitwise. The complete checkpoint must match the existing iteration-4
checkpoint value-for-value after only the already registered normalization
of historically omitted default provenance and exclusion of diagnostic-only
configuration provenance. Learning output must be byte-identical.
Map-diagnostic structure and scientific values must match value-for-value
after only registered creator/provenance normalization; the historical and
EL-F10 executables are already known to produce different whole-file hashes
for otherwise equivalent map-diagnostic NetCDF. Any other difference stops
the study.

## Required closure before comparison

The EL-F10 accounting contract remains unchanged. The total `N`, `C`, and
`Q` snapshots must re-finalize the diagnostic-on iteration-4 map exactly. The
target ledger must contain each final-PTC target occurrence once and classify
its admission or rejection. Removing target `N_t`, `C_t`, and `Q_t` must
close the exact deletion identity

\[
D_4 = \frac{C_{t,4}}{C_4}
\left(M_{4,-t}-M_{t,4}\right)
\]

within the same prospectively calculated binary64 error treatment used by
EL-F10. Support changes must be reported and an unexplained change stops the
comparison.

## Descriptive persistence analysis

The analysis registration must be frozen before opening the new `D_4` values.
It must verify identical array identity, units, WCS/grid, pixel ordering, and
declared conditioned common support for `D_4` and the retained EL-F10 `D_5`.
It then retains complete `D_4`, `D_5`, and `D_5-D_4` maps and reports:

- normalized inner product (cosine similarity);
- signed Pearson and rank correlations as descriptive summaries only, with
  no independent-pixel significance claim;
- the descriptive least-squares scale
  `beta = <D_4,D_5>/<D_4,D_4>` and residual fraction
  `||D_5-beta D_4||/||D_5||`;
- sign agreement on the registered common nonzero support;
- overlap of the largest absolute responses at fixed 1%, 5%, and 10%
  footprint quantiles;
- the fraction of `D_5` squared response captured by those same high-`|D_4|`
  subsets; and
- RMS, peak, signed sum, and cross terms for the full target footprint, the
  20-arcsec fitted-Neptune aperture, the 20-arcsec injected-source aperture,
  and the existing 40--120 arcsec injection-centered annulus with Neptune
  excluded.

The signed normalization share, processed-signal contrast, absolute
coefficient-mass share, quadratic-support share, cancellation, occurrence
count, and unique-detector count must be compared separately between
iterations. This prevents a similar `D_k` from hiding different mechanisms.

No correlation, overlap, scale, or residual cutoff is a pass/fail threshold.
The complete continuous evidence returns to the owner. Spatial correlation
also means ordinary independent-sample p-values would be misleading and are
not reported.

## What this study deliberately does not do

EL-F11 does not:

- change the repeat-count rule, penalty factor, penalty placement, source,
  recurrence, feedback state, cleaning, mapmaking, or production defaults;
- define a response-aware hard threshold, soft factor, or map-local action;
- compare candidate interventions;
- claim that an increment or detector component is an independently
  calibrated sky product;
- judge UID 4460 or establish behavior for other detectors, scans,
  observations, iteration transitions, or observing modes; or
- qualify FRUIT/JINC, launch Gate D or Stage B, or authorize Unity activity.

If `D_4` shows useful persistence, the next significant decision may be a
separately bound intervention comparison. That later packet must define
candidate selection causally, freeze its thresholds before outcomes are
opened, retain scientific and performance metrics, and require an independent
pointing before any safeguard recommendation. If persistence is weak or
mechanistically unstable, response-aware policy development from this case
should stop or be redesigned.
