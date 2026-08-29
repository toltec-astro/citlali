# SCI-NOI v0.1 — Internal Implementation-Informed Dossier

Status: Stage A internal scope evidence; prohibited author input

Inspection authority:
`codex/scientific-contract-library@5f206cf46bb2868aadb00f37dbbbc3944ac4ec8c`

This dossier describes recovered implementation and evidence only far enough
to expose scientific questions. It does not define correct science, assess
conformity, or claim that a path is reachable in every reduction mode.

## Current Configuration And Lifecycle Inventory

The current tree exposes six typed requests under `noise_maps`:
activation, requested count, detector randomization, realization persistence,
empirical-product activation, and empirical coefficient scaling. The current
execution plan preserves requested state, resolves disabled effective state to
zero maps, records a fixed internal Boost MT19937 seed of `5489`, and derives
observation/coadd/product/write cardinalities at completion.

This is configuration and implementation authority only. It does not establish
the randomized target, adequacy of a count, covariance meaning, or production
default.

## Generation Routes Observed In The Current Tree

| Surface | Observed implementation fact | Scientific question exposed |
| --- | --- | --- |
| Ordinary Lali and Pointing pipeline entry | One mutable MT19937 engine is created per pipeline invocation; signs are drawn sequentially before the parallel farm consumes each scan | Assignment identity can depend on scan/order unless a stronger key policy is used; fixed seed is not a complete ensemble identity |
| `randomize_dets=true` | One sign per realization, scan/chunk, and realized detector column | Detector-within-scan coherence law; detector identity and scan partition must be explicit |
| `randomize_dets=false` | One sign per realization and scan/chunk shared by all detector columns | Scan-coherent law, not “no detector randomization”; it preserves/destroys different modes |
| Naive and legacy JINC population | Signs multiply the processed weighted signal immediately before gridding into observation and/or coadd noise buffers | Randomization is after the realized RTC/PTC scan state; exact MAP/JINC operator parity and authorized JINC route remain scientific boundaries |
| Beammap internal iteration | A generator lives across the loop and sign matrices are repopulated inside mapmaking-stage/reset logic | Assignment can depend on iteration/pass/active-map history; historical repair proposed named-pass stable assignments but is not current authority |
| Observation/coadd surfaces | Noise tensors exist for observation and coadd buffers; current source includes direct coadd projection and typed observation-to-coadd accumulation checks | Direct combined-data, slot-matched observation coadd, and resampled observation ensembles must be distinct NOI methods |
| FRUIT noise-only path | A prior map can be subtracted from TOD; detector weights are recalculated/reset and noise maps populated before source add-back | Fixed residual state, partial relearning, and complete adaptive replay are different methods; FRUIT owns the recurrence |

No separately named subscan randomization method was found in the current
requested configuration. Scan/chunk boundaries and PTC segments may not be
identical, so Stage B must not treat “scan,” “subscan,” and “chunk” as aliases.

## Fixed And Relearned State Census

| Recovered route | RTC/PTC | AST | MAP/JINC/coadd | Filter/source/consumer state |
| --- | --- | --- | --- | --- |
| Ordinary current scan-sign path | Realized RTC/PTC scan data and learned coefficients are already present before signs are applied | Coordinates are reused | Existing mapmaker state is reused; observation and possible coadd destinations are populated | No downstream filter or consumer relearning at generation |
| Fixed observation-map coadd derivation | Observation realization maps are fixed inputs | Observation WCS/embedding is fixed | Membership, embedding, coefficients, normalization, and support are fixed | Any later fixed filter can be applied identically |
| Data-derived/recomputed coadd | Upstream realization members may be fixed | Coordinates may remain fixed | Membership or coefficients are recomputed from realizations | Distinct joint method; not covered by fixed propagation |
| Beammap current iterative route | Source-aware RTC and PTC cleaning can rerun across named iterations before mapmaking; signs are regenerated inside the iterative mapmaking surface | No separate AST relearning was identified in the inspected slice | Map/fit state and active-map population change across iterations | Beammap owns iteration and fit interpretation |
| FRUIT current noise-only route | A source model is subtracted; detector weights are recalculated/reset before noise accumulation | Coordinates appear reused in the inspected slice | Mapmaker route is reused for a noise-only pass | Source add-back and recurrence remain FRUIT-owned; this is neither a pure ordinary fixed-state method nor a complete relearned replay by name alone |
| Conceptual full relearned method | Exact rerun set is not approved | Must state fixed versus rerun explicitly | Must state fixed versus rerun explicitly | Must include exact filter/source/selection/stop reruns and joint law |

This census is deliberately descriptive. Exact operator reachability and
conformity require a later source-closed assessment against a selected
application revision.

## Current Inference And Product Arithmetic

The current `MapBuffer::calc_noise_products` surface:

- computes the realization mean and the centered second moment with divisor
  completed `n_noise`;
- clips small negative numerical results to zero;
- computes a valid region from the stored map coefficient and `cov_cut`;
- computes a median of coefficient-times-scatter and its reciprocal scale;
- forms an “empirical” coefficient plane by scaling the original coefficient;
- can replace the primary map coefficient with that plane;
- forms pixel standardized signal as signal times square root of that plane;
- labels a filtered square root of realization variance as a point-source
  uncertainty and divides filtered signal by it; and
- calculates pooled standardized-realization diagnostics.

These are implementation facts, not endorsed meanings. They expose the exact
collisions the new contract must prevent:

- divisor `R` versus `R-1` or joint-design correction;
- conditional stack scatter versus repeated physical-noise variance;
- scalar coefficient-scale diagnostic versus empirical inverse variance;
- NOI empirical weight versus MAP-facing nonprecision coefficient;
- filtered pixel scatter versus a response-qualified source estimator;
- invalid denominator versus a numeric zero sentinel; and
- standardized signal versus significance.

## Filtering Inventory

The filter runner applies a signal filter and then loops over every available
noise realization. It can recalculate products after filtering. Wiener mode
requires noise maps in current config; low-pass-only mode does not.

The accepted historical NOI-002 application record explicitly preserved a
conditioned FLT gap: signal filtering used signal/background affine edge
handling and a response path while realization filtering was zero-centered.
Strict operator/edge parity was not established. That record is evidence only,
but the scientific question is genuine: a “same filter” claim requires exact
operator, edge, support, response, and data-dependence identity.

## Mode And Consumer Inventory

| Consumer | Recovered dependency | Boundary consequence |
| --- | --- | --- |
| Deterministic convolution/low-pass | Can propagate a fixed realization ensemble through the same operator | SCI-FLT owns transfer, edge/support, and response; NOI owns ensemble/inference identity |
| Wiener filtering | Operator may depend on an estimated noise model | Held-fixed and re-estimated filters are separate methods; input ensemble cannot self-validate the inference-bearing operator |
| Beammap | Iterative learned map/fit state and optional noise maps | Beammap owns PSF/calibration/sensitivity interpretation; its ratios are not generic NOI products |
| Pointing and OOF | Quicklook/fit modes historically use `sig2noise`-like quantities and can disable ordinary noise maps | Fit/dynamic-range identities remain mode-owned; no significance promotion |
| FRUIT | Source subtraction/add-back, weight recalculation, recurrence, and selection | Residual randomization and full-procedure relearning must be separate; SCI-FRUIT owns adaptive law and stopping |
| MAP | Has nonprecision coefficients and immutable base/coadd products | Empirical NOI weights attach; they do not overwrite or become MAP coefficients under this contract |
| JINC | Frozen signed estimator currently lacks an authorized ordinary numerical route and response/covariance companions | NOI may attach only to a future exact realized JINC parent and cannot create missing upstream authority |

## Persistence And Provenance Inventory

Current configuration can write realization images or compute products without
writing them. Current arithmetic retains full realization cubes in memory,
while the internal scientific plan describes two-moment and projected-
estimator streaming alternatives.

The historical NOI-001 repair implemented a compact versioned key with exact
observation, iteration/pass, realization, coherence-unit, and channel identity,
plus reconstructible digests. That branch is not current ancestry, but it
demonstrates that exact regeneration need not require storing every sign.

The scientific contract must distinguish:

- persisted realization ensemble;
- transient but exactly regenerable ensemble;
- streaming sufficient-statistic reduction with no realization reconstruction;
- partial or failed persistence; and
- unavailable reconstruction.

Those states have different audit and later-analysis capabilities but do not
by themselves change the estimator target.

## Evidence Inventory And Quarantine

Historical repair/re-audit and application-integration records report focused
fixtures, CTests, product-contract tests, preflight, and accepted reductions.
Current validation registries also contain disabled, generation-only, and
full-output cases with exact cardinalities and close numerical comparison to
historical behavior.

All such results are evidence about exact candidates. They cannot decide:

- whether the ensemble estimates physical noise;
- whether fixed-state or relearned generation is ordinary;
- which finite correction or covariance representation is correct;
- whether a standardized product has a Gaussian or detection-probability
  interpretation;
- whether a realization count is adequate; or
- whether current or historical source conforms to the future SCI-NOI
  contract.

## Current-Tree / Historical-Candidate Divergence

Neither the accepted NOI-002 repair tip
`5b29e13548a6fec884c67b192dec20c92f0bbb62` nor the NOI-001 deterministic-key
repair `38ef72860743636f59d226c9e1ff5ff776d0e9c0` is an ancestor of the
starting scientific-contract-library commit. The current tree has materially
different noise provenance and product code. Therefore no historical
conformance or validation status is carried into this package.

## Dossier Conclusion

The current implementation supports the need for the proposed taxonomy but
does not select it. Stage B must remain implementation-blind. Later conformity
work will need an exact target application revision and separate authorization.
