# Second-audit comparison: Citlali convolve statistical contract

Date: 2026-07-31

Frozen independent memo:
`/private/tmp/convolve-second-audit-precomparison.md`

Frozen memo SHA-256:
`d2ad1ab7a8978e2c93c04a9e03585dd67e80bbdfd860c62797c47d253efb7710`

Compared audit:
`800e8ae433f87d3fb7521fcb1a7fdf1d32532949:doc/CONVOLVE_SIGNAL_UNCERTAINTY_AND_RESPONSE_CONTRACT.tex`

## Verdict

The independent derivation substantively agrees with the audit's core
mathematical and implementation conclusions. Both analyses identify the same
implemented conditional operator, the same baseline and predecessor defects,
the same corrected candidate diagonal variance, the same empirical `N-1`
repair, and the same absence of a point-source response correction.

There is one important internal inconsistency in the audit document: its
simplified binary-edge equations omit the implementation's outer output mask,
although its later exact implementation equations include that mask correctly.
That transcription changes the displayed estimator outside the output window,
and it must be corrected in the audit. It is not a disagreement about the
candidate code or its intended numerical core, because the audit's exact
implementation section and the frozen second derivation both give
`D_T C_k D_T` plus the masked affine fill. Two smaller contract omissions also
need correction: zero-weight values are conditional deterministic inputs, not
modelled random samples, and the audit has two missing LaTeX backslashes in
limiting-case equations.

These are bounded documentation/product-contract amendments rather than a
reason to change the signal estimator or stop before Stage 2.

## Explicit comparison matrix

| Issue or claim | Independent result | Audit result | Disposition | Code/equation evidence | Implementation consequence |
|---|---|---|---|---|---|
| Exact signal operator | `y=D_e C_k[b1+D_e(x-b1)]`; circular convolution; signal-only median fill; final output mask `D_e`. | Exact implementation Equation `implemented-signal` agrees. Executive and Equations `fixed-edge-signal`, `edge-stored` omit the outer `D_T`. | **Partially agree** (bounded audit inconsistency) | Candidate `wiener_filter.h` edge blend/filter/post-window sequence; memo Sec. 1; audit lines 64--67, 251--263 versus 676--695. | Correct simplified equations, response, covariance, and product table to carry output mask. No numerical signal change. |
| Affine status | Affine only conditional on realized window, data-derived median, and kernel; otherwise nonlinear/piecewise affine. | Same in `implemented-signal` discussion and random-fill equations. | **Agree** | Memo Secs. 1--2; audit Equations `random-fill-cov`, `data-derived-fill-cov`, `implemented-conditional-operator`. | State conditioning explicitly in metadata/docs; do not derive analytic median covariance now. |
| Full covariance | `C_y=D_e C_k D_e C_x D_e C_k^T D_e`; convolution induces off-diagonal covariance. | Same in general and implementation sections, modulo outer-mask omission in simplified edge equations. | **Partially agree** | Memo Sec. 2; audit Equations `affine-output-cov`, `implemented-conditional-operator`. | Persisted variance is explicitly diagonal-only; no full covariance product in this repair. |
| Diagonal/taper powers | Exact variance is `e_i^2 sum k^2 e_j^2/W_j`; existing fractional path implies reciprocal taper powers. | Same; recommends continued rejection. | **Agree** | Candidate formal propagation and post-window; memo Sec. 2; audit Equations `taper-var`, dormant-taper subsection. | Keep convolve/lowpass cosine rejection; do not repair or enable fractional taper. |
| Local kernel-support normalization | Baseline's division by `C_(k^2) a` has no matching signal normalization and is wrong for fixed convolution. `fa8` removes it. | Same definite mathematical error. | **Agree** | Baseline/candidate `wiener_filter.h`; memo Sec. 3; audit Equations `baseline-weight`, `candidate1-var`. | Preserve candidate `W=1/C_(k^2)(1/W)`. |
| Positive weight below `cov_cut` | Such samples remain stochastic signal/noise inputs and must contribute `k^2/W`; `fa8` mismatches, `02a` fixes. | Same. | **Agree** | Exact eligibility diff in sequential and OMP headers; candidate low-weight test; audit `candidate-var`. | Preserve all-finite-positive eligibility independent of variance `cov_cut`. |
| Zero-weight input | Filter core can carry a nonzero zero-weight value through signal/noise while formal propagation excludes it. The conditional contract is coherent only if zero-weight inputs are fixed/deterministic (normally zeroed during map normalization), not unmodelled stochastic samples. | Audit notes zero-weight values can enter but its candidate-correctness statement leaves this precondition implicit and its simplified `T` notation can be read as excluding them. | **Partially agree** | Memo Sec. 4; `normalize_maps` plus filter signal/formal paths; audit zero/low-weight discussion and `candidate-var`. | Add explicit zero-weight conditionality and equation-derived regression; do not claim variance for an undefined random input. |
| `cov_cut` meaning | Seeds science/edge window and output/calibration policies, but is not a formal-variance censor in the candidate. | Same. | **Agree** | Edge mask and output threshold code; memo Secs. 3--4; audit baseline/candidate discussion. | Keep roles named separately. |
| Edge fill and median covariance | Core median is data-derived. Candidate conditions on it; formal/jackknife planes omit its variance and cross-covariance. | Same; calls this an unresolved approximation. | **Agree** | Signal blend and noise filtering; memo Secs. 1--2; audit `random-fill-cov`. | State conditional contract; require empirical edge validation before production. |
| Output masking and circular boundary | Signal/noise/kernel are post-masked; FFT convolution remains periodic and can couple opposite array edges. | Exact implementation section agrees; simplified edge equations suppress output mask. | **Partially agree** | FFT helper and filter post-window; memo Sec. 1; audit `implemented-signal`. | Correct docs and add output-mask/circular-wrap tests and metadata. |
| Sequential/OpenMP equivalence | Operators match source-for-source, including thread-local noise FFT; current tests select one build path rather than compare both directly. | Same. | **Agree** | `wiener_filter.h`, `wiener_filter_omp.h`; memo Sec. 1; audit source appendix. | Add an equation fixture runnable under both build configurations and compare outputs. |
| Empirical variance and small `N` | Baseline uses population central moment; `fa8` adds `N-1` but is cancellation-prone; `02a` uses Welford `M2/(N-1)`, rejects mean-subtracted `N=1`, gives exact sample variance at `N=2`; known-zero-mean singleton remains distinct. | Same. | **Agree** | `map.cpp`; candidate tests; memo Sec. 5; audit `jackknife-var` and implementation subsection. | Preserve known `N-1` correction; no further bias analysis; retain tests/metadata. |
| Empirical versus calibrated weight | Empirical variance is pixelwise sample variance; published empirical weight is a scalar calibration of a formal spatial pattern, not generally `1/V_emp`. | Same. | **Agree** | `map.cpp`; memo Sec. 5; audit Equations `implemented-median-ratio`--`implemented-emp-weight`. | Keep formal, empirical variance, and calibrated weight separately named. |
| Point-source/response semantics | Unit-sum gives DC response only. Convolved amplitude is not response-corrected point-source/template amplitude; no authoritative `R` is calculated/applied. | Same; candidate aliases are truthful compatibility products. | **Agree** | Output helpers/metadata; memo Sec. 6; audit response section/product table. | No new photometric estimator; retain aliases as `TYPE=convolved_amplitude`, without `RESPNORM`. |
| `coverage_bool` | Final-weight threshold classification; not exposure, numerical validity, edge window, convolution support, response, confidence, or calibration set. Zero threshold/NaN behavior is especially weak. | Same. | **Agree** | `fits_image_products.h`, output helper; memo Sec. 7; audit product matrix. | Correct overstated description; never use it as convolution support. |
| Minimum support guard | `C_(k^2)a > 1e-6 sum k^2` is a tiny numerical-overlap gate only; strict `>`; it does not renormalize or justify science selection. | Same. | **Agree** | Sequential/OMP formal code; memo Secs. 3, 7; audit `candidate-valid-support`. | Keep numerical validity separate from any scientific support threshold. |
| Proposed support plane | A dimensionless support plane is needed before support-dependent selection; frozen memo identifies normalized `k^2` overlap as the natural variance-leverage diagnostic but allows withholding instead. | Recommends normalized `|k|` mass for absolute operator support, with `k^2` leverage and signed response kept distinct; later lists the exact choice/floor as unresolved. | **Partially agree** (legitimate product choice, not core math) | Memo Sec. 7; audit Equations `l1-support`, `l2-support` and recommendation item 5. | Smallest safe repair may explicitly withhold filtered selection; otherwise choose and test one named support equation without overloading `coverage_bool`. |
| Fruit-loop use | Filtered amplitude/weight must remain fail closed; future filtered S/N selection requires explicit validity/support plus blank/injection calibration, while raw coadd remains feedback amplitude. | Same. | **Agree** | Memo Sec. 8; audit fruit-loop section. | No filtered amplitude/gain routing and no production selector enablement in this branch. |
| Loaded-product provenance | Current-run filter configuration does not identify the operator that produced an external or restart FITS. A retained pre-contract convolve product can therefore be unmarked. | Requires fail-closed filtered selection but does not derive an exact load-time provenance mechanism. | **Partially agree** (bounded enforcement omission) | `fruit_loop_map_io.h`, `TCProc::load_mb`, and signal-HDU metadata; retained convolve products predate `FLFBACK`. | Persist neutral `FILTEROP`; reject unit-sum, explicit withholding, and missing identity on filtered loads. Do not turn full-Wiener compatibility into an approval claim. |
| Validation gates | Unit/equation, serial/OMP, metadata, build/config/regression gates validate implementation; blanks, injections, all-array/edge/covariance and same-SHA Unity evidence are scientific/production gates. | Same substantive plan, with some tests phrased as one combined validation list. | **Agree**, with gate-phase clarification | Memo Sec. 9; audit validation and test matrix. | Run all feasible local implementation gates now; retain proportional external science gates for release/fruit-loop use. |

## Action classification

### A. Repair now

1. Preserve the candidate's fixed-convolution diagonal variance, all-positive
   low-weight treatment, stable `N-1` estimator, fractional-taper rejection,
   and truthful convolved-amplitude aliases.
2. Correct the audit's simplified binary-edge equations to include the outer
   output mask consistently in signal, affine offset, covariance, response,
   support/validity statements, and product references.
3. Correct the two LaTeX transcription errors (`sum_j` and `qquad`).
4. State that formal variance is the diagonal of the stored convolved
   amplitude covariance, conditional on the realized binary edge window,
   same-map median fill, fixed kernel, diagonal input covariance, and
   deterministic zero/non-finite-weight inputs.
5. State corresponding conditional/diagonal semantics on empirical products
   and identify calibrated weight as a scalar-calibrated formal pattern.
6. Correct `coverage_bool` metadata so it does not claim convolution support
   or complete validity.
7. Keep the `10^-6` overlap threshold explicitly numerical, not scientific.
8. Choose the safe support outcome required by the task: either add a named
   dimensionless support plane with equation/tests, or explicitly mark
   filtered products unavailable for production selection. Do not infer
   support from final weight.
9. Add the missing equation-derived tests, including zero-weight
   conditionality, output masking, exact support boundary, and evidence across
   sequential/OpenMP builds.
10. Bind filtered feedback loading to producer metadata: use neutral
    `FILTEROP`, reject unit-sum and explicitly withheld products, and reject
    missing producer identity rather than inferring it from the current run.

### B. Validate before production or fruit-loop use

1. Same-SHA all-array Unity build/reduction and product/metadata inventory.
2. Blank controls for formal/direct-empirical/calibrated variance, false S/N,
   spatial covariance, and edge-distance behavior.
3. Compact, beam-shaped, resolved, and extended injections across amplitude,
   array, coverage, and edge distance.
4. Empirical bound on omitted median-fill and realized-mask uncertainty.
5. Response and support-floor choice validated against completeness, bias, and
   false selection; correlation-aware morphology and threshold calibration.
6. No production filtered fruit-loop selector until these gates pass.

### C. Legitimate future work, not required here

1. Full covariance or compact covariance-kernel products and GLS photometry.
2. Analytic covariance propagation for the same-map sample median.
3. A named response-corrected beam/template-amplitude estimator.
4. Fractional-taper redesign, local normalization, or a spatial empirical
   correction model.
5. Renaming legacy compatibility HDUs or redesigning point-source products.
6. Production fruit-loop selector/gain/amplitude implementation.

## Decision gate

**Proceed to Stage 2.** The outer-output-mask discrepancy is a definite audit
document defect, but the audit's exact implementation derivation already has
the correct operator and agrees with the independently frozen memo. Correcting
the simplified equations does not change candidate application numerics or any
persisted product value. There is no unresolved disagreement over the signal
estimator, formal variance formula, taper powers, low-weight treatment,
response correction, or current product meanings.
