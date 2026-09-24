# Codex directive: POINT 152389 — complete the map-only JINC comparison

**Owner:** Grant Wilson  
**Date:** September 24, 2026  
**Status:** EXECUTION WORK ORDER — implement, validate, run, assess, review, and stop.  
**Scope:** POINT 152389; a1100, a1400, and a2000; ordinary MAP versus the single JINC realization specified below, on identical saved processed samples and original adaptive ordinary-MAP FRUIT histories.

## 0. Execution authority and resolution of the previous stop

This is the consolidated continuation of `CODEX_POINT_152389_MAP_ONLY_JINC_COMPARISON.md`. It supplies the missing numerical selections and the numerical-adequacy profile while retaining the original experiment's scientific boundaries. It supersedes that directive's missing-authorization stop and any current decision record that still marks these selections pending. It does not retroactively change sealed preparation records or prior experimental products.

**For all three arrays, including a2000, the selected values are `rmax = 3` and `nsub = 16`. This directive explicitly supplies both previously blank a2000 values; no inference from the earlier partial table is needed.**

The signal-error criterion, reference peaks, cancellation treatment, certificate scope, and execution sequence are specified here. Certificate implementation and generation are engineering work to complete under this order, not another missing owner decision. Do not return another preparation-only report merely because that implementation remains to be written.

Proceed through the complete prescribed experiment without another approval cycle for routine implementation, validation, reporting, or scheduling choices. A genuine missing/corrupted scientific parent, incompatible upstream contract, or demonstrated inability to establish the required bound remains a real blocker; report it precisely, preserve evidence, and continue independent unblocked work. Never fabricate a certificate, relax the tolerance, or declare an unexecuted comparison complete.

> Change the final sample-to-map operator, not the samples or the histories that produced them. This is not a new PTC reduction, a JINC-driven FRUIT experiment, or a kernel-optimization campaign.

## 1. Scientific objective and fixed reference

Determine whether this JINC realization provides a useful alternative representation and measurement of the same processed data. Compare source response and shape, pixel-phase dependence, signed structure, background correlations, measurement-scale additive uncertainty, usable support, and computational cost.

The reference remains the original adaptive ordinary-MAP **threshold-3, unity-gain, complete-replacement** FRUIT history and its C+D branches. Use the existing compact positive/negative and shoulder perturbations as regression benchmarks. Do not tune JINC to recover the shoulder.

Further shoulder decomposition, reconstruction, and reference-basis-sequence counterfactual work remain parked. A completed REFSEQ result is not the baseline. Neither a surviving shoulder deficit nor a qualified a2000 background model is permission to change the experiment.

This order authorizes final-map accumulation, the necessary numerical certification, the narrow saved rejoined-parent adapter, and the diagnostic calculations below. It authorizes no production default, new feedback candidate, or new normative response/covariance product.

## 2. Recover and preserve the existing work

Read applicable `AGENTS.md`, the current authorized worktree state, governance/review instructions, and the accepted MAP+/PTC/AST/JINC/FRUIT boundaries. Reuse the reviewed JINC primitive, saved-input recovery, native sample/AST access, fixed-state Apply utilities, and existing measurement/noise/reporting code. Do not create a parallel reduction framework.

Recover and authenticate:

- The original map-only JINC directive and its completed preparation handoff, `DECISIONS_NEEDED.md`, `owner-numerical-binding.json`, recovery records, reviews, manifests, and source identities.
- The C+D directive, completed response/robustness handoff, exact injection operands, branch states, measurements, reports, and manifests.
- The A+B post-FRUIT noise/uncertainty directive, completed companion, and its measurement, covariance-generation, and complete-ensemble failure rules.
- The original ten-cycle reference and its bound M0/T10 products. Consult frozen-PTC reconstruction evidence only as needed to recover saved states correctly; do not rerun that investigation.

Discovery hints under `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/` include:

```text
successor-point-152389-map-only-jinc-20260924/
successor-point-152389-response-robustness-cd-20260923/
successor-point-152389-post-fruit-ab-20260923/
successor-point-152389-fruit-ten-cycles-20260923/
successor-point-152389-frozen-ptc-shoulder-20260924/
```

Paths are discovery hints, not identities. Verify consumed inputs against archived expectations before execution and again at completion. Reuse the reviewed historical-closeout authentication repair; do not replace independently recorded expectations with hashes calculated from whatever now occupies an old path. Preserve all old bundles, unrelated worktree changes, and historical qualifications.

The previous preparation's successful replay and fixtures are reusable evidence, not proof that the new selected observation configuration is certified. Bind the actual current source, native binary, environment, and compatibility; do not claim that old review automatically covers changed executable code.

## 3. Exact selected JINC realization

### 3.1 Scientific and coefficient authority

Use SCI-JINC v0.1/r0.3, frozen at:

```text
a9f43877e01a661db13bd85b2e7f34ea5ac82fb7
```

Retain the independently admitted uniform-coefficient permission:

```text
family:   SCI-PTC:uniform_constant@draft-0.1
registry: SCI-PTC:analysis_gridding_coefficients@draft-0.1
profile:  SCI-PTC:uniform_coefficient_handoff@draft-0.1
boundary: SCI-PTC/JINC v0.2-draft.1/r0.4; UFC-OD-009
```

The experiment explicitly selects the same already-realized dimensionless uniform coefficients, gamma = 1, as the baseline. Authenticate their exact payload and occurrence association. This is not an inverse-variance family, a new default, or permission to substitute a different weighting scheme. Uniform-family JINC permission is resolved; do not request it again.

### 3.2 Owner-selected parameter tuples

These are **selected experimental values**, not merely historical candidates:

| Array | Exact angular scale s, radians | a | b | c | rmax | nsub |
|---|---|---:|---:|---:|---:|---:|
| a1100 | (1.1/1000)/45 | 1.1 | 1.67 | 2 | **3** | **16** |
| a1400 | (1.4/1000)/45 | 1.1 | 2.17 | 2 | **3** | **16** |
| a2000 | (2.0/1000)/45 | 1.1 | 3.17 | 2 | **3** | **16** |

For the exact-real reference, the displayed finite decimal parameter values and scale formulas denote their exact mathematical values. Record the production representation separately and include parameter-conversion effects in the kernel error enclosure. Rounded arcsecond displays are not numerical inputs.

The scale semantics are **historical angular kernel scales defined by the stated formulas**. They are not the nominal calibration-beam FWHM values and imply no physical illuminated-aperture interpretation. Preserve the existing nominal 5/7.5/10-arcsecond beam convention and calibration units independently.

There is no mode-dependent selection in this experiment: all requested branches, snapshots, references, and fixtures use the same tuple for their array. Do not sweep or adapt s, a, b, c, rmax, nsub, pixel size, or support after viewing results.

### 3.3 Analytic and discrete operator

With J(x) = 2 J1(x)/x and the removable limit J(0) = 1, use:

\[
\kappa_\alpha(r')=
J\!\left(\frac{2\pi r'}{a_\alpha}\right)
\exp\!\left[-\left(\frac{2r'}{b_\alpha}\right)^{c_\alpha}\right]
J\!\left(j_{1,1}\frac{r'}{r_{\max,\alpha}}\right),
\qquad r'=r/s_\alpha,
\]

where j_(1,1) is the exact first positive J1 root, not a rounded decimal definition. Preserve analytic zeros and every finite signed lobe of the complete product.

Use the original full target WCS, dimensions, and 2-arcsecond grid. Preserve its exact frame, projection, reference point, axis order/signs, rotation/handedness, and compatible affine metric. Do not substitute a source-disk destination, alter WCS to force compatibility, or reconstruct continuous AST positions from containing-pixel centers.

Retain the accepted rules:

- Center: componentwise floor(u + 1/2), with the prescribed positive-axis half-pixel tie.
- Phase: the accepted upper-bin midpoint quantization, independently on each axis, with 16 bins per axis. Exact zero phase selects +1/32 pixel on each axis; do not add a zero-phase special case.
- Square half-width: h = ceil(s rmax / Delta), giving h = 8/10/14 and 17x17 / 21x21 / 29x29 footprints on the current grid.
- An occurrence whose rounded center lies outside the finite destination contributes nowhere. Otherwise crop only out-of-map square members.
- No radial crop, corner suppression, wrapping, reflection, edge completion, or per-sample footprint renormalization.

For admitted membership I_ip and positive upstream coefficients omega_i, accumulate at array level:

\[
N_p=\sum_i I_{ip}\omega_i\kappa_{ip}t_i,\qquad
C_p=\sum_i I_{ip}\omega_i\kappa_{ip},\qquad
Q_p=\sum_i I_{ip}\omega_i\kappa_{ip}^{2},
\]

\[
m_p=N_p/C_p,\qquad
T^{(\kappa^2)}_p=\sum_i I_{ip}\kappa_{ip}^{2}/f_{s,i}.
\]

Do not replace C by |C| or Q, square omega in Q, include omega in T, or average already-normalized network maps. A finite negative C is eligible; exact C = 0 is unavailable, not zero sky. Q and T are not empirical inverse variance or physical exposure.

Preserve the contract's five base numerical roles and compact generative replay record. Keep certificate/response/noise diagnostics outside the normative base-role set.

## 4. Exact numerical-adequacy profile

Create the narrowly scoped, versioned profile:

```text
POINT-152389-JINC-NUMERICAL-ADEQUACY-v1
```

This section supplies its scientific requirements. Implementing its conservative bounds, records, and evaluation is authorized engineering work.

### 4.1 Domain and exact reference

The profile covers every selected JINC primary-map, injection-reference, phase-fixture, and gridding-control signal value used in this experiment, on its actual admitted occurrences and full declared destination. Reuse certified kernel state where the bindings match, but certify each distinct accumulated signal realization.

The reference is the exact-real-arithmetic **selected discrete operator**, with the parameters and rules in section 3. Authenticated sample values, realized upstream coefficients, processed sample frequencies, and the bound continuous AST coordinate payload are fixed inputs with their recorded finite numerical values. The profile covers JINC's handling of those inputs; it does not certify the physical astrometry, optical PSF, upstream calibration/cleaning, or arithmetic that originally created the saved histories.

Regenerate/bind native continuous AST coordinates and audit their same-processed-sample identities under the existing upstream contract. Do not invent an observation-wide high-precision AST transformation problem. Wrong sample-coordinate joins, centers, phase bins, WCS usage, or membership are operator failures, not small errors excused by the signal tolerance.

Intentional midpoint quantization and the selected finite square kernel define the target. Their difference from a continuous ideal is assessed by the scientific fixtures, not charged to finite-precision error. Numerical errors in realizing that target are covered.

### 4.2 Frozen reference amplitudes and exact acceptance inequality

Use the absolute original REF T10 fitted Gaussian amplitude **above the fitted constant**, from the authenticated saved original measurement field:

```text
measurement.fit.parameters[0]
```

The required values are:

| Array | P_a, mJy/nominal beam |
|---|---:|
| a1100 | 937.4216858608636 |
| a1400 | 664.7340776999814 |
| a2000 | 813.6176996671275 |

Bind the exact archived binary64 values/hex encodings and authenticated source hashes already recorded in `owner-numerical-binding.json`. Verify agreement with the original ten-cycle and C+D references. The displayed decimals identify those saved values; do not refit to obtain P_a. A genuine mismatch is an input-authentication failure, not authority to select a different peak.

For every numerically accepted JINC signal pixel, require:

\[
\boxed{
|\widehat m_p-m_p^\star|
\le 10^{-6}P_a+10^{-6}|\widehat m_p|
}
\]

Both tolerance constants are the exact decimal 0.000001. All signal terms are in mJy/nominal beam. Use the same frozen P_a for all snapshots, branches, injection references, and gridding fixtures in an array. Never use a new JINC peak, central-pixel value, amplitude-plus-background, measured map RMS, or best-looking map to relax this criterion.

This is a deterministic, componentwise requirement for **every accepted pixel**, not an RMS-over-map, average-case, or probabilistic criterion. It establishes numerical fidelity only, not a six-digit astrophysical flux measurement or the final 5% science goal.

### 4.3 Signed-denominator propagation and local support

Establish conservative absolute bounds E_N,p and E_C,p such that:

\[
|\widehat N_p-N_p^\star|\le E_{N,p},\qquad
|\widehat C_p-C_p^\star|\le E_{C,p}.
\]

Let E_R,p conservatively enclose the exact division/output residual:

\[
|\widehat m_p\widehat C_p-\widehat N_p|\le E_{R,p}.
\]

The hatted quantities here are the actual computed representations used by the output and diagnostics. Include output conversion when it changes the value. Use binary64 for the new numerical map planes to avoid unnecessary serialization loss.

Require strict denominator separation:

\[
|\widehat C_p|>E_{C,p}.
\]

A sufficient full signal-error bound is:

\[
B_{m,p}=
\frac{E_{N,p}+|\widehat m_p|E_{C,p}+E_{R,p}}
{|\widehat C_p|-E_{C,p}}.
\]

Require B_m,p <= 10^-6 P_a + 10^-6 |m_hat,p|. A directly certified interval quotient or a demonstrably tighter conservative bound is an allowed implementation of the same requirement; it does not change the profile.

Make the bound arithmetic conservative too, including the denominator margin and tolerance comparison. Do not round an error bound downward or the allowed threshold upward and call the resulting comparison rigorous. State the floating-point assumptions and treatment of underflow, overflow, subnormals, fused operations, and reduction ordering.

Apply the original algebraic requirements as well: valid finite inputs, exact-discrete Q > 0, and C != 0. Retain negative denominators when all requirements pass. Do not introduce a fixed arbitrary cancellation-ratio cutoff, replace the historical ratio rule with another unexplained constant, clip the numerator, or fill unavailable pixels.

A pixel without an established full bound is numerically unavailable, even when its machine result is finite. Local failures do not automatically fail a whole map, but they must remain visible and may make a dependent measurement unavailable. Do not silently shrink the required source/template/aperture domain to conceal failures. Report native/common support and all exclusions under the existing missing-support rules.

### 4.4 Ancillary accumulators and certificate coverage

For exact positive Q and T^(kappa^2), additionally require conservative relative error <= 10^-6 with respect to the exact discrete value. Use a certified positive lower reference bound to test this conservatively when needed. Handle exact-zero cases explicitly; Q = 0 cannot support a normalized signal. Record actual ancillary output rounding. These arithmetic checks create no new statistical interpretation.

Cover parameter/root/constant representation, Bessel and exponential evaluation, kernel radius/evaluation arithmetic, selected coordinate/phase/geometry implementation, contribution products, within-worker sums, worker/chunk/network merges, normalization, and final numerical output. Certificate checking does not replace exact occurrence and membership audits.

Nonlinear source fits and map-domain noise realizations are validated by their existing numerical tests and rules, not by claiming that this gridding certificate proves their inference accuracy.

### 4.5 Certificate implementation: bounded, not a second reduction

Prefer this approach:

1. Independently enclose the exact coefficients of the finite selected phase/kernel tables. Compare the actual production cache values against those enclosures to obtain valid coefficient-error bounds. Reuse demonstrably identical/symmetric entries and cache the evidence.
2. Propagate those coefficient errors and arithmetic errors through the **actual** observation sums. Use the actual reduction algorithm and counts/tree, rather than a guessed operation count or a generic unexplained safety factor.
3. Combine numerator, denominator, and division bounds per pixel; test the profile before releasing values to the scientific comparisons.

Use existing suitable interval/ball/bounded arithmetic when available. A small validation-only reference helper using existing or standard-library facilities, with explicit rounding and analytic remainder bounds for this finite kernel family, is authorized when needed. This is not permission to build a general special-function or validation platform. Refine reference precision or accumulation engineering as necessary to establish honest bounds without altering the kernel, scientific inputs, or tolerance.

Do not reevaluate millions of sample-to-pixel contributions in arbitrary precision as the default strategy. The finite cache and bounded sums should carry the proof. Do not mistake two matching floating-point implementations, agreement at two arbitrary precisions without a remainder/error argument, or successful fixture comparisons for a complete exact-real bound.

Record compiler target, relevant floating-point flags, and actual significand widths (`std::numeric_limits<double>::digits`, `std::numeric_limits<long double>::digits`, or equivalent DBL_MANT_DIG/LDBL_MANT_DIG). Do not claim that a `long double` oracle has more precision than `double` without verifying this on the executing architecture. An independently written same-precision direct sum remains a regression check, not an automatically higher-precision certificate.

Each certificate must bind the profile/version/digest, selected parameter values and representations, exact input identities, WCS, coefficient realization, source/binary, kernel-cache realization, reduction policy, output identity, error method, domain, pass/fail/support results, and lifecycle. Publish compact summaries plus sufficient immutable replay information; no dense per-contribution provenance is required. Per-pixel bound/support diagnostics may be retained as explicitly diagnostic companions outside the five normative roles.

### 4.6 Sequencing: preflight versus realized-map certificate

There is no requirement to certify an uncomputed observation result.

**Before observation accumulation**, finish the profile implementation, bounded kernel reference/cache certification, independent fixtures, source review, ordinary-input replay, and prospective science manifest.

**During authorized observation accumulation**, compute provisional arrays and the associated bounds. This provisional accumulation is explicitly authorized work required to produce realized-map certificates. Do not use an uncertified value in response plots, source fits, noise learning, or a claimed science output.

**After a map's checks**, release its accepted values and explicit unavailable support to dependent diagnostics. Retain failed provisional artifacts and causes separately. Once prerequisites pass, continue automatically; do not stop to ask for permission already granted here.

## 5. Identical saved inputs and T10 rejoining

Use the original adaptive C+D branches, with their own original PTC states and feedback histories. Prefer authenticated saved processed/rejoined samples. When a requested sample snapshot was not stored, reconstruct only that snapshot from its immutable parent, applied candidate, saved numerical means/basis, and native fixed-state Apply semantics.

Schematically, at T10:

\[
v_b=\mathcal P m_{b,9},\qquad
z_b=\operatorname{Apply}_{\Theta_b}(y_b-v_b),\qquad
t_b=z_b+v_b.
\]

Here script-P is the saved map-to-sample projection, not the reference amplitude P_a. Use the native ordering, mean treatment, masks, and support rather than substituting this shorthand for the actual implementation. Recomputing native application coefficients on the identical saved residual is allowed when Apply requires it; new basis learning is not.

At M0 there is no feedback operand. At T10 restore only the actual historically applied **m9**, never unapplied m10. Freeze the saved numerical means; do not import the fresh-mean intervention from REFSEQ. Preserve full native segment support during reconstruction, including entries that were valid for PTC but excluded from map contribution.

The diagnostic-only rejoined-descendant JINC adapter is explicitly authorized. Preserve its exact parents, quantity, units, removed-mean/null-space state, support, and feedback ancestry. Do not relabel it ordinary PTC output or loosen general production admission.

Both mapmakers receive identical t_b values, occurrence identities, producer validity, realized coefficients, and continuous AST coordinates. Their spatial deposition and local numerical support may legitimately differ. Audit and report those differences rather than forcing identical map support.

At fixed contribution/operator state, verify the rejoin identity on the common certified domain:

\[
A_G(z_b+v_b)=A_Gz_b+A_Gv_b,\qquad G\in\{\mathrm{MAP},\mathrm{JINC}\}.
\]

Use the appropriate propagated numerical bounds for this arithmetic check. Do not add the old raster m9 directly to a JINC residual map. Do not redraw, rescale, rephase, reconvolve, or restore an injection twice.

## 6. Fixed execution matrix and gates

For each of the three arrays, use exactly these five saved states:

| Saved branch | Snapshot | Role |
|---|---|---|
| Original adaptive C+D REF | M0 | First-pass comparison |
| Same REF | T10 | Primary processed endpoint |
| C-core-plus | T10 | Saved +10% compact benchmark |
| C-core-minus | T10 | Saved -10% compact benchmark |
| C-shoulder | T10 | Saved broader/asymmetric benchmark |

The science work consists of:

| Product/work | Required extent |
|---|---|
| Primary maps | 15 saved inputs x 2 mapmakers = **30 maps** |
| Injection-only references | 3 saved perturbations x 3 arrays x 2 mapmakers = **18 references**, using REF T10 operator state |
| Continuous-profile phase fixture | 9 offsets x 3 arrays x 2 mapmakers = **54 maps** |
| Background characterization | REF M0 and T10, both mapmakers, all 3 arrays |
| Source-plus-noise ensembles | **256 trials per supported REF T10 array/mapmaker**, at most **1,536** |
| Additional numerical work | Bounded oracle/cache/certificate, replay, rejoin, covariance-generation, and regression checks only |

Exactly reusable mathematical references may be reused only with demonstrated identity; report logical product coverage separately from the actual count of numerical accumulations. Such reuse does not authorize additional science states.

Execute REF replay/controls before their dependent perturbation comparisons. Prioritize REF M0/T10, compact endpoints, then shoulder endpoints. Use available parallelism among independent work; dependency ordering does not require serial execution of unrelated arrays.

For every requested state, ordinary MAP must reproduce its archived signal, normalization, hits/support rows, and measurements to the existing accepted tolerances. Recover those exact tolerances; the new JINC bound neither relaxes nor replaces ordinary replay tolerances. Preserve the original ten-cycle-to-C+D REF replay binding.

A failed ordinary replay blocks dependent comparisons for that state. Do not fix it by changing means, weights, candidates, PTC state, or tolerance. Repaired local implementation defects require focused tests and rerun of affected invalidated work, with failed attempts and changes preserved; no alternate scientific settings may be tried.

Use real structural or instrumented evidence for zero new observation Learn/ALS calls, zero candidate-generation calls, and zero new feedback applications. Loading or projecting an already applied historical operand for replay is not a new FRUIT cycle. Do not populate hard-coded zero counters and present them as instrumentation.

Required selected-operator fixtures include signed lobes/analytic zeros, constant-field preservation, exact/near center and phase ties, even-lattice center behavior, square corners, edges, outside-center admission, missing samples, coefficient changes, rotated/reflected compatible WCS, cancellation, and serial/parallel behavior. Include certificate boundary cases: separated positive and negative denominators, intervals containing zero, a large normalized error despite nonzero C, output rounding, and intentional error injections that must fail.

## 7. Response comparison and unchanged measurements

For each saved perturbation I and mapmaker G, compute:

\[
\Delta_G^I=A_{G,I}t_I-A_{G,R}t_R,
\qquad H_G^I=A_{G,R}u_I,
\qquad u_I=\mathcal P I.
\]

u_I is the exact original archived projected injection operand. H uses **REF T10** occurrences, validity, coefficients, normalization, coordinates, and mapmaker state. It bypasses PTC and contains no new cleaning or feedback. Do not convolve an old ordinary-map reference to approximate H_JINC, add an optical beam, or recalibrate it.

Record legitimate branch-versus-REF operator/support differences. Keep each branch's original adaptive history; do not substitute REF basis states for perturbed branches.

On explicitly declared supported physical domains, compute:

\[
g_G=\frac{\langle H_G,\Delta_G\rangle}{\langle H_G,H_G\rangle},\qquad
\eta_G=\frac{\|\Delta_G-H_G\|}{\|H_G\|},\qquad
\rho_G=\frac{\|\Delta_G-g_GH_G\|}{\|H_G\|}.
\]

Check eta^2 = (g-1)^2 + rho^2. The symbol rho here is the **orthogonal response distortion**, not the separate denominator-cancellation statistic; label both unambiguously.

Recover the original benchmark domain and retain its metric wherever fully supported. Also report each operator's native domain and the explicit intersection needed for comparable paired results. Show lost/gained pixels and causes. When the original domain is not supported, mark its metric unavailable and label the common-domain result separately; do not silently replace the headline domain. Negligible reference norms make ratios unavailable.

Show H, Delta, Delta-H, signed core/wing/bowl/background structure, and the even map M_plus + M_minus - 2 M_REF at matched angular scales. Carry numerical map bounds through these arithmetic differences where relevant; a residual at numerical-error scale is not a scientific improvement.

Reuse the exact accepted seven-parameter Gaussian-plus-constant descriptive fitter: original model, parameter meanings, initialization/search, weights, constraints, solver, 40-arcsecond-radius physical footprint, and missing-support/failure rules. Run it on the new full branch maps using that same footprint. Preserve physical centroids, gauge-safe ellipse tensors/shape summaries, and all constraint/failure flags, including near-circular angle degeneracies.

For each perturbation and measurement f, compare its actual branch increment with the ideal map-addition control:

\[
f(M_{G,R}+H_G^I)-f(M_{G,R}).
\]

These full-map control fits are explicitly authorized. Do not fit signed difference maps with the positive-source constraints. Reuse exactly matching archived controls where authenticated, otherwise compute the required control once; do not run unrelated historical control fits.

Keep the original frozen-template weights and fixed-aperture centers/radii/pixel-area factors/background rules. Do not move apertures to new fitted centroids, redefine them from JINC widths, change fit weights to inverse covariance, force equal peaks, or derive a new beam area. Preserve peaks in mJy/nominal beam and aperture quantities in their original units. Show raw brightness-area integrals and background-subtraction effects separately; apply no flux correction.

Delta is a regridded difference between completed adaptive ordinary-MAP histories. It is not a pure fixed-cleaner response, a JINC-feedback response, or whole-instrument recovery. Smoothing a residual into a smaller norm is not proof of recovered information.

## 8. Nine-phase continuous-profile gridding fixture

At the REF T10 continuous sample coordinates, evaluate one noiseless compact profile using the saved compact benchmark's shape/normalization. This is a separate **continuous-profile gridding fixture**, not a replacement for the original raster-projected C injections.

The exact nine source-center offsets are the Cartesian product:

```text
(-1/3, 0, +1/3) pixel along WCS axis 1
(-1/3, 0, +1/3) pixel along WCS axis 2
```

Express displacements through the actual WCS into physical coordinates. Keep the target grid, sample geometry/validity/coefficients, input normalization, and selected kernel fixed. Evaluate the profile directly at continuous sample positions without a raster intermediate or another beam convolution. Feed identical realized sampled values to both operators.

Use the declared full fixture destination/support; do not hide truncation by renormalization. Measure centroid error, fitted core widths/shape and peak, original fixed-aperture response, and signed sidelobes across phases. Report unavailable fits/support rather than changing constraints or footprints.

The source-center offsets move only this synthetic profile. They change no observation parent, AST association, phase-bin policy, source measurement rule, or feedback state. Do not increase nsub or change rmax after examining these results.

## 9. Background covariance and comparable measurement uncertainty

### 9.1 Domains, masks, and retained limitations

Characterize REF M0 and REF T10 for both mapmakers, using the existing A+B companion and model family. Do not expand to other snapshots, D branches, new observations, or a new covariance framework.

Freeze the masks before inspecting new inferred noise amplitudes. For an explicit conservative prescription, recover the saved A+B base source/bowl exclusion and its saved larger exclusion, retaining their recorded centers and physical definitions. Extend each **once** by a square dilation of h+1 original-grid pixels per axis for that array: 9/11/15 pixels for a1100/a1400/a2000. This covers the selected JINC deposition footprint with a one-pixel conservative phase/discretization margin. Use the resulting base and larger exclusions unchanged for both mapmakers and both snapshots. Archive their exact realized masks and geometry.

Do not replace a source/bowl envelope with the compact feedback mask. Keep measurement pixels distinct from covariance-training exclusions. Report native and fixed common background support; never fill unavailable pixels or shrink masks to obtain a favorable estimate. Inadequate remaining background or source-depth coverage is a recorded limitation.

Carry forward the a1100 network-3/scan-5 condition and the a2000 background-model qualification without detector or noise-model repairs. A qualified but mathematically usable a2000 covariance may still support a clearly labeled model-based calculation. Qualification alone is not permission to skip the array or demand a new scientific program. Invalid covariance or truly unavailable measurement support cannot be concealed.

### 9.2 Background learning and analytic propagation

Report offsets/means/medians, centered dispersion, second moments, tails, signed directional covariance, pair counts, angular scales, regional variation, and available sampling/network-mixture diagnostics. Distinguish background residual structure from proven stochastic noise. Record modes removed by centering. Do not subtract gradients, clip tails, high-pass the maps, or remove a fitted source and declare the remainder pure noise.

Fit separate parameters for each map within the existing compact family:

\[
C_G(p,q)=s_G(p)s_G(q)\,\xi_G(p-q),\qquad \xi_G(0)=1.
\]

Retain anisotropy, negative correlations, actual missing support, finite-domain treatment, supported lag/pair-count information, and explicit regularization/positive-semidefinite treatment. Unsupported correlation ranges remain unknown, not experimentally demonstrated zero.

Do not import ordinary MAP's numerical covariance or admission scale into JINC. Do not equate hits, C, Q/C^2, or coefficient-squared time with empirical precision. Establish sampling-to-amplitude transfer from the observed background under the model, or label extrapolation/unavailability at the source.

For each genuinely linear saved template/aperture/background operator q, calculate q^T C_G q and a diagonal-only diagnostic. Verify propagation with existing generation checks and one manageable direct pixel-pair calculation. Use the larger source exclusion only for background/analytic sensitivity; no second refit ensemble is authorized. Do not resurrect the earlier A+B north/south, mixture-model, or 64-trial sensitivity campaign.

### 9.3 Exactly bounded T10 ensembles

For each supported REF T10 array/mapmaker, run 256 deterministic source-plus-noise realizations. The maximum is six ensembles and 1,536 trials. Refit no M0 ensemble.

Use that map's fitted Gaussian-plus-constant processed-source baseline plus newly generated correlated noise:

\[
M^{(b)}=\widehat\mu_{G,10}+n^{(b)},\qquad
n^{(b)}\sim\mathcal N(0,\widehat C_{G,10}).
\]

Do not add more noise to the already noisy observed map. Explicitly label the Gaussian-baseline approximation, especially when JINC introduces non-Gaussian signed structure. Generate a joint field covering the union of all fit/template/aperture/background-estimation pixels, with correct padding/cropping, amplitude modulation, and missing-support semantics.

Use the existing deterministic generator implementation. Freeze its version, generator algorithm, master seed and per-trial identifiers in the prospective manifest before draws. Derive streams from explicit stable array/mapmaker/trial identities, never worker order or Python's randomized hash. Reuse the established seed convention where available; selecting and recording the exact stable seed values is engineering work, not another owner decision. Repeating the same trial for a numerical-regeneration check is not an additional independent realization.

Refit all seven free parameters with the unchanged real-data initialization/search and bounds; do not initialize at the known generating truth or freeze nuisance parameters. Recompute each estimated aperture background on the same trial. Use that trial's fitted constant when the original aperture rule uses it. Preserve source/background dependence and joint measurement covariance.

Report centered sample scatter with B-1 normalization, noiseless reference, Monte Carlo mean shift/precision, quantiles, and all trial-level constraint/failure flags. Do not divide measurement scatter by sqrt(256), apply mean-shift corrections, replace failed draws, or keep drawing until 256 successes occur.

Retain finite bound-limited fits and flag them. Follow an already accepted exact complete-ensemble rule if one exists. Otherwise any failed nonlinear trial makes that measurement's primary complete-ensemble aggregate unavailable; survivor statistics are partial diagnostics only. Independently valid linear measurements remain usable. An aperture depending on a failed fitted background is not an independently successful linear measurement.

Matching or related random seeds do not supply the physical cross-covariance of the two observed maps. Do not infer the uncertainty/significance of JINC-MAP by adding separate variances. No paired detector-noise or full adaptive-FRUIT ensemble is authorized.

### 9.4 Response-referenced comparison

Use the authenticated archived compact input amplitude delta_a = 0.10 P_a; verify its correspondence to the original saved operand without rescaling that operand. For each existing measurement f, compute:

\[
r_{G,f}=\frac{f(M_{G,+})-f(M_{G,-})}{2\delta_a},\qquad
\sigma_{\mathrm{input},G,f}=\frac{\sigma_{G,f}}{|r_{G,f}|}.
\]

The second quantity is available only for supported, non-negligible response. Freeze any ratio-floor rule prospectively using the existing response utilities, not the observed desirable answer. Show the separate positive/negative responses, even component/nonlinearity, raw output scatter, response slope, and the ratio together.

Interpret this as conditional measurement scatter referenced to the tested pre-PTC compact increment. It is not an absolute flux error, response correction, detection probability, confidence interval, or full-FRUIT uncertainty. Preserve qualifications where the noise model is unrepresentative, source baseline inadequate, response nonlinear, or fits constrained. Do not rank methods solely by lower pixel RMS, a smoother map, or this one ratio.

## 10. Implementation, resources, and validation discipline

Keep numerical ownership in the existing MAP+/JINC and upstream components. The diagnostic harness may authenticate inputs, orchestrate bounded calls, and report; do not grow Engine into a second science path or reactivate a legacy production reduction.

Use compact arrays, finite phase caches, reusable parents/intermediates, bounded segments, and worker-local or otherwise race-free accumulators. No observation-wide dense sample-to-pixel matrix, full sample covariance, or per-contribution provenance archive. Small source-footprint and numerical-fixture dense calculations are allowed.

Use useful available host parallelism across arrays, maps, phases, and trials, avoiding nested numerical-library oversubscription. Choose resources from the actual machine; do not inherit a low worker count without evidence. Keep numerical reductions deterministic under the recorded policy and verify scheduling-independent trial regeneration. Bitwise equivalence across every thread/chunk configuration is not required; matched-oracle numerical bounds and the existing replay rules are required.

Measure actual worker/library-thread counts, loading and authentication, AST association/reconstruction, any saved Apply, kernel/reference/cache startup, certificate construction/evaluation, ordinary/JINC accumulation, diagnostics, covariance fitting/generation/refits, serialization, wall/CPU time, and memory. Separate startup from steady-state cost, process high-water marks from aggregate concurrent memory, and summed CPU/work from campaign elapsed time.

No environment migration, new dependency installation, Unity campaign, broad optimization, or unrelated repair is authorized. Use available dependencies or a narrow standard-library validation helper. Report any genuine remaining technical obstruction with its attempted bounded remedies; do not manufacture a new scientific-approval requirement.

Run affected native/JINC/AST/rejoin, successor, baseline, default-CLI and required configuration tests, plus focused certificate/response/covariance/report regressions. Record actual test results, not inherited counts as though rerun. Separate executed local evidence from unperformed Spack V2/Unity validation.

Obtain the existing required independent reviews at appropriate gates, including the numerical-bound argument, actual selected kernels and geometry, input equivalence, preservation of upstream state, output/certificate correspondence, complete trial reporting, numerical tables, and rendered figures. Fix demonstrated local defects, preserve failed attempts, rerun affected work, and complete the authorized experiment. Do not tune scientific choices to make a review or comparison favorable.

## 11. Prospective record, outputs, and completion

Use a **fresh** bundle; do not overwrite the sealed preparation bundle. Suggested destination:

```text
successor-point-152389-map-only-jinc-full-20260924/
```

Use a distinct handoff, for example:

```text
handoff/POINT_152389_MAP_ONLY_JINC_FULL_2026-09-24.md
```

Before new scientific comparison, freeze this directive, current source/binary/environment identities, exact input lineage, all selected tuples including a2000 3/16, P_a values and source hashes, numerical profile/method, snapshot/reference matrix, AST and WCS bindings, common-sample audit, original/common domains, both noise masks, phase offsets, measurement rules, seed convention/values, inherited and new numerical tolerances, and execution/dependency order. Record realized support after computation without retrospectively changing the prospective geometry or policy.

Retain primary ordinary/JINC maps, required base accumulators/replay state, certificates and support/cause diagnostics, independent checks, exact injection references, signed differences, full measurement/control tables and fit flags, covariance applicability and sensitivity summaries, trial-level values/failures, runtime/memory evidence, reproduction commands, test/review records, and final manifest. Reference large immutable ancestors rather than copying raw archives.

Provide a compact, legible figure set covering matched REF maps/support; compact and shoulder response/reference/residuals; pixel-phase centroid/shape/peak behavior; directional background covariance; response-referenced measurement scatter; and meaningful numerical-support/certificate failures. Use matched angular scales, correct handedness, explicit signed scales, and disclosure of any panel rescaling. Inspect the rendered figures for clipping and misleading normalization.

The owner-facing assessment must answer:

1. Did the same processed samples and original histories feed both mapmakers? Did ordinary replay and the JINC numerical certificate pass? What cells/pixels/measurements remain unavailable, and why?
2. What changed in gridding response, centroid/core shape, signed lobes/bowls, phase sensitivity, and native/common support?
3. How do compact positive/negative and shoulder responses compare against each operator's own injection-only reference? Which limitations are inherited from the saved cleaning/feedback histories?
4. Does apparent noise reduction translate into useful measurement precision after accounting for response and covariance? Which conclusions depend on the Gaussian baseline, nonlinear constraints, or source-location background transfer?
5. What are the actual runtime/memory and certificate costs, and is there a demonstrated bounded implementation bottleneck?
6. Which **one** next action is justified: a JINC-feedback quantity/interface/round-trip investigation, a specific JINC numerical/implementation repair, or a correlation-aware admission experiment on an established map representation? Recommend it; do not execute it.

Add a short future-feedback boundary note: the JINC map is a signed, normalized local estimate of the same processed brightness in mJy/nominal beam, not unconvolved sky truth. Explain what still needs to be defined to project a candidate to sample space without applying a response twice. Implement no new inverse, back-projector, deconvolution, or candidate policy.

Report every planned matrix/ensemble cell as completed, completed with qualification, unavailable with cause, or not run with cause. Local unavailable pixels can coexist with a completed map; unavailable required measurement domains must remain visible. Do not call the whole scientific comparison complete solely because code/tests, preparation, or certificates are complete. Conversely, a scientifically unfavorable result or an explicitly retained model qualification does not make an otherwise completed experiment a failure to execute.

Commit reviewed changes locally under the existing worktree practice. Do not merge, push, reset, discard unrelated work, rewrite old evidence, or activate production. Preserve a reproducible final source/binary/environment binding and report the final local commit/worktree status.

## 12. Final stop and explicit exclusions

Stop after the specified maps, reference/phase diagnostics, background analysis, supported 256-trial ensembles, assessment, validation, independent review, and local evidence closure.

No new PTC Learn/ALS, adaptive residual history, threshold, damping, gain, feedback candidate, m10 application, cycle 11, JINC-driven FRUIT, correlation-aware/RBF admission, covariance-weighted source fit, kernel/phase/support sweep, detector/RTC/CAL repair, new observation, full adaptive-noise ensemble, calibrated flux correction, production/default change, integration, or push is authorized.

**Completion means a controlled, numerically supported answer about this selected JINC realization on the same processed data. It does not mean the shoulder is solved, information removed upstream has been restored, optimal JINC parameters have been found, or JINC feedback is ready.**
