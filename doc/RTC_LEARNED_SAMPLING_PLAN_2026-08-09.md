# RTC Learned Sampling Plan

Date: 2026-08-09

Status: owner-approved design; implementation and numerical tolerances not yet
launched or selected

## Objective

Add an optional learned RTC sampling mode without changing existing fixed
behavior. Learned mode uses a conservative metadata-derived bootstrap during
learning, resolves a complete immutable low-pass/downsampling plan, and applies
that plan in later apply iterations.

The initial objective is `maximum_safe_reduction`, not unconstrained
mathematical optimality: maximize the integer downsampling factor only after
the astronomical-transfer, alias-rejection, sampling, and downstream
compatibility constraints pass.

## State Model

| State | Authority | Required meaning |
| --- | --- | --- |
| Requested | Accepted low-level configuration | `fixed` or `learned`, allowed factors, fallback policy, and approved tolerances |
| Bootstrap | Observation preflight | Conservative learn-phase factor/filter derived from metadata; no detector-signal optimization |
| Learned | Reduction learning owner | Complete candidate matrix and its exact input identities |
| Resolved | Learn/apply boundary | One immutable downstream-compatible plan selected from admitted candidates |
| Applied | RTC execution owner | Exact factor, FIR, phase, rate, support, and completion state that actually ran |

No later state writes back into the request. The numerical processor consumes
the resolved plan through a one-way execution adapter.

## Metadata Bootstrap

The first implementation consumes:

1. measured native detector cadence and its validity;
2. approved per-array beam FWHM, or beam covariance when available, with source
   identity and validity;
3. telescope time and physical AltAz pointing over each science-valid scan;
4. scan identity, boundaries, Hold/turnaround exclusion, and a one-telescope-
   sample boundary guard;
5. permitted integer factors and realizable FIR family; and
6. owner-approved preservation and alias tolerances.

For adjacent valid telescope rows, speed is derived from telescope position
and telescope time. Nonfinite positions/times, nonpositive time steps, gaps,
and inadmissible pointing jumps do not enter the maximum. The safety authority
is the maximum remaining in-scan speed. p50, p95, and p99.5 are retained as
diagnostics and cannot replace the maximum.

The first bootstrap plan is common to the observation. It uses the smallest
admitted beam and largest valid scan speed. If no safely decimated candidate
exists, native cadence is the supported conservative fallback. If native
processing is unavailable or the required metadata cannot establish a safe
plan, learned mode fails before RTC science processing.

## Analytical Candidate Evaluation

For a circular Gaussian beam with angular FWHM `theta_fwhm` scanned at speed
`v`, the normalized temporal power response is

```text
sigma_t = theta_fwhm / (2 sqrt(2 ln 2) v)
P_beam(f) = exp(-4 pi^2 sigma_t^2 f^2)
f_half = sqrt(ln 2) / (2 pi sigma_t)
```

For an elliptical beam, evaluation uses the scan-direction projection of the
beam covariance and maximizes the resulting temporal bandwidth across valid
intervals.

For every candidate integer factor `M`, the planner calculates:

- output rate and Nyquist frequency;
- samples per beam FWHM;
- the exact realized FIR magnitude and software phase/group delay;
- compact-source attenuation/broadening from the beam-times-FIR response;
- astronomical power that would fold into the retained band;
- general stopband rejection independent of the astronomical beam; and
- transition margin and filter realizability.

The exact phase-zero decimator is `y[n] = x[M n]`. This calculation proves the
software response on the assigned compatibility grid; it does not determine
the physical detector integration event or authorize an absolute timing or
sky-placement correction.

## Resolution Policy

The learner may retain preferred candidates for each scan and array. Stage A
resolves them under one common observation factor and FIR so that all arrays
and scans retain a common output cadence and transfer. The selected plan is
the largest factor satisfying every hard constraint; ties may prefer smaller
accepted noise bandwidth and then lower filter cost.

No candidate is made admissible by changing a tolerance, dropping a limiting
scan/array, replacing maximum speed with a percentile, or selecting a fallback
that was not requested.

Per-array fixed-observation plans and per-scan plans are later scopes. They
require explicit contracts for heterogeneous time grids, scan concatenation,
flag aggregation, filter transients, PSD/frequency identity, PTC conditioning,
coverage/weights, map response, restart, and persisted product shapes.

## Learn/Apply Boundary

The resolved plan records:

- observation, scan-set, array-set, beam, telescope, native-rate, and input
  identities;
- bootstrap plan and all admitted/rejected candidates with reasons;
- selected factor, output rate, phase label, FIR coefficients/digest,
  normalization, support, state/reset, and edge policy;
- calculated astronomical response, alias bounds, and limiting constraint;
- learner algorithm/version, requested policy, tolerances, and completion; and
- physical event semantics as unavailable while that dependency remains open.

Apply validates these identities and executes the exact plan. It cannot add
new candidates, retune coefficients, or silently fall back. Restart restores
the state-complete resolved plan and rejects incompatible inputs or policy.

Learning state that must survive a cadence change uses native detector row or
native-time support. Bootstrap-downsampled positions are not durable sample
identities. The transition from a bootstrap map to the first apply map is a
known transfer change and is excluded from convergence evidence.

## Product And Provenance Burden

The resolved plan is an authoritative intermediate computational state because
RTC and mapmaking consume it. It is not an independent sky estimator. Final
maps and requested derived products bind its digest and material realized
summary. An optionally persisted complete candidate matrix is diagnostic and
does not require archival-grade replay unless a declared consumer requires it.

## Failure And Advisory Policy

- A science-producing learned request fails before RTC when no admissible plan
  exists, required metadata are invalid, the resolved identity mismatches, or
  the analytical transfer/alias gates fail.
- A fixed request that is scientifically undersampled follows its separately
  approved admission policy; learned mode does not silently repair it.
- Oversampling is valid. It may produce an efficiency/noise-bandwidth advisory
  and a learned recommendation, but never a failure solely for being high.
- A diagnostic-only RTC role may consume an under-resolved product only after
  a separate explicit product-role decision; this plan does not create that
  exception.

## Staged Delivery

### Stage A: observe and resolve without execution change

- implement typed requested/bootstrap/learned/resolved plan values;
- calculate maximum-valid scan speed and analytical candidate matrix;
- emit deterministic diagnostics and provenance without inventing default
  scientific tolerances;
- resolve and compare a recommendation with fixed execution only when exact
  owner-approved tolerances are supplied;
- do not alter the factor, filter, detector samples, flags, map, or product
  cadence.

### Stage B: common observation apply

After Stage A review, numerical-tolerance approval, exact application-base
selection, and fresh authorization, execute one common resolved plan during
apply. Validate restart, native-row learning-state mapping, convergence
boundaries, writers, and downstream products.

### Later optional scopes

- noise-aware objective;
- per-array fixed-observation filter or cadence;
- per-scan cadence;
- heterogeneous-transfer map combination or homogenization; and
- empirical/on-sky transfer characterization.

Each is a separate scientific and architectural decision.

## Decisions Still Required

Before Stage B, the owner must approve:

- compact-source response-loss/broadening tolerance;
- general alias-leakage/stopband tolerance;
- minimum sampling or equivalent residual beam-power threshold;
- allowed factor set and maximum filter cost/order;
- native-cadence fallback versus fail policy by reduction role; and
- whether any array-specific filtering is allowed while cadence remains
  common.
