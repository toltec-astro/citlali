# SCI-FRUIT v0.1 — Replication, Dependence, And Inference Target

Status: **Stage A requirement; no sampling design is approved**

Every science profile and frozen qualification protocol must state:

- whether the inference target is a finite population or a superpopulation;
- the primary independent sampling unit;
- clustering by observation, source, weather, scan, detector, and any other
  shared realization;
- which modes, orientations, amplitudes, and iterations repeat within a unit;
- the candidate/historical pairing unit;
- the cluster-aware covariance, randomization, or resampling unit;
- dependence among metrics, scales, and strata;
- nominal sample size and justified effective sample size;
- the missingness, failure, scientific-unavailability, and exclusion
  mechanisms; and
- the claim domain supported by the sampling design.

The conservative default is:

> Qualification applies to the exact frozen finite held-out population.

A superpopulation claim requires separate authority for its sampling design or
generative model and a matching uncertainty construction. Multiple injections,
spatial modes, orientations, amplitudes, or iterations within one observation
are repeated measurements, not independent astronomical observations by
default. Pairing does not remove clustering; cluster-aware inference remains
required.

No population, independent unit, covariance method, or superpopulation model
is selected by this file.
