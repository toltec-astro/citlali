# SCI-PTC v0.1 — Bounded Method Reference Boundary

Status: scientific-owner approved Stage B author reference

Date: `2026-08-19`

## Purpose

This record gives the Stage B author bounded, verified context for established
method families without making any paper the TolTEC contract. The paper
identities are citation authorities. The author may use only the paraphrased
claims and limitations below and may not open the full papers or import an
unstated equation, threshold, default, performance claim, or assumption.

## Admitted Reference Context

### Weighted PCA with missing values

Stephen Bailey, [*Principal Component Analysis with Noisy and/or Missing
Data*](https://arxiv.org/abs/1208.4122), 2012.

- Bounded claim: a noise-weighted expectation-maximization PCA can use
  measurement-error weights, with a missing datum represented as zero
  statistical weight rather than a zero-valued observation.
- Limitation: this does not select EM-PCA for TolTEC, prove its noise model,
  define PTC flags, or authorize zero-filled ordinary PCA.

### Robust low-rank plus sparse decomposition

Emmanuel J. Candès, Xiaodong Li, Yi Ma, and John Wright,
[*Robust Principal Component Analysis?*](https://arxiv.org/abs/0912.3599),
2009.

- Bounded claim: under stated structural assumptions, a matrix decomposed into
  low-rank and sparse components can be recovered by a principal-component-
  pursuit formulation, including some missing-entry cases.
- Limitation: sparse corruption is a model, not a synonym for every TolTEC
  flag or transient. The paper does not establish TolTEC identifiability,
  thresholds, contamination fractions, or response preservation.

### Nonlinear PCA transfer measurement

Thomas P. Downes et al., [*Calculating the transfer function of noise removal
by principal component analysis and application to AzTEC observations*](https://arxiv.org/abs/1103.3072),
2011.

- Bounded claim: interpretation of maps cleaned by nonlinear/iterative PCA
  requires measuring the response of the actual reduction to the signal of
  interest; a nominal component count is not the transfer function.
- Limitation: the published application concerns AzTEC and compact point
  sources in its declared domain. It does not establish TolTEC response,
  thresholds, extended-source transfer, or production performance.

### Iterative source/noise separation

A. Kovács, [*CRUSH: fast and scalable data reduction for imaging
arrays*](https://arxiv.org/abs/0805.3928), 2008.

- Bounded claim: an iterative sequence of estimators can separate source and
  noise components for imaging-array data, illustrating that correlated-noise
  treatment may be composed with sky estimation rather than being one isolated
  PCA subtraction.
- Limitation: such joint recurrence crosses the base SCI-PTC boundary into
  FRUIT/MAP or successor authority. CRUSH does not define TolTEC's operator.

### Correlated-noise mapmaking

G. Patanchon et al., [*SANEPIC: A Map-Making Method for Timestream Data From
Large Arrays*](https://arxiv.org/abs/0711.3462), 2007.

- Bounded claim: a maximum-likelihood mapmaker can explicitly model strong
  inter-detector timestream correlations, demonstrating that subtracting a
  finite set of modes is not the only scientifically meaningful treatment of
  correlated noise.
- Limitation: this is an adjacent PTC/MAP alternative, not authorization for a
  SANEPIC implementation, a covariance model, or a base transformed-TOD PTC
  estimator.

### Model-specific singular-value thresholds

Matan Gavish and David L. Donoho, [*The Optimal Hard Threshold for Singular
Values is 4/sqrt(3)*](https://arxiv.org/abs/1305.5870), 2013.

- Bounded claim: an asymptotically mean-square-error-optimal hard singular-
  value threshold exists for a particular low-rank matrix observed in the
  paper's white-noise framework, with aspect-ratio-dependent generalization.
- Limitation: those assumptions do not provide a universal threshold for
  heteroscedastic, flagged, correlated, source-bearing TolTEC data. PTC rank
  selection remains tied to the approved conjunctive least-aggressive
  admission policy and admitted evidence.

## Permitted Synthesis

The rationale may use these references to explain that:

- missing-data handling is part of the estimator, not a preprocessing detail;
- low-rank, sparse-corruption, and robust-loss assumptions are different;
- nonlinear/data-dependent cleaning needs a measured or otherwise supported
  response family;
- iterative joint sky/noise separation and correlated-noise mapmaking are
  important alternatives but belong outside the base transformed-TOD PTC
  authority; and
- no rank, threshold, or estimator is universally optimal without its model
  and declared scientific admission criteria.

The author must not use the references to claim TolTEC implementation,
conformity, validation, observational performance, or production readiness.
