# SCI-PTC v0.1 — Internal Discovery Dossier

Status: Stage A internal scope record; permanently excluded from the
implementation-blind scientific author packet

Date: `2026-08-19`

## Purpose

This dossier records what the manager inspected to avoid repeating prior work
and to define the package boundary. It is not scientific authority, an
implementation assessment, a repair plan, a validation report, or production
authorization. Facts learned from implementation-facing material may become
sanitized questions or responsibility boundaries, but never scientific
answers merely because current software behaves that way.

## Discovery Surface

The manager inspected or searched:

- the historical `SCI-PTC-001` independent core, audit inventory, owner
  decisions D001--D006, optional response plan, and later incoming handoffs;
- current SCI-RTC and SCI-CAL package boundary records;
- current Citlali `ARCHITECTURE.md` and `SCIENTIFIC_CONVENTIONS.md`;
- the historical model-protected-notch design note; and
- repository history sufficient to detect later PTC-named authority;
- three scientific-owner reviews of the Scope Brief; and
- six primary method papers solely to verify the bounded claims recorded in
  `AUTHOR_METHOD_REFERENCE_BOUNDARY.md`.

No Unity access, reduction, numerical execution, test, source audit, or repair
was performed.

## Implementation-Informed Scope Map

The current documented execution surface suggests that PTC sits between
detector-timestream conditioning/calibration and map accumulation, operates on
sample-by-detector matrices, produces detector-resolved coefficients and
diagnostics, and retains state across some iteration/restart paths. These
observations justify asking about identities, shapes, lifecycle, coefficients,
restart materiality, response, and downstream consumers. They do not establish
the correct estimator, support, normalization, thresholds, convergence,
response, covariance, or scientific validity.

The documented current unit conventions use `mJy/beam` for active map products
and attach units to TOD products explicitly. The scientific package therefore
requires the admitted CAL parent and exact unit to be explicit; it does not
infer a unit from a product name or current file token.

The historical model-protected-notch note places temporal line suppression on
model-subtracted residual TOD immediately before PTC. Scientifically, this
exposes three separable responsibilities:

1. RTC or another named temporal-conditioning operator owns notch filtering
   and its response;
2. FRUIT or another named recurrence owner owns model subtraction, add-back,
   and feedback parentage; and
3. PTC owns the masks and admitted residual state actually used to fit and
   apply correlated-mode cleaning.

The v0.1 package must not turn a convenient hook location into silent ownership
of all three operations.

The owner review further exposed three boundaries that the initial Stage A
draft had not stated sharply enough:

1. PTC owns a calibrated-`x` sample-domain subspace estimate, subtraction,
   and response—not a physical attribution of every correlated mode;
2. raw `r` remains part of complete RTC ancestry, while only a separately
   conditioned and exactly related `r` product may enter an optional PTC
   diagnostic branch; and
3. a PTC response companion is distinct from the optional map-center
   point-source functional, which additionally requires an exact named MAP
   reference operator.

These are scope and ownership corrections. They are not claims about current
implementation. On `2026-08-19`, the scientific owner resolved the remaining
choice diagnostic-only for the first implementation/base v0.1: `r` analysis
may not supply calibrated-`x` subtraction or control `x` membership, output,
or coefficients.

## Quarantined Historical Status

The completed 2026-08-08 audit described candidate-specific implementation,
dependency, contract, and evidence findings and retained `existing_use_only`
status. Later owner amendments changed several policy interpretations. Those
records remain historically important, but none enters Stage B authorship or
determines the new scientific contract by inspection.

Likewise, later RTC, CAL, ALIGN, MAP, and PTC handoffs, repairs, re-audits,
tests, validation records, and production claims are excluded. They may be
consulted only in a future, separately authorized conformance program after
the PTC scientific authority is frozen.

## Adjacent Authority Status

- SCI-RTC v0.1/r0.7 is a final candidate, not frozen scientific authority.
- SCI-CAL v0.1/r0.3 has a frozen rationale architecture but unresolved Q01--Q09
  and is not frozen scientific authority.
- SCI-MAP v0.1 remains a conditional downstream package with unresolved owner
  decisions.
- The historical PTC decisions D001--D006 are approved scientific policy and
  may be sanitized into this package without importing their audit context.
- The `2026-08-19` review-derived D007--D017 are proposed scope decisions and
  do not become author authority until the owner approves the revised packet.

Consequently, SCI-PTC may define a structurally exact contract while marking
upstream-dependent numerical or calibrated claims unavailable.

## Firewall

The Stage B author may not open this file or any raw discovery source named
here. The author receives only the exact content-bound packet in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md). If that packet is
insufficient, the author must return a precise owner question rather than
search implementation or audit history.
