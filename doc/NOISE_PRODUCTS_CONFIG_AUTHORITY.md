# Noise-Products Config Authority

This document freezes the bounded Phase 2 contract for the six current
`noise_maps.*` inputs. It does not change jackknife generation, empirical
variance calculation, weight calibration, filtering, or FITS output numerics.

## Authority Flow

Merged YAML is read once into a requested `NoiseConfig`. Optional legacy keys
retain their established defaults: realization output defaults off, while
empirical products and empirical weighting inherit requested noise enablement
when omitted.

`NoiseExecutionPlan` preserves that request and resolves effective activation
from effective mapmaking state. Disabled effective state uses zero noise maps
without rewriting the request. A one-way adapter supplies effective count and
detector-randomization policy to the existing observation and coadd map
buffers. Those numerical buffers never write values back into typed config.
The Wiener-filter dependency check reads this effective noise policy rather
than the preserved request.

## Randomization

Citlali already uses deterministic Boost MT19937 generators constructed at
each reduction-pipeline entry. The implicit default is made explicit as fixed
internal seed `5489`; it is recorded as effective provenance and is not a new
user-facing knob. Detector randomization remains the requested boolean policy.

## Realized Cardinality

At successful CLI completion, noise provenance derives final-iteration map
cardinality from the required mapmaking execution contract. It records logical
observation and coadd realization counts, empirical-product map count, and
realized realization-image write count. Required map output failures already
propagate, so successful run completion establishes completion for those
logical products.

## Gate

Local acceptance requires CLI/test builds, the full CTest suite, a frozen
six-path boundary audit, semantic reduction-audit tests, all config profiles,
and full config preflight. Unity validation requires disabled point, enabled
generation-only science, and a small full-output fixture with empirical
products and realization writes enabled.
