# WP-7 Network-Timing Scientific-Owner Authority Correction

Date: 2026-08-29

Scientific owner: Grant Wilson

Status: approved bounded successor authority for the WP-7 timestream route

Supersedes only the observation-wide detector-grid interpretation identified in
the [network-timing crosswalk](WP7_NETWORK_TIMING_AUTHORITY_CROSSWALK_2026-08-29.md).
Every unrelated SCI-ALIGN v0.1/r0.3, SCI-RTC v0.1/r0.12, SCI-AST v0.1/r0.3,
and SCI-PTC v0.1/r0.5 decision remains unchanged.

## Governing Rule

Network-specific timing is authoritative by default. A cross-network common
time grid shall be introduced only when the mathematics of a particular
operation requires simultaneous measurements from more than one network.

An operation requires that relation only when an output estimate at an analysis
epoch depends jointly on measurements from two or more networks at that epoch.
Processing several networks, sharing configuration, pooling statistics, using
a rectangular container, or implementation convenience does not meet this
criterion.

## Scientific Identity And Timing

Ordinary ALIGN establishes a network-scoped occurrence relation

```text
(observation, network, occurrence) -> reconstructed time
```

and binds telescope or auxiliary facts to that exact network occurrence and
time. The occurrence may retain native row, packet-counter, acquisition
interval, clock-correction, uncertainty, and other producer facts, but a time
value or row position alone is not its identity.

Paired `x/r` values share the exact originating network occurrence and temporal
relation. Their numerical validity, causes, and availability remain
coordinate-local where the existing contracts require that distinction.

Ordinary RTC consumes and produces network-keyed timed streams. For the exact
`M=1` pass-through, every output occurrence identity and output time equals its
network input exactly. An RTC operation that changes sampling defines a new
output occurrence, time, representative-source, and support relation separately
for every network unless that named RTC method is explicitly authorized as a
cross-network synchronous operation.

Within one network, the phrase "paired occurrence axis" means the shared
occurrence relation of `x` and `r`. It is not a common analysis grid.

## Ordinary Consumers

The following operations use network timing and do not require a cross-network
common analysis grid merely by entering the stage:

- ordinary RTC, including network-local temporal learning and filtering;
- CAL;
- ordinary AST association and coordinate construction;
- MAP/JINC accumulation from independently timed detector occurrences; and
- network-level PTC/PCA.

This decision does not change any stage's existing unit, coordinate, validity,
response, covariance, eligibility, or numerical-policy authority.

## Explicit Common Analysis Grid

"Common analysis grid" is reserved for a requested cross-network relation.
Array-wide PTC/PCA and any future RTC method whose equations couple
simultaneous measurements from multiple networks may request such a relation.
The request identifies the consuming method, participant networks, analysis
epochs, admission rule, support, and failure behavior.

ALIGN owns construction because ALIGN owns timing knowledge. The relation is a
derived, non-destructive view. Every admitted analysis-grid cell retains:

- analysis-grid occurrence and time;
- source network and source occurrence;
- source-network time and assignment residual or other exact relation;
- source validity, causes, origin, and support; and
- the ALIGN plan/relation identity.

The view does not authorize detector-signal interpolation, synthesis, zero
filling, cross-network value mixing, or loss of the source-network axis.
Strict half-cadence slot admission remains ALIGN-owned and applies when this
explicit relation is requested; it is not an ordinary-RTC ingress predicate.

## Package Consequences

### SCI-ALIGN

The singular observation-wide detector-reference lattice and stable identity
`(observation, s)` are superseded for ordinary detector streams. Ordinary
identity is network-scoped. A requested common analysis grid has its own
derived relation and identity and must not replace source occurrences.

### SCI-RTC

"Assigned grid", "aligned grid", and "common grid" in the paired RTC
requirements are superseded where they were interpreted as one cross-network
axis. For ordinary RTC they mean the exact network-scoped paired occurrence
axis. `x` and requested conditioned `r` remain on the same axis within their
network. Pair-coherent actions remain coordinate-pair coherent and do not
imply cross-network synchronization.

### SCI-AST

Ordinary AST binds geometry and observing state at each exact network occurrence
time. An AST implementation may process or store multiple networks together,
but that container shape is not a timing projection. AST consumes a common
analysis grid only when its named downstream operation requests one.

### SCI-PTC

Network-level PTC/PCA uses one network's time axis and is independent of other
networks' occurrence support. Array-wide PTC/PCA is mathematically
cross-network and therefore must explicitly request an ALIGN-owned common
analysis grid before constructing its sample-by-detector matrix. Network- and
array-level grouping remain alternative configured routes.

## Preservation And Change Control

This correction changes timing identity and the placement of cross-network
projection only. It does not reopen RTC artifact policies, filtering,
level-shift amplitudes, donor behavior, calibration, astrometric geometry,
PTC rank or estimator policy, VAL policy, MAP/JINC estimators, response,
covariance, uncertainty, or publication authority.

The prior observation-wide-grid clauses remain historical frozen-package
content. They are not silently edited or reinterpreted. This dated successor
authority and its explicit crosswalk control the WP-7 successor until the
scientific-contract library publishes a mechanically consolidated revision.

No implementation-conformity, observational-validation, performance,
readiness, or production-activation claim follows from this decision alone.
