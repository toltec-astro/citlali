# SCI-RTC v0.1/r0.9 binding scientific-owner decisions

Date: 2026-08-20

Status: binding owner authority for the bounded r0.9 cleanup. These decisions
clarify scientific architecture and reporting obligations. They introduce no
numerical thresholds, operation-specific parameter values, implementation
conformance claim, observational qualification, or production-readiness claim.

## Decision 1 — Operation availability is not role-partitioned

All RTC v0.1 operation classes are available to every application context
until evidence or explicit policy disqualifies one for a particular use.
Available or admitted does not mean enabled, selected, numerically resolved,
or scientifically qualified. Historical labels such as Beammap, Pointing,
OOF, Science, and diagnostic-only are contextual facts, not operation gates.

## Decision 2 — Three one-way lifecycle objects

RTC preserves three distinct one-way objects:

1. `RTCApplicationContext`: requested use, exact input and interval, named
   consumers, and external constraints;
2. `RTCResolvedPlan`: the immutable selected numerical plan after all required
   evidence and compatibility predicates pass; and
3. `RTCRealizedRecord`: the exact operations, intervals, state, exceptions,
   outputs, and completion that actually occurred.

A later object does not rewrite an earlier one. More detailed substates may be
retained within this lifecycle without collapsing these ownership boundaries.

## Decision 3 — One consumer-neutral atomic RTC bundle

RTC produces one consumer-neutral atomic bundle. Downstream consumers bind the
members and separately requested diagnostic detail they need. Consumer or role
labels do not change the bundle schema or raw signal meaning.

## Decision 4 — Preserve upstream mapping and paired identity

The upstream IQ-to-$x/r$ and ALIGN mapping authorities remain distinct and are
preserved in lineage. Exact paired $x/r$ identity is retained, while numeric
and scientific validity of $x$ and $r$ remain independent.

## Decision 5 — Coordinate authority is explicit

Coordinate-dependent RTC operations require explicit coordinate product
identity, frame, topology, detector binding, support, and validity. RTC does
not infer missing coordinates, clamp invalid coordinates into a usable domain,
or silently convert frames.

## Decision 6 — Non-finite state is typed and cause-preserving

Non-finite samples, factors, coordinates, coefficients, and state follow typed,
cause-preserving rejection, unavailability, or an explicitly authorized prior
recovery rule. They are not silently coerced to zero or represented only by an
undifferentiated generic non-finite.

## Decision 7 — Covariance and uncertainty claims disclose scope

Whenever RTC makes a covariance or uncertainty claim, the claim explicitly
states the components and correlations it includes and excludes. Excluded and
unknown components or correlations are not zero; unknowns remain unavailable.
A truthfully labeled conditional or component-limited claim is permitted and
need not supply a maximal total-covariance manifest. Only declared complete
coverage may be called total or full precision.

## Decision 8 — Despiking transforms data; normal reporting is compact

When despiking is selected, it actually replaces or recovers accepted target
$x$ cells under the resolved policy, or records the selected explicit
failure/no-correction disposition. Detection-only evidence is not called
despiking. Normal production output retains compact treatment state and useful
spike-population counts and characteristics. Complete event-by-event fits and
realized donor links may be separately requested verbose or diagnostic detail.
Toggling that detail is scientifically and numerically inert.

## Relationship to prior authority

These decisions supplement, and do not reopen, binding Decision 9 in
`SCIENTIFIC_OWNER_DECISION_R0.8.md`: level shifts remain additive baseline
changes with finite physical-time transition support, unmodeled transition
cells, optional valid stable-plateau offset correction, and no gain-change
model in v0.1.
