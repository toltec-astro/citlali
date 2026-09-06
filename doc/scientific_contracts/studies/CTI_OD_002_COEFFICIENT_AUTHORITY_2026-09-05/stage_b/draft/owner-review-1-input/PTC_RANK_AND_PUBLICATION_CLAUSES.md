# Frozen PTC clauses recovered for owner review 1

## Program adherence and prior-work recovery

The owner asked to establish the inherited positive-rank authority and to
identify any inherited publication constraint. These are minimal verbatim
scientific clauses from the already recovered frozen SCI-PTC v0.1/r0.5
requirements, not implementation evidence or new derivation. Disposition:
**cite** for rank/route provenance and required-product completion; do not
rederive PTC fitting or extend these clauses to unsupported route families.
The owner-requested provenance recovery admits only the four clauses below.

The existing Stage A packet and manifest remain immutable. This supplement
is a separately content-bound scientific input for the owner's targeted
revision request. It does not grant access to omitted sources.

## Exact source and freeze binding

- Frozen authority: SCI-PTC v0.1/r0.5, owner freeze dated 2026-08-23.
- Freeze record: `doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md`.
- Freeze-record SHA-256: `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66`.
- Requirements source: `doc/scientific_contracts/packages/SCI-PTC/v0.1/src/common/requirements.tex`.
- Whole-source SHA-256: `f2047600cc06c234a78aa3ddf6a575abf2f9592b3e3da810491f6db0150fe21c`.
- Verified exact at original canonical `00b974c9039d4c3025dcce18f26bca69d36af9c3` and current canonical `5244e04638db5aa92180feab52acd782229cfa32`.
- Requirements 076–077 concerning PTC-disabled and the required transformed
  parent already exist in the released `PTC_COEFFICIENTS.md`; reuse them there.

The identifiers and paths above are provenance locators, not retrieval
permission. No surrounding fitting, audit, implementation, or validation
material is admitted. Required-product atomic completion does not itself
specify observation-sized versus segment-sized publication; the author must
state an exact bounded proposal under the owner's publication request and
identify any conflict rather than silently relaxing inherited completion.

## PTC requirement 072

Exact whole-source line 99.

```tex
\PTCRequirement{072}{Atomic required outputs}{Rationale 8.1}{D005}{None}{A required product shall be complete only after all required components and links are durable. Partial publication or required-output failure shall propagate and shall not be labeled complete.}
```

## PTC requirement 090

Exact whole-source line 124.

```tex
\PTCRequirement{090}{Ordinary route identity}{Rationale 4.2--4.3}{WP1-D006--D008}{SCI-CAL}{The first ordinary route shall be one explicitly requested calibrated-$x$, configured-rank, group-local PCA/SVD operation with exact CAL parent, segment, output identity, lifecycle, and failure state.}
```

## PTC requirement 094

Exact whole-source line 132.

```tex
\PTCRequirement{094}{Explicit positive rank}{Rationale 4.3}{WP1-D007}{Owner policy}{The ordinary request shall supply an integer rank $k_{{\rm req},g}\ge1$ for each group or one explicitly common positive rank applied to named groups. Each group shall independently verify that the requested rank is realizable on its exact support and degeneracy state.}
```

## PTC requirement 095

Exact whole-source line 134.

```tex
\PTCRequirement{095}{Rank failure is not routing}{Rationale 4.3 and 8.2--8.3}{WP1-D005,D007}{None}{Rank zero, noninteger rank, or unrealizable positive rank shall fail closed with exact group and cause. It shall not yield centering-only output, be clipped or substituted, become PTC-disabled, select RTC-terminal export, or produce a map.}
```
