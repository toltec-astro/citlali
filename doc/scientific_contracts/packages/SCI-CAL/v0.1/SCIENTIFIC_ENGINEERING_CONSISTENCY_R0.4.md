# SCI-CAL v0.1 Rationale r0.4 / Engineering r0.3 Consistency Report

Date: `2026-08-20`

Status: fresh high-effort implementation-blind review passed; scientific
authority not frozen

## Verdict

`PASS` — no blocking science/engineering inconsistency was found.

The review used only the approved scientific-authority packet, active
governance and crosswalk records, shared implementation-independent contract
modules, current document sources, and the two canonical PDFs. It did not
inspect implementation code, tests, audit material, repository history, the
internal dossier, prior consistency reports, or prior reviewer output. It
does not establish implementation conformity.

## Consistency Result

- Beammap/source-APT production, TolProj child transformation, immutable
  TolTECA delivery, once-only SCI-CAL consumption, and downstream MAP/FLT
  response ownership agree across the two views.
- Source/child APT identity, occurrence-scoped association, delivery identity,
  parent/child distinction, and the no-reinterpretation boundary agree.
- Factor direction, units, top-of-atmosphere plane, source/target atmosphere
  separation, interpolation-before-reciprocal rule, support intersection, and
  once-only application agree.
- Atmosphere and broadband limitations, pipeline noncommutativity, conditional
  covariance/variance/weight semantics, response-quality limitations, claim
  layers, and conformance dispositions agree.
- Q01--Q09 remain open with the same owners, required evidence, and
  claim-specific consequences; the earlier Q001 is fully subsumed by Q06.
- The inventories remain complete and sequential at 11 assumptions, 50
  requirements, and 30 edge predictions, with complete crosswalk coverage.
- The r0.4/r0.3 version, date, provenance, and frozen four-item reference
  boundary are coherent. Both canonical PDFs match their sources and remain
  readable on every rendered page.

## Nonblocking Observations

The reviewer recorded four editorial observations that do not alter the pass
verdict:

1. Source-observation index `s` is inferable from the notation table and
   equations but is not given its own explicit prose definition.
2. Historical filename `SCIENTIFIC_OWNER_DECISIONS_V0.2.md` retains the old
   pre-version-axis naming convention; active records correctly identify
   contract v0.1, rationale r0.4, and engineering r0.3.
3. The downward connector in rationale Figure 1 lands in whitespace rather
   than visibly touching the selected-child box; the caption and surrounding
   text state the relationship unambiguously.
4. The REQ-049 detailed-crosswalk location could distinguish more explicitly
   between the scientist-facing narrative and the formal shared engineering
   table; the scientific trace itself is sufficient.

These observations do not require an owner decision, change scientific
meaning, or justify another stylistic round under `CAL-GOV-D010`.

## Remaining Gate

The repaired pair is ready for explicit scientific-owner freeze disposition.
Q01--Q09 remain open authority gates and continue to block only their stated
claims and products. No implementation, validation, or production-readiness
claim follows from this consistency result.
