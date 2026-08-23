# Timestream Exposure-Lineage Boundary

Boundary identity: `TIMESTREAM_EXPOSURE_LINEAGE_BOUNDARY v0.1/r0.1`

Status: owner-decision-complete candidate; exact artifact approval pending

Prepared: `2026-08-23`

Scientific owner: Grant Wilson

Boundary owners: SCI-ALIGN, SCI-RTC, SCI-CAL, and SCI-PTC scientific owners

## Purpose And Authority

This package-neutral boundary preserves exact physical-acquisition and
valid-original exposure truth from original sample occurrences through the
processed-timestream chain. It supplies candidate authority for the
timestream facet of `F-019/XOD-018` and post-audit check `TS-CLAR-002`. It does
not define a MAP, projected, coadded, retained, or generic usable exposure.

It composes, without superseding:

- SCI-ALIGN v0.1/r0.3 Equation 18 and its origin, validity, source-relation,
  interval, and exposure requirements;
- SCI-RTC v0.1/r0.12 representative-occurrence, transitive-support, response,
  replacement, synthesis, and consumer-handoff authority;
- the current SCI-CAL v0.1 signal/identity/lineage and once-only composition
  authority, pending CAL's separate final freeze; and
- SCI-PTC v0.1/r0.5 immutable-parent, support, transformed-signal, fixed-state
  response, cause-preservation, and output-identity authority.

## Immutable Original-Occurrence Facts

For original detector occurrence `d` associated by ALIGN with stable slot `s`
and nominal cell `I_s`, ALIGN owns

```text
e_acq[s,d] = physical native integration support attributable to that
             original occurrence within I_s,

e_vo[s,d]  = e_acq[s,d] when the original payload and required ALIGN-local
             mapping facts are valid, and zero otherwise.
```

An original-invalid occurrence may therefore have `e_acq > 0` and `e_vo = 0`.
Synthesized, missing, surrogate, or unoccupied support adds zero acquired
exposure. These are facts about stable original occurrence identity, not
about output array cells.

Original acquisition/exposure status remains attached to the original sample
occurrences and is never recreated or increased by RTC, CAL, or PTC.

## Required Lineage Through The Timestream Chain

Every downstream sample or product binds a recoverable directed parent
relation sufficient to determine which original occurrences contribute to
it:

```text
original acquisition occurrence
  -> ALIGN occurrence/source relation
  -> RTC output support and response
  -> CAL output identity and response
  -> PTC transformed-output support and response.
```

The relation preserves, for every contributing original occurrence:

- stable original occurrence and ALIGN slot identity;
- `e_acq` and `e_vo` unchanged;
- original, invalid-original, synthesized, missing, surrogate, guarded, or
  replaced cause state as applicable;
- exact parent/product/plan/generation relation at each stage;
- numerical support and response authority, including coefficient sign where
  material;
- selection, mask, boundary, replacement, and state-transition causes;
- response availability and uncertainty availability; and
- compatibility, reconstruction, and provenance state.

The support/response representation may be compact, sparse, factored, or
generative. Dense per-output serialization is not required. It must recover
exact contributing original-occurrence membership and, whenever response is
claimed available, the realized response without undocumented defaults. Typed
response unavailability remains explicit and is never replaced by a
filter-width or cadence approximation.

## Stage Rules

### SCI-RTC

RTC preserves the selected representative occurrence and the complete
transitive support through replacement, filters, state, edges, and
phase-zero selection. Synthesis, donor replacement, filtering, decimation,
finite output, or a longer support interval creates no new acquisition.
Reused donors and overlapping filter support remain links to the same original
occurrences rather than independent exposure.

### SCI-CAL

CAL changes the admitted signal convention and propagates its response and
lineage. Multiplication by calibration or atmosphere factors does not create,
duplicate, increase, or reclassify original exposure. CAL classification and
validity facts remain separate from `e_acq` and `e_vo`.

### SCI-PTC

PTC transforms the calibrated detector timestream through its exact frozen
operator and preserves parent/support/response lineage. Basis-fit,
loading-fit, application, output-retention, QC, and response permissions are
distinct scientific-use propositions. None rewrites whether an original
occurrence was acquired or valid-original. A projected, reconstructed, or
finite transformed value is not an independent new acquisition.

## No Output-Cadence Or Kernel Exposure

RTC output cadence, selected-point spacing, filter width, support duration,
kernel coefficient sum, hit count, finite payload count, PTC coefficient, and
statistical weight are not physical exposure. Overlapping downstream samples
may depend on the same original occurrence, so summing a nominal exposure per
downstream sample can double count acquisition.

Downstream exposure accounting operates on stable original-occurrence
identities and their immutable facts, not on RTC output cadence or filter
width. Any aggregation first resolves its exact original-occurrence
population and deduplicates according to an owner-approved use policy.

The RTC representative occurrence remains an identity and coordinate anchor.
It is not a newly owned representative-exposure scalar and does not replace
the full contributing lineage.

## No Generic Usable Exposure

WP-2 preserves whether each original occurrence was physically acquired and
valid-original, together with enough lineage to determine which downstream
quantities depend upon it. It does not define a generic usable exposure. Any
later exposure quantity qualified by a scientific use is owned and defined by
that use.

A later derived exposure is authorized only when it has:

- a distinct stable name and scientific-use owner;
- exact original-occurrence population and parent relation;
- inclusion, exclusion, overlap, and deduplication rules;
- exact formula, units, support, and missing/conflict behavior;
- compatibility, lifecycle, and provenance; and
- continued preservation of the original `e_acq` and `e_vo` facts.

Such a quantity may not replace, rewrite, or transform away the ALIGN facts.
The SCI-VAL independent-exposure proposition and every retained-use exposure
remain later profile work.

## Failure And Compatibility

If a required original-occurrence identity, `e_acq`/`e_vo` state, or lineage
relation is missing, no exposure value may be reconstructed from duration,
support, weight, hits, cadence, or neighboring products. The affected
processed-timestream exposure-lineage role is unavailable with exact cause;
an exposure-qualified product cannot be published as complete.

Compatibility requires this exact boundary identity, stable original-
occurrence semantics, immutable ALIGN exposure meanings, exact downstream
parent/support relations, no-new-exposure stage rules, and preservation of
typed unavailable and provenance states. A successor shall name this revision
and map every changed role.

This boundary does not assess implementation conformity, validate response
recovery, establish a generic exposure estimator, or authorize any MAP-facing
exposure role.
