# SCI-VAL v0.1 — Targeted r0.2 Manager Review

Status: owner-approved targeted revision implemented; scientific-owner review
pending; scientific authority not frozen

Date: `2026-08-20`

## Admitted Authority And Firewall

The four original packet artifacts retain the exact hashes recorded in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md). The targeted revision
is governed by [`REVISION_DIRECTIVE_R0.2.md`](REVISION_DIRECTIVE_R0.2.md), with
SHA-256
`5b8f36288917bb12c342ada192d2dee0b87bb40f8f9868acdcc11eff489d8ef0`
and decisions `VAL-R02-D001--D009`.

No implementation, audit, validation, test, Unity, or production-status
material was admitted as scientific authority. Exact adjacent meanings enter
only through [`SOURCE_BINDING_REGISTER.md`](SOURCE_BINDING_REGISTER.md); an
unavailable MAP-owned policy remains unavailable.

## Content Review

The targeted revision establishes the following architecture without making
VAL a policy author:

- SCI-VAL Core owns shared fact and profile envelopes, four-axis logic,
  deterministic evaluation, cause preservation, immutable lifecycle, and
  aggregation mechanics.
- The Profile Registry binds immutable profiles to their actual scientific
  owner, authoritative source/version or digest, domain, restrictions,
  exception permissions, compatibility, and missing behavior. Registration
  does not transfer or duplicate policy authority.
- `VAL.core.independent_exposure@1` is the first mandatory canonical profile.
  Direct representative synthesis or replacement is a non-exceptionable
  contract invariant for this profile; an attempted override is policy invalid
  and cannot produce eligibility.
- Base-v0.1 aggregation requires an exact homogeneous profile/version,
  lifecycle stage, object type, and applicability domain. Heterogeneous
  aggregation is unavailable absent an explicit owner-approved transformation.
- Reverse propagation creates successor-generation facts and decisions and
  cannot rewrite or feed the denominator decisions from which it was derived.
- `map_upstream_admission` replaces the overloaded former MAP label.
  Package-qualified PTC fit/application/coefficient and MAP profile names are
  reserved but remain unbound until their actual owners register complete
  profiles.

The 8-page rationale is scientist-facing and narrative. Its physical page 2
contains the worked retained/replaced-occurrence table. The 19-page engineering
view retains the complete formal tuples, truth tables, equations, numbered
obligations, falsifiers, replay identity, and conformance-record guidance.

## Normative Stability And Artifact QA

- Preserved requirements: `SCI-VAL-REQ-001--042`
- Appended requirements: `SCI-VAL-REQ-043--049`
- Preserved predictions: `SCI-VAL-PRED-001--018`
- Appended predictions: `SCI-VAL-PRED-019--024`
- Totals: `49` requirements, `24` predictions, `73` exact crosswalk rows
- Scientific rationale: `8` letter pages
- Engineering conformance specification: `19` letter pages
- Durable verifier: original and r0.2 authority hashes, registry/source
  bindings, sequential IDs, exact crosswalk, dual-view split, builds, PDF text,
  and stable PDF checks pass
- Poppler render and inspection: all `27` pages inspected after the final build

The first engineering render exposed one identifier that could divide across a
page boundary. The item macros now keep each normative record as an indivisible
block. The rebuilt rationale and engineering pages show no clipping, overlap,
broken table, bad glyph, or unreadable content.

## Owner Review And Claim Limit

No new owner question arose. The six existing questions
`SCI-VAL-OWNER-QB001--QB006` remain open with their original blocked claims:
partial eligibility, non-gating conflict precedence, review-action semantics,
response/uncertainty roles, exception lineage, and partition equivalence.

This review records document content and artifact QA only. It makes no claim of
implementation conformity, representation fidelity, observational validation,
scientific freeze, production readiness, or adjacent-package readiness.
