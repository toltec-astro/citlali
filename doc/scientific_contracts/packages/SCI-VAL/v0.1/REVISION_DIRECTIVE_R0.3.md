# SCI-VAL v0.1 — Owner-Approved Surgical Revision Directive r0.3

Status: owner-approved revision authority; implementation-blind document work
only; no scientific freeze, conformity, validation, or readiness decision

Date: `2026-08-21`

## Bound Feedback

This directive binds the complete owner-supplied review artifact with SHA-256:

`9e04c73f8cad5731536720e741d78c53541fe8a378490e16d9aefda9a9c56635`

The digest identifies the exact 279-line review supplied for the surgical
r0.3 revision. The review approves the r0.2 Core/Registry architecture and
scientist-facing narrative while withholding a scientific-freeze decision.

## Owner Decisions

| ID | Owner-approved decision |
| --- | --- |
| `VAL-R03-D001` | The mandatory canonical registry identity is `SCI-VAL:independent_exposure@1`. The former draft key is not a registered identity or compatibility alias. The namespace identifies the registering package and does not confer policy ownership; actual owner and source remain immutable registry fields. |
| `VAL-R03-D002` | Narrative prose names the exact scientific owner recorded in the immutable registry binding rather than hard-coding a person. The registry record retains the exact owner and source metadata. |
| `VAL-R03-D003` | Every noncanonical profile in a worked example is explicitly hypothetical. A hypothetical result is conditional: it would be eligible only if that exact profile were registered, applicable, and satisfied. A hypothetical profile cannot produce an actual decision. |
| `VAL-R03-D004` | Every aggregate proposition has its own registered profile identity, scientific owner, version, applicability domain, operator, threshold, propagation authority, and exact compatible atomic-source-profile binding. It cannot masquerade as another atomic instance. Base atomic inputs remain homogeneous; no heterogeneous transformation is supplied by v0.1. |
| `VAL-R03-D005` | A conflict in identity, parentage, profile authority, applicability, or another structural-gate fact gives `applicability_unknown` and `decision_unavailable`. After the domain is established, a known decisive exclusion yields `ineligible` despite an unrelated unknown or conflicting non-gating fact. With no decisive exclusion, a required non-gating conflict gives `decision_unavailable`. A conflict about exception applicability cannot neutralize the underlying restriction. All conflict evidence is preserved. |
| `VAL-R03-D006` | Response and uncertainty roles form the exact closed set `structural_gate`, `required_permission`, `decisive_exclusion`, and `advisory`. The named-use owner supplies the role. VAL applies its deterministic outcome and never supplies or changes the role. |
| `VAL-R03-D007` | The package title remains stable. Both document covers add a subtitle clarifying that VAL governs producer facts and use-specific eligibility, not final map validity. |
| `VAL-R03-D008` | The adjacent-source table in each view is labeled an r0.3 registry snapshot. `SOURCE_BINDING_REGISTER.md` is the continuing source-binding authority, so a source update does not by itself require rewriting VAL Core narrative. |
| `VAL-R03-D009` | `SCI-VAL-OWNER-QB001--QB006` are dispositioned at the correct layer: QB001 science resolved with serialization engineering-deferred; QB002 science resolved by the conflict rule; QB003 science resolved with exact metadata serialization engineering-deferred; QB004 science resolved by the four roles; QB005 science resolved because changed exception resolution changes lineage and decision identity; QB006 general conservative science resolved while sufficient-summary details remain profile-local. No general SCI-VAL scientific question remains open. |
| `VAL-R03-D010` | The engineering companion receives a complete mechanical consistency review for Core/Registry authority, canonical profile binding, invalid-profile handling, four axes, conflict precedence, response/uncertainty roles, aggregate profile identity, homogeneity and generation boundaries, package-qualified names, exact source replay, and review/exception provenance. Existing `SCI-VAL-REQ-001--049` and `SCI-VAL-PRED-001--024` identities are preserved; wording may be corrected and new IDs are appended only if a genuinely independent obligation is required. |

## Claim Boundary

This directive authorizes a document revision only. It selects no flag or
schema representation, PTC or MAP profile, numerical policy, threshold,
implementation, evidence result, scientific freeze, or readiness state.
