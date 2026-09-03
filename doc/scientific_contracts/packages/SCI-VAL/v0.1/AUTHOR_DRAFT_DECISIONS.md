# SCI-VAL v0.1 Stage B Author Draft Decisions

Status: implementation-blind first-pass record plus owner-approved targeted
r0.2 and r0.3 revision dispositions, through 2026-08-21. The original
derivation choices and layered question dispositions are preserved below.
This record contains no
implementation finding, validation, scientific freeze, or production
authority.

## Author firewall and admitted-input verification

The author read only the four admitted repository files.  Before drafting,
their SHA-256 values were verified exactly:

| Admitted file | Verified SHA-256 |
| --- | --- |
| `SCOPE_BRIEF.md` | `98510ed385164ee2f3339284a3b15434da4821b85a43b19de1f9f691186594f9` |
| `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `32dc62160dff5dcb15e4af83d0df3311024494f30de075784603d4b4bfb4a52c` |
| `AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md` | `7296112f48fd1edc8eb4b4527883aad86b3dbade19509ab8268e9c6f8b7e4964` |
| `DECISION_LOG.md` | `29c771980de40d7953faa1325c568492cc0c32ac3976494c11fc65bd5b8dae60` |

No source, implementation, schema, test, audit, validation record, prior-work
file, dossier, historical handoff, full adjacent contract, Unity evidence, or
production-status material informed this draft.

## Owner-approved r0.2 revision disposition

The exact first-round review and subsequent owner direction are recorded in
[`REVISION_DIRECTIVE_R0.2.md`](REVISION_DIRECTIVE_R0.2.md). The revision
preserves the first-pass algebra while making these bounded changes:

- SCI-VAL Core and the owner-bound Profile Registry are separate layers;
  neither authors a scientific-use policy.
- `VAL.core.independent_exposure@1` is the first mandatory canonical profile.
  Direct representative synthesis/replacement is non-exceptionable; an
  attempted override is policy-invalid.
- PTC and MAP questions use package-qualified names. Reserved names without a
  complete owner/source/domain/restriction binding remain unavailable.
- Base aggregation is homogeneous in exact profile/version, lifecycle stage,
  object type, and applicability domain. Reverse propagation creates a new
  generation and cannot rewrite its own denominator.
- Adjacent meanings are bound through
  [`SOURCE_BINDING_REGISTER.md`](SOURCE_BINDING_REGISTER.md); no unavailable
  MAP policy is manufactured.
- The rationale and engineering documents are now separate genres. The
  engineering view retains the complete formal authority.

These changes append normative identifiers; they do not renumber or retire
the original `SCI-VAL-REQ-001--042` or `SCI-VAL-PRED-001--018` identifiers.

## Owner-approved r0.3 revision disposition

The exact second review and owner decisions are bound in
[`REVISION_DIRECTIVE_R0.3.md`](REVISION_DIRECTIVE_R0.3.md). R0.3 renames the
active canonical key to `SCI-VAL:independent_exposure@1` with no alias from the
former draft key; makes all illustrative noncanonical profiles explicitly
hypothetical; requires a distinct registered profile for every aggregate;
closes conflict precedence and exception-conflict behavior; establishes the
four response/uncertainty roles; labels source tables as snapshots of the
continuing register; and dispositions all six general scientific questions.
The existing 49 requirement and 24 prediction identities remain sufficient,
so r0.3 corrects wording without appending or renumbering IDs.

## Draft derivation decisions

| Draft decision | First-pass resolution | Reason |
| --- | --- | --- |
| `SCI-VAL-AUTHOR-DD001` | Model eligibility as a partial proposition: `eligible`, `ineligible`, or `decision_unavailable` only when evaluated; use `∅E` for no proposition under not-requested, known-inapplicable, or non-authoritative artifact states. | Avoids inventing a fourth disposition and avoids aliasing not-requested/inapplicable/failure to decision-unavailable. |
| `SCI-VAL-AUTHOR-DD002` | Use producer-fact knowledge states true, explicit false, unknown, and conflict/out-of-domain, with applicability separate. Structural conflict gives applicability unknown and decision unavailable. After domain establishment, normalize non-gating conflict to unresolved while retaining it; a known decisive exclusion still gives ineligible, and otherwise a required conflict gives decision unavailable. | Makes the open-world negative rule, structural/use-specific distinction, and owner-approved r0.3 conflict precedence explicit. |
| `SCI-VAL-AUTHOR-DD003` | Normalize each applicable restriction to permission `T`, exclusion `F`, or unresolved `U`; compose with false-dominant conjunction. | This is the smallest deterministic algebra satisfying decisive-false, unknown-only, and no-rescue rules. |
| `SCI-VAL-AUTHOR-DD004` | Treat an explicit resolved and permitted same-policy exception as a traceable transformation of only its named exceptionable restriction, composing as permission while preserving cause, restriction, and exception. A registry-declared contract invariant is not exceptionable; unknown or conflicting exception applicability leaves the underlying restriction in force. | Distinguishes an authorized exception from rescue by an unrelated or unresolved permission and incorporates `VAL-R02-D003` plus `VAL-R03-D005`. |
| `SCI-VAL-AUTHOR-DD005` | Treat `review` for possible conservative influence as an action annotation, not an eligibility disposition; absent an independently resolved admission predicate, the disposition is decision-unavailable. | Preserves the approved three-value eligibility domain while retaining the four allowed policy responses to possible influence. |
| `SCI-VAL-AUTHOR-DD006` | Require the use owner to assign each response/uncertainty fact exactly one of `structural_gate`, `required_permission`, `decisive_exclusion`, or `advisory`; VAL applies the deterministic role semantics. | Prevents profile authors from encoding the same conceptual role through incompatible truth-table conventions. |
| `SCI-VAL-AUTHOR-DD007` | Define aggregation under its own registered aggregate profile, which binds the exact homogeneous atomic source profile; require exact profile/version, lifecycle-stage, object-type, and applicability-domain compatibility, permutation invariance, and owner-declared sufficient-summary equivalence for partition interchange. Reverse propagation starts a new generation. | Prevents aggregate/atomic identity aliasing, semantically mixed fractions, hidden order/partition dependence, and circular denominator feedback without claiming arbitrary threshold or quantile associativity. |
| `SCI-VAL-AUTHOR-DD008` | Define stricter-policy monotonicity only for an owner-declared relation on an exact common domain, as inclusion of admitted fact sets. | Avoids inferring policy order from names, versions, thresholds, or apparent predicate count. |

## Owner question dispositions

1. **`SCI-VAL-OWNER-QB001` — science resolved; engineering deferred.** No
   proposition is evaluated for not-requested or known-inapplicable cases;
   `∅E` is the mathematical notation. The exact non-aliasing serialization
   carrier remains engineering-deferred.
2. **`SCI-VAL-OWNER-QB002` — science resolved.** Structural conflict gives
   applicability unknown and decision unavailable. In an established domain,
   decisive exclusion dominates unrelated non-gating unknown/conflict;
   otherwise a required conflict gives decision unavailable.
3. **`SCI-VAL-OWNER-QB003` — science resolved; engineering deferred.** Review
   is action metadata, not an eligibility disposition. Its exact metadata
   serialization remains engineering-deferred.
4. **`SCI-VAL-OWNER-QB004` — science resolved.** The exact owner-supplied role
   set is `structural_gate`, `required_permission`, `decisive_exclusion`, and
   `advisory`.
5. **`SCI-VAL-OWNER-QB005` — science resolved.** A changed exception resolution
   changes resolved-profile lineage and decision identity even if the
   requested profile version string is unchanged.
6. **`SCI-VAL-OWNER-QB006` — general science resolved; profile-local detail.**
   Partition interchange is never universal. Each aggregate profile owns any
   sufficient summary, associative combine rule, and exact equivalence.

No general SCI-VAL scientific question remains open. The deferred items above
are explicitly representation or profile-local work and do not acquire a
scientific answer by inference.
