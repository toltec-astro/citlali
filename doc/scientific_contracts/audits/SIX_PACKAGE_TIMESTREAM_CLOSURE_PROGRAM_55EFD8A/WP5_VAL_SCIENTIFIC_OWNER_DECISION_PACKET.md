# WP-5 VAL Scientific-Owner Decision Packet

Opened: `2026-08-24`

Status: `WP5-OWNER-D001--D003` approved; remaining profile decisions await
review one at a time

Scope: non-MAP processed-timestream source bindings and VAL profiles. VAL Core
is not reopened. MAP and coadd profiles remain deferred.

## WP5-OWNER-D001 — Current Non-MAP Source Bindings

Question:

> Should VAL's continuing source-binding register replace its stale
> adjacent-package rows with the exact frozen timestream authorities, while
> leaving VAL Core unchanged and MAP explicitly deferred?

Owner response:

> approved

Disposition: **approved**.

Consequences:

1. The continuing register now binds frozen ALIGN r0.3, AST r0.3, RTC r0.12,
   CAL r0.5-r0.4, and PTC r0.5, together with the exact approved Tune/readout
   interface and the approved WP-2--WP-4 manifests.
2. ALIGN/RTC direct representative-origin semantics remain compatible with
   `SCI-VAL:independent_exposure@1`; completion of that profile's registry row
   remains a separate decision.
3. AST coordinate validity remains distinct from signal validity and
   independent exposure.
4. CAL classification remains a producer fact whose consequence is owned by
   each named scientific use.
5. Source binding alone registers no PTC policy.
6. SCI-MAP remains explicitly deferred and unbound; no MAP profile becomes
   evaluable.
7. Historical Core-view source tables remain snapshots. The continuing
   register changes no prior VAL evaluation identity and requires no Core
   rewrite.

## WP5-OWNER-D002 — Complete The Atomic Independent-Exposure Profile

Question:

> Should `SCI-VAL:independent_exposure@1` be explicitly atomic-only, with
> aggregation and reverse propagation marked `not_applicable` under this
> identity, while any future detector, scan, or observation aggregate requires
> a separate complete owner-bound profile and creates a new derived lifecycle
> generation?

Owner response:

> approved

Disposition: **approved**.

Consequences:

1. `SCI-VAL:independent_exposure@1` is complete as an atomic-only profile
   bound to the current source register, frozen ALIGN r0.3, frozen RTC r0.12,
   and the approved WP-2 exposure-lineage boundary.
2. Aggregation and reverse propagation are `not_applicable` under this exact
   identity. No aggregate is registered by this decision.
3. A future aggregate must have its own immutable owner-bound registry record,
   population and support, operator and denominator, missing behavior,
   threshold if any, uncertainty and failure rules, and propagation authority.
4. Any future authorized propagation creates a new derived proposition and
   lifecycle generation. It cannot overwrite the atomic decision or recreate,
   increase, or rewrite SCI-ALIGN physical-acquisition and valid-original
   facts.
5. This completion defines no generic usable exposure, detector fraction,
   retained or projected exposure, coadd quantity, numerical threshold, or
   MAP policy.
6. The existing atomic truth rule, missing/conflict behavior, and absence of
   inferred response or uncertainty roles are unchanged. VAL Core is not
   modified.

## WP5-OWNER-D003 — Compact Common PTC Policy Factoring

Question:

> May the seven distinct PTC named-use profiles reference one immutable common
> restriction fragment, provided the fragment itself grants no permission and
> every use remains a complete separately registered proposition?

Owner response: **approved with binding interpretation**.

Disposition:

1. The common restriction fragment is compact, PTC-owned, immutable, and
   versioned. It contains only restrictions proved to apply without scientific
   variation to all seven named PTC uses.
2. The common fragment is not a VAL profile, is not independently evaluable,
   produces no eligibility result, and grants no permission.
3. Every PTC use `U` remains a complete, separately registered proposition:

   \[
   \mathcal R_U
   =
   \mathcal R_{\rm common}
   \cup
   \Delta\mathcal R_U.
   \]

   Each profile retains its own named action, use-specific restrictions,
   applicability, and unknown or fail-closed behavior.
4. Permission never transfers between named uses:

   \[
   E_U \not\Rightarrow E_V,
   \qquad U\ne V.
   \]

5. A restriction enters `\mathcal R_{\rm common}` only when it is genuinely
   required by all seven uses. Any scientific variation keeps the restriction
   in `\Delta\mathcal R_U`.
6. CAL's `engineering-only` classification remains a preserved producer fact.
   Whether it excludes a PTC use belongs to that named PTC policy; it is not a
   universal CAL or VAL veto on PTC mathematics.
7. Every registered PTC profile references the common fragment by exact
   identity and digest. Changing the fragment creates a new version and cannot
   silently change an existing profile or evaluation identity.
8. This decision creates no runtime common-policy object, inheritance
   mechanism, serialization requirement, sidecar, duplicated provenance
   payload, or separate engineering route. A separate route is required only
   if a later genuinely different scientific policy requires one.
9. D003 approves the factoring rule, not the fragment's contents. The common
   fragment remains unbound until the remaining use-by-use decisions establish
   which restrictions, if any, satisfy the universal-use test. No PTC profile
   becomes usable through D003 alone.
