# WP-5 VAL Scientific-Owner Decision Packet

Opened: `2026-08-24`

Status: `WP5-OWNER-D001--D002` approved; remaining profile decisions await
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
