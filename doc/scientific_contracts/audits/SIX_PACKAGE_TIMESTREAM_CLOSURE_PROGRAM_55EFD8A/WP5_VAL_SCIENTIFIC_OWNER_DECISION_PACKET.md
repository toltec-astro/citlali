# WP-5 VAL Scientific-Owner Decision Packet

Opened: `2026-08-24`

Status: `WP5-OWNER-D001` approved; remaining profile decisions await review
one at a time

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
