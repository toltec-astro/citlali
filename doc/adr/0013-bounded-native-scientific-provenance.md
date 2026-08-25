# ADR 0013: Bounded native scientific provenance

Status: accepted and implemented; bounded native-gap replay complete,
production JINC exact-SHA replay pending

## Context

The Stage 7 native Science route keeps detector/sample flags, eligibility,
operation exclusions, revision state, weights, and intermediate arrays while
processing an observation. The original
`citlali-native-cohort-product-provenance-v2` publication contract copied that
runtime ledger into the canonical YAML sidecar. On observation 152390 this
expanded 124 scans, roughly 1,220 rows per scan, and 5,518 detectors into about
835 million revision records. RTC support repeated another detector-scaled
record for every output row. Numerical processing and map publication reached
completion, but the mandatory snapshot copy and YAML expansion were not a
bounded product contract.

Compressing that exhaustive history into NetCDF would change the encoding but
would preserve the mistaken requirement. For fully specified raw inputs, APT,
effective configuration, source/build identity, and execution policy, those
per-sample transitions are deterministic consequences that Citlali can
regenerate.

Runtime scientific state, canonical scientific provenance, and diagnostic
execution tracing have different owners and retention requirements. Treating
them as one product obscures those boundaries and makes ordinary canonical
completion scale as detector count times observation duration.

## Decision

Canonical native provenance records authoritative inputs, rules, causes,
identities, and scientifically meaningful exceptional decisions. It does not
serialize deterministic detector-by-sample consequences merely because the
runtime owns them.

The successor native provenance contract is bounded by natural scientific
scope:

- observation scope records raw-observation and compact-v2 APT identities,
  authoritative input digests, complete effective-config identity,
  Citlali source/build identity, policy/schema versions, dimensions, and each
  APT detector exclusion once per detector;
- scan scope records scan identity and dimensions, compact RTC/PTC execution
  identities, named scan-wide causes, population reconciliation, exceptional
  decisions, and map-product identity;
- detector scope records a detector-wide cause once, not once for every
  sample;
- interval scope records a bounded interval and named cause when the
  authoritative fact is genuinely interval-valued; and
- final publication records required product identities/checksums, completion
  and validation state, and makes no canonical product index visible until
  every required output and the bounded sidecar validate.

Runtime sample masks and revision ledgers remain in memory and retain their
existing fail-closed contracts. They are neither weakened nor inferred from
the sidecar. Canonical digests may bind authoritative inputs, bounded
natural-scope summaries, detector-scope decisions, and final products. They
must not bind an exhaustive detector-by-sample history under another
encoding.

Detailed execution tracing is a separate, opt-in diagnostic artifact. A trace
request must select scans, networks, detectors, or sample intervals and carry
a hard record/byte bound. It is not required for canonical completion, does
not enter the canonical product index, and has no default retention claim.

Sample-level persisted state may be introduced only as a separately named
scientific product, restart checkpoint, or cache with an explicit consumer,
owner, schema, completion rule, and retention policy. No such use is approved
by this decision. In particular, an exhaustive compressed NetCDF lineage
artifact is rejected.

## Required invariants

The successor implementation and its tests must prove that:

1. every exclusion class has a named owner, cause, and natural scope;
2. eligible, APT-excluded, operation-excluded, invalid, zero-weight, and
   contributing populations reconcile without silently excluding or replacing
   valid samples;
3. excluded samples have zero scientific weight and no map contribution;
4. identical frozen inputs, APT, effective configuration, source/build, and
   execution policy reproduce the same scientifically relevant admission and
   output identities under the declared numerical policy;
5. failed validation or publication cannot publish canonical completion or a
   canonical product index;
6. canonical provenance size is independent of detector-sample cardinality
   except for genuine bounded detector, scan, interval, or exception records;
   and
7. config, input, APT, software, policy, product, completion, and validation
   identities are explicit and independently checkable.

## Compatibility and migration

`citlali-native-cohort-product-provenance-v2` remains historical evidence for
the Stage 7 activation candidate. It is not silently reinterpreted. The
bounded contract receives a successor schema version, and validators must
name which schema they admit.

Commit `9c3b71e79` remains an independently reviewable execution repair. Its
typed-null APT handling, duplicate-tone exclusion, finite geometry policy,
and zero-weight behavior remain runtime scientific authority and are not
reverted by this publication change.

Observation 152390 is the first full successor gate: all 124 scans, maps,
diagnostics, bounded sidecar, validation, and product index must complete in a
frozen local replay before one exact-SHA Unity confirmation. Empirical data
pathologies are investigated separately from software completion.

## Consequences

- Canonical completion no longer depends on allocating or serializing an
  observation-wide detector-by-sample ledger.
- The runtime retains exact sample authority and can regenerate deterministic
  intermediate consequences.
- Canonical review becomes centered on authorities, causes, reconciled
  populations, identities, and products rather than implementation history.
- Debug investigations require an explicit bounded trace request.
- A future external need for sample-level state must return as a new product
  decision; it cannot reactivate exhaustive lineage by convenience.
