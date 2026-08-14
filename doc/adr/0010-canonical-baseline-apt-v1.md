# ADR 0010: Canonical Citlali Baseline APT v1

- Status: Accepted
- Date: 2026-08-14
- Decision owner: project owner
- Implementation state: bounded candidate; not integrated or activated
- Normative artifact contract: [`../CANONICAL_APT_V1.md`](../CANONICAL_APT_V1.md)

## Context

Citlali's historical Beammap APT output preserved useful detector-resolved
quantities but did not provide one complete typed producer contract for field
authority, raw-channel binding, occurrence identity, canonical content
identity, byte transport, and publication completion. Historical APTs also
cannot prove a persistent measured-detector namespace merely because they
carry a `uid` column.

The frozen APT-E2E-001 audit established the architectural boundary. Citlali
must produce a self-contained canonical baseline artifact from current
authoritative raw/telescope/Beammap state. The row-to-raw-channel relation must
be embedded in that artifact. Content identity must be independent of
presentation order, publication occurrence must remain distinct from content,
and byte integrity must be separately checkable. This binding contract is
separate from CAL physical-science closure and from any downstream admission.

Several current fields also have different authority states. Beammap owns
fitted and derived quantities; a KIDs fit report may contribute its own exact
fit flag; and `fg`, `pg`, `ori`, `loc`, and `responsivity` have unresolved
authority. Omitting those current fields or silently assigning meaning would
both be wrong.

## Decision

Citlali defines artifact schema `citlali-canonical-apt-v1` and Beammap profile
`citlali-beammap-baseline-apt-v1` with these rules:

1. `uid` is a unique, sparse-permitted, nonnegative artifact-local `int64` row
   key. V1 limits it to `2^53 - 1` for exact interchange. It is never a
   persistent detector identity.
2. Every row embeds the exact `(nw, kids_tone)` raw-channel relation. A raw
   manifest declares one unique canonical `toltecN` input and positive channel
   count per network, and rows form a complete manifest bijection.
3. The observation/subobservation/scan tuple and scientific context are
   explicit. The v1 coordinate frame is exactly `altaz`. Tune identity is not
   reconstructed.
4. The field contract fixes five protected structural columns, 27 required
   registered fields, and an allowlist of 20 optional extensions. Every
   declaration states exact type, unit, nullability, authority, authority
   reference, non-finite policy, registry, description, and identity role.
5. `fg`, `pg`, `ori`, and `loc` remain required, nullable, explicitly
   `unavailable`-authority nonidentity fields. Current numeric values remain
   semantic content but are not promoted or reconstructed. `responsivity`
   follows the same unresolved-authority discipline.
6. Optional `kids_flag` is the exact signed `int64` legacy KIDs fit-report
   `flag`, copied under declared authority `kids:fit-report-v1`. It is distinct
   from Beammap `flag` and `flag2`, allows nonbinary integral values, and is
   omitted when no fit report exists, including simulation.
7. Semantic identity uses labelled type/length framing and SHA-256 over
   canonical field declarations, observation/context, the raw inventory, and
   all values. Fields sort by name, inputs by `(network, interface)`, channels
   by `(network, channel)`, and rows by `uid`.
8. An opaque issuer-provided occurrence and event reference live in a separate
   envelope. The envelope SHA-256 binds semantic identity to occurrence,
   output role, producer, software revision, configuration reference, and
   event time. Neither opaque reference is derived from content, path, or time.
9. Byte transport has a third SHA-256 scope over exact canonical ECSV bytes,
   bound to the envelope digest and byte count.
10. The producer writes and rereads a staged typed ECSV, recomputes all
    identities, validates an envelope-bound adjacent `.ecsv.sha256` receipt,
    publishes without replacement, revalidates the final artifact, and makes
    the receipt visible last as the completion transition.
11. The standalone artifact contract admits exactly the built-in 27+20
    catalog. A generic caller-supplied strict-extension seam does not allow an
    artifact to authorize its own columns; schema evolution requires a
    separately accepted successor artifact contract.

## Authority And Source Rules

The current raw KIDs inventory supplies network, canonical interface, channel
count, channel relation, and ToneFreq. Current telescope/output observation
state supplies and cross-checks the complete observation tuple and scientific
context. A KIDs fit report corroborates only network and observation identity
and is the declared source of optional `kids_flag`; it cannot supply an
unproven subobservation, scan, tune, or persistent detector identity.

Duplicate KIDs items for one network fail closed in v1 because no accepted
producer fixture demonstrates a legitimate split-network case. Contrary
authoritative evidence is an owner stop and requires a successor decision, not
an implicit merge rule.

## Consequences

- A canonical artifact carries its own exact raw binding and semantic
  contract; no component-private relation sidecar is needed.
- Identical semantic content may be issued as multiple occurrences, while
  byte transport can still detect a different presentation.
- Row reorder does not change semantic or envelope identity, but it may change
  transport identity. UID sparsity and presentation order are independent.
- Validators can reject duplicate/out-of-range UIDs, incomplete or duplicate
  raw relations, undeclared or redefined fields, wrong authority metadata,
  lexical drift, noncanonical ordering, digest mismatch, and incomplete
  publication.
- The adjacent receipt is only an envelope-bound completion marker. It cannot
  replace embedded semantics, raw relation, or content identity, and a
  post-hoc validator cannot prove historical directory-entry publication
  order.
- Producer adaptation must preserve detector set, row order, exact integral
  values, ToneFreq bits, and all current science values. Any scientific drift
  stops the package.

## Non-Goals And Deferred Decisions

This ADR does not:

- invent or authorize a persistent measured-detector identifier;
- reconstruct tune identity or repair historical APTs;
- make historical APTs current production inputs;
- change matching, calibration, fitting, maps, RTC, PTC, flags, detector
  selection, or numerical policy;
- establish the physical scientific authority of unresolved design,
  polarization, responsivity, or calibration fields;
- close CAL or ALIGN work;
- activate a validation profile or downstream reader;
- change TolTECA, TolProj, TolAPT, `toltec_beammap`, or any external service; or
- authorize a custom extension registry without a successor artifact contract.

## Activation And Supersession

The decision is accepted, but the implementation remains a bounded candidate
until its required gates, exact candidate commit, owner-controlled integration,
and independent disposition are recorded. The artifact is not a production
input and has no downstream ingestion authority in this state.

A successor ADR and artifact-contract version are required to change UID
scope/range, admit a persistent detector namespace, support split-network raw
inputs, change the fixed field catalog or canonical encoding, alter occurrence
semantics, or change publication completion. Historical artifacts retain their
own contracts and are not reinterpreted by this ADR.
