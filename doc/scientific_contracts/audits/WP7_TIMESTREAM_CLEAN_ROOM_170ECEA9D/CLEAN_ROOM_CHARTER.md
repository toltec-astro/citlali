# Six-Package Timestream Clean-Room Re-Audit Charter

Source commit: `170ecea9de1ee810da7d7e45a489a4545ccd623d`

Status: sanitized launch authority for an independent implementation-blind
scientific-contract audit

## Endpoint

Audit whether the admitted authorities provide a coherent, source-closed
processed-timestream chain for TolTEC detector data:

```text
native paired readout
  -> ALIGN occurrences
  -> RTC conditioned detector timestreams
  -> CAL ordinary calibrated signal
  -> PTC transformed signal and companions
```

Audit the separately requested RTC-only terminal route that ends after RTC.
Its endpoint is completion of the consumer-neutral logical RTC output stream
over the declared domain plus finalization of observation-level RTC facts.
Incremental production and consumption are normal; “atomic bundle,”
“publish,” and “export” do not require one observation-sized serialized
intermediate. Do not infer external-consumer acceptance or a CAL, PTC, or
mapping product on that route.

The endpoint is a typed logical bundle. A requested route may end in a
well-typed unavailable, rejected, disabled, or failed state. A numerical
signal does not imply response, uncertainty, exposure, or downstream-use
authority.

## Scope

The admitted primary packages are exactly:

- SCI-ALIGN v0.1/r0.3;
- SCI-AST v0.1/r0.3;
- SCI-RTC v0.1/r0.12 with the approved `2026-08-25` explanatory
  owner-identifier source correction;
- SCI-CAL v0.1 science-rationale r0.5 and engineering-conformance r0.4;
- SCI-PTC v0.1/r0.5; and
- SCI-VAL v0.1/r0.3 Core with its continuing source and profile registries.

The admitted composition authorities are the exact ALIGN-to-AST,
RTC-to-AST-grid, detector-geometry/field-rotation, timestream-exposure, native
readout, reference-first handoff, notation-parity, and PTC named-use records
listed in the source manifest and readable-source allowlist. The approved WP-7
authority addendum also admits the exact atmosphere contract, atmosphere node
table, TolTECA-v1 passbands, WVR interpolation and unavailable-opacity rules,
observation-wide opacity classifier, and RTC logical-stream clarification.

Continuous AST coordinate roles on the ALIGN and RTC grids are in scope.
Pixel deposition, map projection, map admission, map weighting, map exposure,
map response, map validity, coaddition, reprojection, and mosaicking are out of
scope. SCI-MAP is not an admitted authority. An outward MAP reference in an
admitted package must be recorded only as deferred or unavailable and must not
be followed.

## Readiness Names

Report each level separately. Higher claim tiers must not be inferred from a
lower tier.

| Level | Question |
| --- | --- |
| `TS-A` | Is the admitted authority graph acyclic and are signal, coordinate, policy, response, uncertainty, exposure, and lifecycle meanings kept distinct without contradiction? |
| `TS-S` | Is one exact ordinary acquisition-to-PTC processed-signal route source-closed, including identity, units, order, estimator/application mathematics, use-specific admission, lifecycle, failure, and truthful unavailable states? |
| `TS-C` | Does `TS-S` also carry complete ALIGN-grid and RTC-grid coordinate roles plus original-occurrence acquisition/exposure accounting through the processed timestream? |
| `TS-R` | Does `TS-C` also provide one exact named source/beam-to-PTC conditional response role, with frozen-state assumptions, support, null space, and unavailable-state semantics explicit? |
| `TS-U` | Does `TS-C` also provide one exact named conditional uncertainty or covariance-lineage role with axes, units, support, correlations, approximations, and omissions explicit? |
| `TS-T` | Does one specifically named stronger scientific claim have every response, uncertainty, nuisance, cross-covariance, applicability, and owner authority it requires? |

`TS-R` and `TS-U` independently extend `TS-C`; neither implies the other.
Failure to reach a stronger tier does not erase a valid lower-tier result.
Observation-instance realization is a separate axis and is not assessed here.

## Clean-Room Firewall

During independent extraction, the auditor may inspect only this charter, the
source manifest, the readable-source allowlist, the sanitized composition
notes, and the exact allowlisted source-commit objects.

The auditor must not inspect:

- Citlali implementation, configuration, schemas, tests, generated products,
  validation results, reductions, performance evidence, or production state;
- previous audits, repair directives, closure trackers, prior scenario sets,
  historical implementation behavior, chat history, web sources, or
  undocumented practice; or
- any SCI-MAP source.

No implementation default, familiar convention, or current behavior may fill
a missing contract fact. Missing, conflicting, inapplicable, unavailable, and
not-requested states must remain distinct.

The complete approved native-interface authority set is admitted despite
incidental administrative references in its exact decision and approval
records. Those references carry no scientific meaning and must not be used to
assign or map independent findings before the independent outputs are locked.

## Independent Work Order

Before receiving any comparison material, the auditor shall derive and lock:

1. an admitted-source inventory and authority graph;
2. an interface and ownership matrix;
3. a distinction/invariant ledger;
4. a new scenario suite covering ordinary, RTC-only, unavailable, conflict,
   response, uncertainty, exposure, lifecycle, and provenance states;
5. findings stated without legacy identifiers; and
6. separate `TS-A`, `TS-S`, `TS-C`, `TS-R`, `TS-U`, and `TS-T` results.

The independent report, scenario report, and their combined SHA-256 manifest
must be locked before any regression comparison begins. Later comparison may
map independently derived results to earlier records, but it may not alter the
locked independent extraction.

## Claim Boundary

This audit may establish contract architecture and source closure only. It
shall make no claim of implementation conformity, representation fidelity,
observational validation, achieved performance, deployment, production
readiness, mapping readiness, Stokes reconstruction, or downstream consumer
acceptance.
