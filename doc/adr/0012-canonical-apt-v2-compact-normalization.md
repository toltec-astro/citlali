# ADR 0012: Compact normalized canonical APT v2

Status: accepted owner-directed repair; Citlali implementation verified
locally; cross-repository equivalence and activation gates pending

## Context

The valid v1 matched product for ObsNum 148669 against Beammap 148670 is
249,525,124 bytes. Its 5,234-row scientific table is 3,541,490 bytes, while
261,700 detector-by-field transformation records consume 233,948,454 bytes.
Those records repeat ordinary field rules, parents, operations, and authority
objects. The cost is a contract defect, not a compression or parser-tuning
problem.

V1 also repaired row-position joins only after the legacy matcher had already
selected rows by ordinal. A successor must move the observation matcher to its
scientific owner and persist only occurrence-scoped endpoints.

## Decision

Canonical APT v2 is a compact, relocatable, content-addressed bundle of strict
ECSV tables with one root ECSV manifest and one adjacent completion receipt.
The logical representation is normalized to one APT row per target, one
relation/disposition row per target, one rule per field, one record per source,
and only genuine exception records. It must scale as O(detectors + fields +
exceptions), never detector×field.

Citlali owns the contract, ECSV encodings, identities, guardian, issuance, and
receipt-last transaction. TolAPT owns the preserved observation tone-matching
science and returns occurrence-scoped relations. TolProj orchestrates those
public interfaces and never writes canonical APT bytes. Consumers admit only
receipt-complete v2 products through the Citlali guardian.

New v1 issuance is disabled. V1 is explicit migration/comparison input only,
and migrated v2 remains marked migration-only. No automatic legacy or bare
ECSV fallback is permitted.

The exact physical and scientific contract is
[`../CANONICAL_APT_V2.md`](../CANONICAL_APT_V2.md).

## Consequences

- The primary scientific component remains an ordinary ECSV table.
- A portable matched bundle includes a flattened verified baseline snapshot;
  relation and copied values can be checked after relocation.
- Components are named by transport SHA and enumerated by the root manifest;
  filenames and absolute source paths carry no identity.
- The root receipt is the sole completion transition. A crash can leave an
  incomplete receipt-less directory, never a falsely complete product.
- Exact ties are exposed rather than broken by row order; no near-tie policy is
  introduced.
- The accepted 148669/148670 result must remain scientifically identical and
  below the 20 MiB hard size gate.
- The verbose v1 histories and 238 MiB external artifact remain evidence and
  are not rewritten or committed as fixtures.

## Superseded decisions

ADR 0010 and ADR 0011 remain the historical rationale for canonical baseline
identity and occurrence-scoped matching. Their v1 issuance/admission decisions
are superseded by this ADR. Their artifact-local UID and authority principles
remain in force.
