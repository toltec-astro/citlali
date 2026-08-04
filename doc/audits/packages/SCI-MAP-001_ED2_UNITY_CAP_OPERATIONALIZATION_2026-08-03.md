# SCI-MAP-001 ED2 Unity-cap operationalization — 2026-08-03

Status: owner approved bounded operational clarification; no Unity launch,
external evidence, or scientific disposition is authorized.

Decision ID: `MAP-UNITY-ED2-OPS-RESOURCE-002`

Authority: project owner

## Clarification

The ED2 temporary 200-GiB (`214748364800` byte) ceiling applies only to the
declared governed campaign roots on Unity. It is a practical, human-operated
workspace cap, not a cap on the local repository or a request to invent a
source-derived upper bound for every possible NetCDF, filesystem, or scheduler
allocation detail.

The owner accepts that the existing planning estimate is not a component-wise
proof of the eventual full/all-PTC payload. Requiring such a proof from absent
serialization authority is retired as a successor-package blocker. This does
not reinterpret the planning estimate as a scientific result or a guarantee of
the realized storage use.

## Required operational control

The ED2 package must retain the 200-GiB Unity-root cap and provide a
human-managed sequence, with every command separately invoked by the owner:

1. a no-submit Unity preflight that records the applicable governed roots,
   current logical and allocated use, available filesystem capacity, and the
   planning estimate;
2. owner review before staging, capture, or submission;
3. an after-stage record of the same measurements; and
4. an owner stop before the next material stage if observed use approaches or
   exceeds the cap, or if available capacity is inadequate.

The runbook must make no automatic submission, continuation, deletion, cleanup,
or cache reuse decision. It must continue to place campaign records and return
products under the governed roots and distinguish local package preparation from
Unity workspace use.

## Scope

This amends only `MAP-UNITY-ED2` operational resource handling. The
fixed observations, arrays, seven cases, full/all-PTC retention route,
candidate identity, numerical gates, scientific claim, CAP-transfer boundary,
and all repair/re-audit/production restrictions remain unchanged. The next
permitted step is to complete and locally verify the revised ED2 package, then
return it for coordinator review before any human-run Unity action.
