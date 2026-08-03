# SCI-MAP-002 JINC realized-provenance owner decision — 2026-08-03

Status: owner approved contract clarification; no implementation work
authorized

Package: `SCI-MAP-002`

Decision ID: `SCI-MAP-002-D003-PROVENANCE-001`

Authority: project owner

## Decision

Each coherent JINC observation or declared processing segment requires one
atomic, one-way provenance record with four distinct stages:

1. **Requested:** authored configuration and digest.
2. **Effective:** validation, defaults, and clamps applied to that request.
3. **Resolved:** stable array/name mapping, pixel geometry, input identities,
   and selected processing plan.
4. **Realized:** JINC support/phase/conditioning convention; admission and
   cancellation summary counts; formal, empirical, mask, coverage, and kernel
   product identities; kernel-template and enabled PCA/filter realization
   identity; output file/HDU joins and digests.

The record is compact metadata. It must not contain a per-sample,
per-detector, or per-pixel payload. Each stage has one authority and flows
only forward; no legacy/typed synchronization is implied. Required output
failure prevents a realized-success record from being published.

Future validation must prove exact serialization, one-way lifecycle behavior,
cardinality/array identity, product joins, and failure-path suppression of
realized success. This decision does not authorize a schema implementation,
code change, Unity evidence, repair, re-audit, or production-status change.
