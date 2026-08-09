# SCI-TEL-INPUT-001 owner decision brief

Date: 2026-08-09

Status: independent audit complete; stopped for coordinator and owner review

## Recommended disposition

Use controlled verdict `amend`.

- Contract: `proposed` - the frozen independent core is internally complete,
  but owner adoption is pending.
- Implementation: `nonconformant`.
- Validation: `bounded_incomplete`.
- Production: `existing_use_only`.

This recommendation is not an approval, repair authorization, integration
decision, or production expansion.

## Why owner action is required

The exact TolTECA producer and Citlali consumer authorities do not implement
the frozen structural ingress contract:

1. Telescope-file association is not unique, immutable, or digest-bound.
2. Recomputed cache reuse is based only on pathname existence; output is
   copied and mutated directly at its final path without atomic completion.
3. No raw/product/cache/completion/provenance or row-lineage identity exists.
4. The producer can add five NetCDF objects outside the frozen three-variable
   allowlist when they are absent.
5. Neither side rejects row count/order, first/last, timestamp, cadence/gap,
   duplicate/non-monotonic, non-finite, Hold-transition, or coherent-pairing
   failures before state mutation and downstream use.
6. Citlali loads telescope variables independently, treats failures as
   optional warnings, and does not perform transactional admission.
7. Recomputed source RA/Dec share internal destinations with native telescope
   RA/Dec; ordered loading can overwrite the recomputed values. This is a
   structural routing finding. Choosing scientific authority or a remedy is a
   Tier-A SCI-AST decision.
8. ALIGN interpolates telescope streams without the required row/time
   admission and numerically interpolates row-valued `Hold`.
9. ENG-STATE transaction/provenance facts and SCI-VAL eligibility facts are
   absent.

## Bounded positive evidence

The coordinator-approved read-only inspection covered exactly two existing
raw/recomputed pairs: beammap 148670 and pointing 152389.

Within each pair:

- normalized NetCDF headers, dimensions, variable ordering, dtypes, shapes,
  and attributes match;
- `time` cardinality is preserved;
- every non-allowlisted variable has identical raw logical typed bytes;
- exactly `ActParAng`, `SourceRaAct`, and `SourceDecAct` change;
- all three recomputed arrays are finite;
- `TelTime`, telescope `PpsTime`, and `Hold` bytes are preserved; and
- the beammap pair contains 1370 preserved Hold transitions, while the
  pointing pair contains none.

This cannot prove general cache safety, atomicity, retry, provenance,
same-row derivation, rejection of coherent plus/minus-one row corruption, or
physical event semantics. No authorized gap tolerance was supplied, so no gap
count was inferred.

## Decisions requested from the owner

1. Adopt or amend the frozen structural core while preserving its exact frozen
   bytes and history.
2. Decide the exact repository/maintainer ownership of each repair finding.
   Any TolTECA modification requires explicit TolTECA maintainer opt-in.
3. Decide whether to authorize a bounded repair dispatch. No repair has begun.
4. Preserve the Tier-A stop for coordinate authority/alias remedy, ALIGN
   interpolation response, acquisition-event meaning, physical row
   displacement, timing correction, uncertainty, or astrometric response.
5. If a repair is authorized and completed, require a fresh role-separated
   independent re-audit against exact successor commits and the preregistered
   structural falsification cases.

## Mandatory retained restrictions

`physical_event_semantics` remains `unavailable`. Detector and telescope
`PpsTime` remain distinct. Do not claim a sample start/end/centroid, absolute
phase, half- or whole-sample correction, detector-time absolute oracle,
absolute timing, sub-sample astrometric placement, timing-sensitive
source-mask fidelity, or related precision/response result.

The audit performed no application, test, build, configuration, validation,
production, TolTECA, or canonical coordination edit; no Unity access, local
reduction, external contact, delegation, fixture production, repair, re-audit,
downstream launch, push, or merge.
