# Config Source Provenance

This document fixes the durable Phase 2 record of the configuration Citlali
actually receives. It does not duplicate TolTECA's numbered-file merge.

## Ownership Boundary

TolTECA owns discovery and precedence of the reduction directory's `NN*.yaml`
authoring files. Its generated low-level Citlali YAML is the input supplied to
the Citlali CLI. Citlali owns only the order in which files passed on its own
command line are merged; later CLI files override earlier files.

Because the current TolTECA interface does not pass its complete ordered
authoring-source list to Citlali, the manifest states that limitation rather
than reconstructing or guessing it. TolTECA can extend the interface later
without changing the current ownership split.

## Required Record

Successful reductions atomically publish:

- `config_source_manifest.yaml`, schema
  `citlali-config-source-manifest-v1`;
- one collision-safe copy of every CLI input config, in precedence order; and
- `citlali_merged_config.yaml`, the canonical merged snapshot used for the
  run.

The manifest records original path, copied filename, byte size, and SHA-256
for every source, plus byte size and SHA-256 for the merged snapshot. Repeated
basenames receive stable `source_NNN_` prefixes so one input cannot overwrite
another in the reduction directory.

The reduction auditor recomputes every recorded size and digest. Missing,
reordered, overwritten, or modified copied inputs therefore fail a required
provenance check without reading large science products.

## Stop Rule

Do not teach Citlali to scan reduction directories or reproduce TolTECA's
numbered-overlay rules. Full upstream authoring provenance belongs in a future
explicit TolTECA-to-Citlali manifest contract.
