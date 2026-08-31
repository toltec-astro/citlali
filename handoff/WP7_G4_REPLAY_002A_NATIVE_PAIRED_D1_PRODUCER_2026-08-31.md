# WP7-REPLAY-002A Native Paired D1 Producer Contract

Date: 2026-08-31

Status: **implementation candidate built and validated; awaiting owner review;
not integrated**

## Bounded work order

After the failed `WP7-REPLAY-002` D2 measurement-seam source review, the
project owner approved the first recorded architecture option: establish a
bounded canonical network-native paired D1 producer contract before any
repaired D2 observer is designed. `WP7-REPLAY-002A` authorizes only the
in-memory paired x/r product contract, exact native occurrence and detector
identity, coordinate-local availability/validity/cause state, native-run
identity, bounded logical-memory evidence, and focused tests.

This unit does not authorize a producer adapter, application-route wiring,
persistent TOD, a common analysis grid, RTC/PTC integration, prefilter or
residual planes, source masks, filter design, factor selection, sample removal,
downsampling, or production activation.

## Exact identities

- Canonical parent: `4dc7844e59e03cf2d18a9262fe5b75d3ff078681`
- Candidate branch locator: `codex/wp7-g4-replay-002a`
- Exact implementation commit:
  `d7d19bc90d7c994fa767ec2a9fd35e4d8599f032`
- Exact implementation tree:
  `af8d4d7c6e8f855845590e63d59ae4a3d43d00f5`
- Scientific source packet:
  `170ecea9de1ee810da7d7e45a489a4545ccd623d`
- Scientific closure:
  `20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa`
- Exact interface artifact SHA-256:
  `f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969`

The implementation commit changes exactly four paths:

| Path | Disposition | SHA-256 |
| --- | --- | --- |
| `include/citlali/core/pipeline/timestream_native_paired_readout.h` | Added canonical public D1 product contract | `920e492a4e0bbbebe4f3336941ced62452fe9301bd2583525eec3e2e54cfe125` |
| `tests/timestream_native_paired_readout_header.cpp` | Added isolated public-header compilation unit | `0c9aee050edeb74d2f92a14c293fd782fda01697a31a0c2c4494c23358081e45` |
| `tests/test_timestream_native_paired_readout.cpp` | Added six focused product-contract tests | `704e3b4551e46699240cf9a13d8e7cb2e18936eeb7328859bb1ba0017db23843` |
| `tests/CMakeLists.txt` | Added the focused target, header gate, discovery, and `check` dependency | `dcd2da14ae077ad716d373abb6982625b87898afe713ee6bf8c5f4933c94adcd` |

The later documentation-only control record does not alter this implementation
identity.

## Source-review disposition

The captured divergent D2 prototype remains preserved at exact commit
`916fa07600cf6c5e9ea7317a396fdce160a6c419`; its source-review record is exact
commit `34f609e4b1dc9a04f8157063c7a1662b707d96a7`. That review rejected canonical
reconstruction because the prototype depended on a noncanonical
`PairedReadout`, did not build its focused test, and lacked required detector,
route, evidence-scope, run, and residual-invalidity bindings.

No prototype source blob was imported. This candidate was written on the
canonical parent against canonical native-timing and identity authority. The
captured prototype remains design evidence only and supplies no application
authority.

## Implemented contract

The candidate establishes:

- exact producer-interface identity
  `TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1`;
- ordered, row-major paired x/r matrices on one exact network-native occurrence
  axis per network, without `NativeAlignmentPlan` or a common analysis grid;
- direct reference to canonical `NativeNetworkAlignment`, including exact
  native identity, packet counter, and contiguous physical-run partition;
- exact parent-readout and paired-x/r occurrence keys plus half-open
  integration support;
- fixed-width coordinate-local state preserving payload availability, producer
  validity, acquisition support, finiteness, and typed causes, with pair state
  derived rather than stored as a third dense cell-state plane;
- complete x/r coordinate and mapping-authority references for meaning,
  unit/scale, sign, reference, normalization, metric, validity domain,
  uncertainty, Tune/readout/input-IQ/transform provenance, native time and
  cadence, timing validity and uncertainty, parent/pair records, runtime
  binding, compatibility, and failure semantics;
- exact detector occurrence, detector-association, and tone/channel identity,
  while explicitly treating network-local storage column as nonidentity;
- observation ownership of the sorted exact participant-network set and
  fail-closed rejection of ambiguous cross-network detector occurrence
  identities;
- fail-closed identity, shape, and finiteness admission;
- logical memory evidence for two double payloads plus two fixed-width
  coordinate states per cell, with axes/text accounted separately and the
  referenced native axis not double-counted; and
- move-only, immutable-facing network and observation products.

It intentionally establishes a contract product, not the producer adapter or
an executable route that populates the product.

## Validation

Validation against exact implementation commit `d7d19bc90...` passed:

- isolated C++23 public-header syntax compilation;
- 6/6 focused native-paired-readout contract tests;
- 839 CTests discovered, with all 838 runnable tests passing and only the
  established disabled `MapFitterLifecycle.ExactProductSequence` test not run;
- `citlali_cli` build and exact Git identity
  `v2-ngc4449-memory-repair-187df04b-9-gd7d19bc90`;
- 130 configuration unit tests, all four TolTECA mode kits, 8/8 compact
  compatibility cases, 100% compact-surface coverage, and every authority
  audit;
- all 207 baseline-tool tests;
- all 62 build-tool tests;
- all 26 WP-7 tool tests;
- validation ledger: 60 records valid;
- science-change ledger: 3 changes and 5 integration commits valid; and
- exact parent, tree, four-path diff, path hashes, clean index/worktree, and
  `git diff --check` identity gates.

The local build reused already-present fallback dependency source snapshots
after a fresh configure could not reach GitHub. Its build and CTest results are
supplemental local compilation/regression evidence, not a reproduction of the
accepted Spack-backed V2 environment. No affected-mode reduction is triggered:
the unit activates no route, configuration, numerical operation, or product
publication.

## Stop boundary

The candidate is not owner-accepted, integrated, pushed, or production
authorized. Work stops at this exact implementation identity. Owner review is
required to accept, reject, or request repair of `d7d19bc90...`; no producer
adapter, D2 observer, RTC/PTC integration, filter/downsampling work, or next G4
increment may begin from this record.
