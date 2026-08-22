# Compact-v2 Native ALIGN Stage 6 Science Projection

- Date: 2026-08-22
- Branch: `codex/converge-apt-align-jinc`
- Implementation commit: `d09234f37b2eda851f35106d994be2620e2468bc`
- Implementation tree: `12137d9701025ac6c3faa6e40eb6f44e1c56ba39`
- Accepted plan commit: `a3f2bf465a26048b24017ebd50876c4a2684b1b8`
- Stage 5 prerequisite/starting commit: `9832cad9b78ac4159eb2b7c4870145b603b59dff`
- Frozen fixture SHA-256: `a4dfdfe4b45638952f57f5f258badfab84f5d6ce1d022abfefc47a9e84091701`

## Result

Stage 6 is implemented and locally validated. The immutable
`NativeScienceProjection` can be created only from the exact measured-scan
mapping handle retained by a completed Stage 5 prepared operation and the
same ledger's exact last committed operation. A later issued operation, a
foreign prepared operation, or any unequal handle is rejected before a
projection exists.

The adapter converts each complete Stage 5 relational segment into a bounded
mapmaking calling surface while preserving, for every detector/sample cell:

- exact network-native sample identity and committed revision;
- the finite post-PTC value and mapped-valid or mapped-invalid state;
- exact common-slot support as relational provenance only;
- the latitude/longitude pair evaluated from that cell's network-native
  telescope and pointing-offset carrier;
- exact signed-int64 output UID, array, network, and optional typed APT flag;
- detector offsets; and
- the map index supplied by the current map-grouping authority.

An absent detector sample is not materialized. A mapped-invalid sample retains
its identity, value, revision, and pointing, but its flag is set and it cannot
contribute to source masking or map accumulation. The projection rejects an
unknown or unresolved automatic grouping; only an already resolved current
map grouping can cross this boundary.

The projection's compatibility preflight runs before either native mapmaking
entry point can mutate a map. It requires exact sample values and flags,
matrix shapes, pixel axes, resolved grouping, map-index inventory, typed
detector identity, APT flags, and bitwise-equal detector offsets. This prevents
a caller from mixing the immutable snapshot with stale, foreign, incomplete,
unequal, nonfinite, or synthetic rectangular input.

The existing naive and JINC public paths remain source-compatible wrappers
around their established implementations. New explicitly opt-in native entry
points perform the projection preflight and select the snapshot's detector
pointing vectors. The established contribution, weighting, kernel, variance,
coverage, support, and accumulation code is otherwise shared. Sequential and
parallel JINC both use the same bridge, and the existing JINC unique-owner
preflight still runs before its parallel worker or destination mutation.

No ordinary runtime caller has been redirected to these entry points.

## Focused contract coverage

The six Stage 6 cases prove:

1. the frozen native-gap fixture produces four admitted relational rows, no
   synthetic row for the missing network-7 sample, distinct native pointing
   for unequal network times, and one cell identity/pointing source shared by
   validity and source-mask decisions;
2. native naive and JINC results are bit-exact with the existing rectangular
   implementations when all networks have identical times, including all map
   signal, weight, kernel, coverage, JINC grid-weight, denominator-support,
   and contributor-count matrices;
3. mapped-invalid cells contribute to neither naive nor JINC output;
4. uncommitted, stale, foreign, incomplete, duplicate, nonfinite, unequal,
   synthetic, unknown-grouping, and unresolved-grouping candidates fail
   before map mutation, while duplicate JINC ownership remains rejected by
   the established ownership preflight;
5. permutation of the typed detector request reconstructs exactly the same
   presentation-ranked snapshot; and
6. parallel JINC produces the exact same checksum at OpenMP thread counts 1,
   2, 4, and 8.

The reviewed exact accumulation checksums are:

- naive: `8052882556844240840`;
- JINC: `4269599267376700904`.

The public projection header also compiles in isolation. The existing science
map, current JINC contract, and JINC ownership suites pass unchanged.

## Validation

The local build uses AppleClang 21.0.0, Release mode, OpenMP 5.1, and the
accepted disconnected dependency-source set.

| Gate | Result |
| --- | --- |
| Stage 6 focused cases | 6/6 passed; JINC exact at OpenMP thread counts 1, 2, 4, and 8 |
| Complete SCI-ALIGN executable | 49/49 passed |
| Public-header isolation | passed |
| Existing science-map suite | 31/31 passed unchanged |
| Existing current-JINC suite | 22/22 passed unchanged |
| Existing JINC ownership suite | 6/6 passed unchanged |
| Complete CTest | 770/770 runnable passed; one established disabled test not run |
| Baseline-tool unit suite | 203/203 passed |
| Required config preflight | 127/127 unit tests; four mode kits; 8/8 compact cases; zero skips/gaps; all audits passed |
| Validation ledger | valid, 60 records |
| Intended-science-change ledger | valid, 3 changes and 5 integration commits |
| Session-exit audit | 728 dependencies; zero library/CLI exits; zero growth |
| Frozen fixture identity | SHA-256 unchanged at `a4dfdfe4b45638952f57f5f258badfab84f5d6ce1d022abfefc47a9e84091701` |
| CLI implementation boundary | `v4.0.0-3676-gd09234f37`; binary SHA-256 `27dae67383ebd2f48c058f0d5db1bb29b125ef7c9f5db0b4c0ac5e04b9e3f0a7` |
| Diff and log hygiene | `git diff --check` passed; zero unexpected error-level messages |

The complete CTest command discovered 771 tests. The established disabled
`MapFitterLifecycle.ExactProductSequence` test did not run; every one of the
770 runnable tests passed. The locally generated implementation-boundary
version also reports the pre-existing shared dependency overlay as
`kids 04088da-dirty`; the Citlali source worktree itself was clean at the
implementation commit, and the exact binary digest above closes the local
identity.

## Stop boundary

Stage 6 stops at the accepted boundary. It does not claim product lineage,
publish a product, change an established naive or JINC numerical algorithm,
add public `Engine` state, or activate an ordinary Science, Pointing, Beammap,
OOF, or other runtime route. The detector/automatic Beammap raw-APT producer
lane and the existing non-detector Beammap calibration-table lane are
unchanged and cannot acquire matched-consumer lineage through this API.

No Unity run is required to close this bounded, opt-in local stage. Stage 7
may begin only as a separate commit. It must add compact observation/scan/
product lineage, lifecycle and required-publication failure coverage, and
explicit asymmetric mode routing. Before Stage 7 can be accepted, the owner
must complete the five-part Unity campaign in the accepted reconstruction
plan, including native-gap Science or Pointing, identical-time/no-gap,
same-scan naive/JINC, detector/automatic Beammap producer, and non-detector
Beammap calibration-lane cases with exact source, binary, config, input, log,
and retained-product evidence.
