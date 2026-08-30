# WP-7 AST Scan-Motion Representative Acceptance Package

## Disposition

The bounded `wp7-ast-scan-motion-v1` implementation passes its representative
observation-152390 execution gate at exact implementation revision
`f2bf3f1e00226d6e7f63e99d4da61d37ea4ddf3d`. The first independent exact-SHA
review of package snapshot `b0e5dde2ac532a7a36e141bf22c7560e0fbbc8a1`
returned `HOLD` solely because the validator did not enforce the documented
record digest. This bounded repair closes that finding; fresh exact-SHA
re-review remains pending. The package therefore closes the representative-
execution prerequisite but does not yet authorize AST as an RTC planning input
or authorize any nonidentity RTC method.

The retained
[v1 evidence record](WP7_AST_SCAN_MOTION_ACCEPTANCE_152390_V1_2026-08-30.json)
has SHA-256
`9b1652fc158de6aa17732213d25333316b95045e96d4d2fa97737e6edf9015fe`.
The exact runner executable has SHA-256
`376879038ef10f950f997eb82224ded119767bbd7efc94b0c40bf071e6fa279c`.

## Bounded claim

For the exact approved real TolTEC `Science`/`Lissajous`, 50 Hz, J2000 role:

- the immutable source owns the exact telescope `TelTime`, `SourceRaAct`, and
  `SourceDecAct` producer planes once;
- the compact derived product applies the approved 30 ms continuity, strict
  greater-than-2-arcsec telemetry-defect, eleven-record quadratic derivative,
  validity/cause, support, and raw scan-maximum rules;
- a shuffled three-part engineering schedule produces bitwise-identical
  records, support, causes, and scan summary;
- independent ALIGN views map the product onto every admitted network's native
  occurrence/time axis while preserving source record/time/weight support;
- no cross-network common analysis grid is requested or constructed; and
- the runner inspects the complete in-memory product but publishes only this
  compact evidence record, not a persistent AST product or TOD schema.

This claim does not include another scan program, pointing correction,
common-grid relation, RTC filtering or decimation, CAL, VAL, PTC/PCA, MAP, or
production-route activation.

## Exact input and build binding

The run binds observation `(152390, 0, 2)` through the approved canonical APT
bundle and all 11 admitted raw network sources. The telescope artifact is
`tel_toltec_2026-02-19_152390_00_0002.nc`, 24,157,872 bytes, with SHA-256
`2845455a620635955c00a4731e0d9720cfa456fece79d1729cf755a366a1ad6b`.
The APT manifest has SHA-256
`67f9b0bec16bd06befe74a7c66d87ae4e4f69891607ab01ad69d2138b7bf833d`;
its canonical semantic and envelope identities and every raw filename, byte
count, and digest are retained in the record.

The exact clean-source build gate verified full Citlali revision identity,
ignored-source cleanliness, accepted design commit `46824f7de`, accepted ALIGN
repair `d55deefb3`, and the pinned Kidscpp and Tula revisions, local build
patches, and resulting trees. The runner and validator do not content-hash
scientific planes or add generalized lineage.

## Representative results

The complete 62,109-record telescope product reports:

- 62,109 structurally valid raw directions and one continuity run;
- 62,099 classified quality records;
- exactly two telemetry defects, records `2504` and `12971`;
- 62,097 realized-valid directions and 62,067 valid derivatives;
- an admitted maximum of `221.40490828695155 arcsec/s` at telescope record
  `16973`, with zero maximum causes; and
- a direct-adjacent raw diagnostic maximum of
  `1421.5293957438141 arcsec/s`, demonstrating why the approved defect
  treatment is scientifically material rather than cosmetic.

Across all 11 network-native axes the runner inspected 1,666,908 occurrences:
1,666,171 have complete mapped support and 737 are explicitly unavailable.
All identity, support, interpolated-value, unavailable-cause, record,
telemetry-support, derivative-support, and summary mismatch counts are zero.
Network 0 begins at `1771483233.0700104` Unix seconds while network 7 begins at
`1771483233.0754585`; the independent native time vectors remain distinct.

The raw derived-record plane owns 2,981,232 logical bytes and references one
source time axis plus two source direction planes. The 11 compact mapped views
own 80,011,584 logical bytes. Process-lifetime peak RSS was 257,687,552 bytes;
that is a whole-run harness measurement, not a route-local allocation claim.

## Reproduction and validation

The opt-in target is:

```sh
cmake --build build --target citlali_wp7_ast_scan_motion_acceptance -j 8
```

Validate the retained record and exact executable bytes with:

```sh
$HOME/tolteca/bin/python -B tools/wp7/verify_ast_scan_motion_acceptance.py \
  handoff/WP7_AST_SCAN_MOTION_ACCEPTANCE_152390_V1_2026-08-30.json \
  --expected-record-sha256 9b1652fc158de6aa17732213d25333316b95045e96d4d2fa97737e6edf9015fe \
  --expected-source-revision f2bf3f1e00226d6e7f63e99d4da61d37ea4ddf3d \
  --expected-executable-sha256 376879038ef10f950f997eb82224ded119767bbd7efc94b0c40bf071e6fa279c \
  --executable build/bin/citlali_wp7_ast_scan_motion_acceptance
```

Before parsing JSON, the validator hashes the raw record bytes and requires the
exact digest printed above. It then hard-pins the observation scope, exact
telescope identity and cardinality, owner defect records, derivative
cardinality, maximum envelope, 11-network APT relation, participant sums,
distinct network times, compact ownership facts, and every zero-mismatch
requirement. Its focused mutation tests reject record/source/executable
substitution, mutated scientific boundaries, chunk or mapping mismatches,
network-time collapse, inconsistent participant totals, a common-grid claim,
and persistent AST publication.

The implementation and evidence snapshot pass:

- all 12 focused AST/ALIGN synthetic, boundary, mapping, dependency, and
  chunk-invariance CTests;
- all 882 runnable repository CTests, with the one established disabled test
  unchanged out of 883 registered tests;
- all 207 baseline-tool unit tests;
- all eight acceptance-validator mutation tests;
- all 129 required config unit tests and every downstream preflight audit;
- public-header isolation compilation through the two dedicated AST header
  translation units; and
- the local `citlali_cli` build.

## First review disposition and bounded repair

The first independent review of exact package snapshot
`b0e5dde2ac532a7a36e141bf22c7560e0fbbc8a1` returned `HOLD` with one `MAJOR`
finding and no `BLOCKER` or `MINOR` findings. The scientific implementation,
representative results, network timing, compactness, scope, recorded hashes,
and repository gates otherwise conformed. The finding was that the package
documented the retained record SHA-256 but `validate_exact_package()` accepted
an already-parsed object and therefore did not enforce that digest. Material
substitutions to otherwise structurally valid result, APT, participant, and
memory fields could consequently pass exact-package validation.

The bounded repair requires the expected record SHA-256, hashes the exact raw
JSON bytes before parsing, and adds mutation coverage for every demonstrated
substitution class. It does not alter or regenerate the retained record, runner
executable, representative result, scientific contract, or implementation.
Fresh exact-SHA re-review of this repaired package remains required.

## Independent review request

Review the exact implementation revision, retained record, validator, and the
evidence/status snapshot containing this package. Return `PASS`, `HOLD`, or
`FAIL` with findings graded `BLOCKER`, `MAJOR`, or `MINOR`. In particular,
verify that:

1. the runner exercises the production AST and ALIGN contracts rather than a
   circular duplicate algorithm;
2. the representative record truthfully binds exact source, executable,
   dependencies, telescope, APT, and raw participants;
3. the defect, derivative, maximum, support, validity/cause, and chunk claims
   follow `wp7-ast-scan-motion-v1` exactly;
4. network-specific timing remains authoritative and no common-grid machinery
   enters ordinary AST mapping;
5. immutable inputs are referenced rather than duplicated into the evidence
   or product; and
6. the package does not broaden the accepted scientific or implementation
   scope.

Until that review passes, the exact `221.40490828695155 arcsec/s` result is a
representative candidate product value, not an authorized RTC planning input.
