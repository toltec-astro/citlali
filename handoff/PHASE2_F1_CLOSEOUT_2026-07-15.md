# Phase 2 F.1 Closeout - 2026-07-15

## Decision

Phase 2, configuration authority and provenance, is a release candidate. Its
implementation and local audit gates are complete. It is not yet closed: one
Unity point reduction from the exact candidate commit remains required.

The owner approved the generated low-level Citlali YAML as Citlali's immutable
configuration and source-provenance boundary. TolTECA owns discovery, ordering,
merge semantics, and future provenance for upstream numbered authoring files.
Citlali records exact source bytes, ordered path/role/precedence/hash entries,
and the canonical merged low-level YAML it actually receives. It does not infer
upstream metadata that was not supplied.

Compact-config production rollout is deferred. The retained compact profiles
are compatibility prototypes, not approved operational defaults. The complete
TolTECA numbered-overlay test matrix in external-review item I8 remains a
mandatory gate before production rollout, not a Phase 2 exit requirement.

## F.1 Checklist

### Authoritative validation contract

Locally satisfied. The generated runtime schema contains 724 normalized YAML
nodes derived from the checked leaf contract and retained default config.
Startup rejects unknown Citlali-owned nodes, including unknown empty
containers. The externally owned `inputs` subtree is outside that rejection
scope. Existing boundary readers diagnose missing required keys, conversion
failures, enums, duplicates, finite-value failures, and domain/range failures;
typed validation errors now enter the same fatal diagnostics. Observation-
resolved astrometry and Beammap photometry are constructed and validated before
installation into numerical processors.

Evidence:

- `config_schema_validation.h` and generated
  `config_leaf_schema_generated.h`;
- `reduction_config_validation_logging.h`;
- 410 passing CTests and 96 passing config tests;
- accepted point, OOF, science, and Beammap astrometry/photometry gates.

### Explicit leaf state classification

Satisfied. `config_leaf_contract_resolved.json` contains 573 unique leaves:
572 executable and one explicitly ignored deprecated leaf. Each record has an
authority, owner, unit, allowed domain, applicable modes, lifecycle/state
class, resolution stage, and validation source. Preflight fails on uncovered
leaves or manifest drift.

### Immutable request and one-way transitions

Satisfied. The 15-domain inventory records three typed-authoritative domains,
11 typed-authoritative domains with one-way compatibility adapters, and one
external KIDs boundary. Requested values are retained separately from effective
plans and observation/realized state. No typed/legacy reverse synchronization
is accepted by the boundary audits.

The final two census gaps are locally closed:

- all 28 learning leaves enter `TimestreamLearningConfig`, then one adapter
  creates the unchanged numerical `ReductionLearningState::Options` value;
- all 14 interface timing offsets enter `InterfaceSyncOffsetConfig`, then one
  adapter populates the unchanged runtime alignment map.

### Parity at the correct phase

Satisfied for migrated execution paths. Focused tests cover parsing,
context-free resolution, one-way adaptation, observation replacement, and
realized lifecycle/cardinality. Boundary audits freeze the adapter direction
and provenance coverage. Processor-derived results are recorded as realized
metadata rather than copied back into immutable requests.

### Atomic observation configuration

Satisfied for supported modes. Astrometry validates finite offsets, MJD
ordering, complete observation coverage, and no extrapolation before
installation. Beammap photometry replaces rather than merges complete finite,
positive per-array source fluxes. Repeated OOF and science observations prove
that later observations do not inherit missing state. TolTECA owns pointing
support selection and TolProj owns calibrator-flux estimation; Citlali owns
application of the supplied values.

### No duplicate mutable facts or processor YAML reads

Satisfied by local static gates. The config-read census reports 26 declared
boundary files, 131 read sites, 133 accesses, and zero records requiring
review. RTC, PTC, mapmaking, post-processing, Beammap, and mode execution consume
typed values or a declared external KIDs configuration. Compatibility aliases
remain at loading boundaries.

### Versioned provenance

Satisfied in implementation and accepted mode evidence, subject to the final
point gate for the two newest additions. Required atomic sidecars retain
requested, effective, observation-resolved, and realized state appropriate to
each domain. The source manifest preserves exact low-level source identity and
the canonical merged input. Calibration and external-tool identities are
recorded where supplied.

The new candidate additions are:

- complete learning policy in processed-timestream requested/effective
  provenance;
- all 14 interface offsets in raw-timestream provenance v2 with unit `s`.

### Behavior-appropriate validation

Satisfied through the accepted matched-mode matrix at `9ea6d7f01`:

- point `redu61`;
- OOF `redu02`;
- science `redu20` through `redu23`;
- Beammap `redu05`.

The matrix validates exact products where deterministic and the approved
science/OOF tolerance where OMP ordering permits tiny numerical drift. Enabled
polarimetry remains deliberately unavailable and is mechanically rejected; its
future scientific contract is not closed off or claimed as validated.

The newest learning and interface-sync changes affect startup authority and
ordinary point execution, so the bounded final gate is a current point run.
They do not justify rerunning hour-long science and Beammap fixtures unless the
point gate exposes an unexpected difference.

### Zero unexplained errors

Satisfied for all accepted snapshots. Reduction audits independently check
serious log records, required provenance, completion state, and output
completeness. A done marker alone is never sufficient evidence.

## Local Candidate Evidence

- `cmake --build build --target citlali_cli -j 8`: pass.
- `ctest --test-dir build --output-on-failure -j 8`: 410/410 pass.
- `tools/config/run_config_preflight.py --require-all`: pass.
- Config tests: 96/96 pass.
- Compact compatibility fixtures: 8/8 pass.
- Compact-surface coverage: 100%, zero gaps.
- Leaf contract: 573 leaves, 15 authorities.
- Config-read census: zero review-required sites.
- Learning and interface-sync boundary audits: no drift.

## Final Unity Gate

Build and run the standard point fixture from the exact candidate commit. The
run is accepted only if all of the following hold:

1. The executable reports the expected candidate revision.
2. The reduction completes with zero unexpected error-, critical-, or
   fatal-level records.
3. The low-level input is byte-identical to the immediately preceding accepted
   point configuration.
4. Every stable product, including complete RTC/PTC TOD and metadata, is exact
   against the matched accepted point fixture.
5. Processed-timestream provenance contains the full requested/effective
   learning policy exercised by the standard point configuration.
6. Raw-timestream provenance uses schema v2 and records all 13 TolTEC offsets
   plus HWPR, with requested/effective equality and unit seconds.
7. The required configuration-source manifest and all applicable provenance
   sidecars pass semantic audit.

After acceptance, update the `learning` and `interface-sync` inventory entries
from partial to complete, record the run and revision in `REFACTOR_STATUS.md`,
mark Phase 2 complete, and begin Phase 3. Do not add more Phase 2 analysis-
control migrations while this candidate is awaiting validation.

## Phase 3 Entry

Phase 3 begins with a design checkpoint, not more YAML migration. Its first
bounded work is the minimal non-CLI reduction session/result contract and
explicit lifecycle ownership. The first compiled `.cpp` boundary follows only
after that ownership boundary is named and measured. Compact rollout,
scientific-kernel modernization, R execution, and enabled polarimetry remain
separate follow-up work.
