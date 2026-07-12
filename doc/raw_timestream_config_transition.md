# Raw Timestream Config Transition

This document defines the finite Phase 2 migration for
`timestream.raw_time_chunk`. It is a contract for changing configuration
authority, not permission to redesign RTC algorithms.

## Scope

The raw-timestream domain owns the 169 frozen YAML paths below
`timestream.raw_time_chunk`. It includes raw flagging and despiking, FIR/notch
and IIR filtering, filter edge guards, downsampling, calibration/correction
requests, kernels, AltAz destriping, and line-audit policy.

The two `timestream.polarimetry` paths currently parsed by the same
`RTCProc::get_config` body belong to the separate polarimetry authority domain.
They are inventoried by the boundary audit so they cannot disappear silently,
but they are not covered by a raw-timestream authority claim. R measured-channel
execution also remains outside this transition.

## Current boundary

The legacy-authoritative direction is:

`merged YAML -> RTCProc::get_config -> RTCProc fields -> typed raw snapshot`

`RTCProc::get_config` contains 171 unique literal paths: 169 raw-timestream and
two polarimetry paths. It has 14 direct process exits. After parsing,
`Engine::get_rtc_config` invokes ten legacy-to-typed mirror helpers. The typed
raw object already supplies several downstream policy accessors, but it is a
snapshot of legacy processor state rather than the accepted request authority.

`tools/config/audit_raw_timestream_boundary.py` freezes this characterization:

- parser path count and SHA-256 digest;
- raw versus adjacent polarimetry path counts;
- direct parser-exit count;
- the single legacy parser call; and
- the exact ordered parser-before-mirrors boundary with ten mirror helpers.

This audit is a drift detector, not an endorsement of the current direction.
Intentional migration steps update its expected state only after focused tests
make the new boundary explicit.

## Preparation checkpoint

The non-wired migration prerequisites are implemented locally. All 169 frozen
raw paths have declared direct typed readers and deterministic request
serialization. The deprecated compact-raw `candidate_sigma_scale` spelling is
an explicit compatibility alias for `candidate_rel_sigma_scale`. Disabled
sections retain supplied expert values in the request; source-protection
activity and the observation-derived extinction model are deliberately absent
from request serialization.

The external RTC access census contains 44 reviewed access shapes: 22 numerical
executor operations, seven observation-state accesses, eight output/realized
state accesses, three raw policy reads, and four polarimetry accesses outside
this domain. The additional reads are the temporary typed/legacy observation
shadow boundary. New or reclassified accesses fail config preflight until
reviewed.

An unwired `RawTimestreamExecutionPlan` now separates requested, context-free
effective, per-observation, and realized state. Context-free resolution records
filter/notch activation, downsample request form and filter dependency,
source-protection intent, and correction intent. Beginning a new observation
clears observation and realized state without modifying the request.

The unwired one-way adapter covers all 169 raw paths. Context-free request
fields are copied into the real `RTCProc` compatibility object, while native
sample rate, derived downsample factor, edge guard/context counts, source-
protection activity, and extinction availability/model are applied separately
from observation state. A complete request round trip through the existing
legacy mirrors, disabled-sentinel checks, and repeated-observation tests pass.
The boundary audit enforces 169/169 reader, serializer, and adapter coverage.
This proves field-transfer parity; it does not make typed observation
resolution authoritative. Production constructs and records the plan only as
a shadow. `RTCProc::get_config` plus the ten legacy-to-typed mirrors remain the
production execution authority.

Pure observation resolution is implemented. It returns an
explicit sample-rate error category while resolving native/effective rate,
downsample factor, and anti-alias Nyquist; computes each filter-edge transient
contribution and final guard/context counts; resolves raw source-protection
activity by reduction type; and selects the observation extinction model.
Legacy `Filter` and `Calibration` delegate to the same extracted pure numerical
policies, so shadow parity does not rely on duplicated formulas. Focused tests
match the real legacy processor for sum/max edge policies and representative
tau values.

The context-free production shadow is now wired. `Engine::get_rtc_config`
directly reads an isolated typed request before invoking the legacy parser,
stores the typed execution plan, adapts the request into a temporary RTC policy
object, and compares deterministic snapshots after the ten legacy mirrors.
Mismatch is a hard configuration failure with both snapshots logged; the
temporary object never drives numerical execution. The boundary audit requires
the typed read, legacy parser, mirrors, and shadow comparison exactly once in
that order.

The per-observation production shadow is also wired. Input preparation records
and compares native/effective sample rate, downsample factor, filter edge
guard/context, and source-protection activity. Observation setup completes and
compares extinction activity/model. A second observation resets observation
and realized state. Mismatches fail with field-level diagnostics while the
legacy processor still drives execution. Frequency-derived downsampling has one
explicitly deferred edge-guard comparison because legacy configures the guard
before deriving the per-observation factor; the typed expected guard and the
deferral are recorded without changing legacy behavior.

## Target state

The target one-way flow is:

`merged YAML -> immutable RawTimeChunkConfig request -> effective plan -> observation plan -> RTCProc execution adapter -> realized metadata`

YAML access ends at typed readers. `RTCProc` remains the numerical execution
object during this phase but does not parse configuration or populate typed
state. Runtime code may read the typed effective/observation plan directly or
processor fields populated by one typed-to-legacy adapter; it must not choose
between both authorities.

## State classification

### Requested

The immutable request retains accepted values for all 169 raw paths, including
values under disabled sections. Disabled options are inactive, not erased or
replaced by legacy sentinels. Compatibility spellings and enum aliases are
canonicalized only at the YAML boundary.

### Effective

Context-free resolution records activation dependencies and normalized values:

- FIR, fixed-notch, dynamic-notch, IIR, despike, kernel, and line-audit
  activation;
- downsampling request form: explicit factor versus requested output
  frequency;
- source-protection intent without claiming it was activated;
- calibration and extinction intent without claiming observation data allowed
  execution; and
- disabled-section policy without overwriting requested expert values.

### Observation-resolved

Observation inputs resolve decisions that cannot be known from YAML alone:

- native sample rate, derived downsample factor, and effective sample rate;
- Nyquist and anti-alias validation;
- filter-edge guard and outer-context sample counts;
- source-protection activation from reduction type;
- extinction model/calibration availability and observation tau validity;
- kernel setup requirements from map count and source context; and
- any calibration-dependent correction activation.

These values must be built and validated as one observation plan before
mutating `RTCProc`. A second observation cannot inherit availability or derived
values from the first.

### Realized

Realized metadata records what execution produced, including applied sample
rates/factors, active corrections, source-protection use, flagged-event counts,
dynamic line/notch discoveries and applications, filter-edge samples actually
used, and emitted diagnostic/product cardinality. Learned or observed values do
not flow back into the request.

## Migration sequence

1. Keep the frozen parser/mirror audit and reviewed external execution census.
2. Maintain direct typed readers and deterministic request serialization for
   all 169 raw paths while the legacy parser remains the comparison oracle.
3. Add pure effective and observation-resolution functions with tests for
   omitted, disabled, repeated-run, finite/range, sample-rate, and calibration
   cases.
4. Maintain one typed-to-`RTCProc` adapter. Compare complete processor policy state
   immediately after context-free resolution and again after observation
   resolution. Do not compare later learned/diagnostic state as config parity.
5. Publish versioned requested/effective/observation/realized provenance and
   validate its semantics.
6. Accept a strict point run with complete RTC/PTC timestream comparison, then
   accept affected beammap and science gates. OOF may reuse the explicit
   pointing execution gate. Polarimetry requires its own authority and
   validation decision.
7. Remove the 169-path legacy parser and all raw legacy-to-typed mirrors. If the
   two adjacent polarimetry reads are not yet migrated, isolate them behind a
   named, finite compatibility boundary rather than retaining a generic raw
   parser.

## Removal gates

The raw parser and mirrors may be retired only when:

- all 169 paths have direct typed readers, validation, and deterministic
  serialization;
- the execution-read census has no unclassified raw policy reads;
- requested, effective, observation-resolved, and realized states have one
  owner and repeated-observation tests;
- adapter parity passes at the context-free and observation-resolved phases;
- direct parser exits are replaced by propagated validation failures;
- provenance is schema-versioned, atomically required, and semantically
  audited;
- point, beammap, and science evidence is accepted with zero unexpected errors;
  and
- the config authority inventory records typed authority and only a one-way
  execution adapter.

## Stop conditions

Do not redesign despiking, filtering, downsampling, kernel, line-audit, or
calibration algorithms in this migration. Do not mutate the immutable request
with sample-rate-derived or observation-derived values. Do not synchronize
typed and RTC state bidirectionally. Do not claim polarimetry authority from
non-polarimetric point/beammap/science evidence.
