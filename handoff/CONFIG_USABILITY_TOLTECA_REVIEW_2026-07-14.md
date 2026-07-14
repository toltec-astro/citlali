# Compact Configuration And TolTECA Usability Review

Date: 2026-07-14

Status: completed read-only review. This note records conclusions for later
implementation work. It does not authorize or implement compact-config runtime
rollout, change Citlali or TolTECA behavior, or replace the governing roadmap in
`doc/REFACTOR_STATUS.md`.

## Review Scope

The review covered:

- the provisional compact-config exposure policy and machine-readable key
  classification;
- the simplification inventory and compact-surface coverage audit;
- compact profiles, examples, translation behavior, and compatibility cases;
- representative historical `70_reduce.yaml` inputs used by the compatibility
  suite;
- the current matched pointing, OOF, Beammap, and science validation
  `70_reduce.yaml` files under
  `/Users/gwilson/work_toltec/local_data/2026-refactor`;
- real `40_setup.yaml`, `72_reduce.yaml`, and later low-level overlays;
- the adopted external architecture review and the living five-phase roadmap.

No files were changed during the original review. The classification command
was also applied read-only to the four current refactor validation files. Those
files contain 1,777 leaf occurrences and 534 unique normalized paths: 96 paths
are user-facing, 432 expert, three hidden/internal, and three deprecated. No
path fell through to the fallback rule. This supports the conservative policy,
but does not itself validate compact expansion, TolTECA merge semantics, or
products.

## Executive Conclusion

Keep compact configuration as an authoring and translation layer, but do not
make it operational yet. The four-intent model is sound:

- `pointing`
- `oof`, which remains a first-class user intent while expanding to the legacy
  pointing engine
- `beammap`
- `science`

The correct overall model is a versioned intent/profile, sparse normal
overrides, a generated legacy-compatible low-level tree, and an explicit
expert escape hatch. The present risk is not inadequate low-level reach. The
risks are policy misalignment, uncalibrated profile defaults, diagnostic
controls that can change processing, and incomplete evidence for real ordered
TolTECA merges.

The living roadmap is authoritative. Phase 2 config authority and provenance
is active. Compact rollout remains paused until the active mode gates and real
TolTECA overlay acceptance pass. Compact deployment is not required to finish
the current typed-config ownership phase.

## Findings

### 1. The exposure policy is directionally correct

The policy appropriately exposes intent, products, geometry, resources, and
common scientifically meaningful choices while keeping detailed implementation
controls available through explicit expert overrides. The existing low-level
schema must remain accepted during the refactor. Expert classification means
hidden from ordinary templates, not removed or forbidden.

The policy is conservative enough to cover both the historical compatibility
inputs and the current validation files without fallback-classified paths.
That is a useful foundation for a later production schema.

### 2. The current coverage audit proves underexposure coverage, not policy compliance

The reported 100% actionable coverage means that every low-level path
classified as user-facing can be expressed by a compact field without using
`expert:`. It does not prove the reverse: that all ordinary compact fields map
only to user-facing paths.

Several compact fields currently reach paths that the policy calls expert or
profile-owned:

- `runtime.parallel` reaches expert `runtime.parallel_policy`;
- `map.grouping` reaches expert `mapmaking.grouping`;
- `map.center` and `map.size` reach expert manual geometry;
- `products.maps` reaches profile-owned/expert `mapmaking.enabled`;
- `processing.tod` reaches profile-owned/expert `timestream.enabled`.

These fields appear prominently in normal examples. A future preflight needs
an exposure-compliance audit in addition to the existing coverage audit. It
should reject any ordinary compact field that reaches an expert destination
unless the value appears under `expert:` or is emitted by the selected
profile.

### 3. Profiles named `*_standard` are prototypes, not approved operational defaults

The documents already record 178 low-level differences between the initial
`pointing_standard` profile and the representative pointing baseline. The
current science and Beammap differences are also substantial:

- `science_standard` selects naive mapping, 1 arcsec pixels, no coadd or noise
  maps, no active cleaner, and no fruit loops;
- the current science validation selects JINC, 2 arcsec pixels, coaddition,
  noise maps, standard PCA, and fruit loops;
- `beammap_detector` selects one iteration, no priors, naive mapping, and no
  fruit loops;
- the current Beammap validation selects three iterations, priors, JINC,
  detector grouping, fruit loops, and detector TOD output.

The current profiles are valid translation prototypes, but the names
`science_standard`, `pointing_standard`, and `beammap_detector` could be
mistaken for scientifically approved defaults. Until owners approve and
validate exact profile snapshots, they should be treated or labeled as
prototype profiles. Production defaults must be derived from accepted current
validation snapshots, not from the oldest compact examples.

Historical compatibility inputs should remain as legacy round-trip cases.
They should not silently define current operational policy. Add a distinct
current-profile suite tied to accepted validation-ledger snapshots.

### 4. Diagnostic capture and scientific intervention are currently conflated

The compact `science_diagnostic` example includes an expert PCA eigenmode
choice. That changes processing and is not merely additional observation.

The `line_audit.enabled` path is classified as a high-level diagnostic toggle,
but the real line-audit subtree also contains fixed/shared/detector notch
application policy. The current Beammap and science configurations demonstrate
that enabling this family can participate in signal processing. A user-facing
diagnostic control must not silently actuate filtering.

Use separate concepts:

- `diagnostics` adds logging and products without changing numerical policy;
- a named `processing_variant` or intervention profile explicitly changes
  filtering, cleaning, flagging, weighting, or model behavior.

Diagnostic profiles should be tested for scientific invariance aside from
their requested products and diagnostic metadata. Intervention variants need
their own product comparisons and provenance.

### 5. Static translation equivalence is necessary but insufficient

The eight compact compatibility cases establish zero-difference static
low-level expansion for representative fixtures after the documented ignored
path. Some cases deliberately use empty passthrough profiles. This proves that
the translation tooling can preserve a selected low-level tree; it does not
prove that normal compact profiles implement approved defaults.

It also does not execute TolTECA's complete numbered-file merge or prove:

- list replacement and indexed-list behavior;
- literal null versus deletion semantics;
- aliases and deprecated spellings;
- unknown/unconsumed-key failure;
- multiple reduction steps;
- expert override precedence;
- clean-CI reproducibility.

Unknown compact keys currently produce warnings and can be ignored unless the
strict option is selected. Production authoring must make unknown or
unconsumed keys fatal.

### 6. Current baseline drift is manageable but must be explicit

The provisional policy was generated from the point validation file, OOF
149056, Beammap 3C273, and an older GOODS-N science file. The accepted roadmap
now includes modern matched OOF and science validation workdirs. The policy
rules still classify their surfaces without fallback, but the old fixtures do
not prove that the proposed standard profiles reproduce current operational
choices.

The historical science boundary inventory also found generated cleaner leaves
that were absent from authoring YAML. Resolve or explicitly model those
injected/defaulted leaves before claiming strict authoring-to-generated
equivalence for science.

The Beammap directory contains `70_reduce_perf_mapaccum.yaml` in addition to
`70_reduce.yaml`. A blind `NN*.yaml` scan can therefore activate an unintended
same-tier file. Production workdirs should not contain multiple active files
with the same numeric tier.

## Recommended Normal Authoring Model

Normal `70_reduce.yaml` content should be sparse. Omitted values mean "use the
selected versioned profile." Examples should not restate every profile default.

### Shared normal controls

Expose these across modes:

- `mode` and `profile`, including an explicit profile/schema version;
- `runtime.threads`;
- `output.subdir` for local layout;
- `map.unit`, `map.method`, `map.pixel_axes`, and
  `map.pixel_size_arcsec`;
- `map.coadd`;
- `products.noise` and `products.noise_count`;
- `products.tod: none | rtc | ptc | both`;
- `products.diagnostics: normal | verbose | sampled_tod | full_tod`;
- semantic `products.map_filtering` and `products.source_finding` presets;
- `processing.clean: off | standard | null_model | marchenko_pastur |
  adaptive`;
- `processing.weighting` as a named policy;
- `processing.fruitloops: off | standard`, with an iteration control when
  enabled;
- a calibration preset such as `standard | flux_only | none`, instead of a
  collection of loosely related calibration booleans.

In a TolTECA workdir, the absolute output directory and input assembly remain
TolTECA-owned. `output.dir` can remain useful for standalone translation, but
should not be an ordinary TolTECA reduction choice. `source.map_regime` should
normally be derived from intent/profile and remain an advanced override for
unusual fields.

Keep these out of ordinary templates:

- the parallel backend;
- mapmaking and timestream master enable switches;
- manual grouping, center, and map size;
- fit-report and absolute output paths in TolTECA workdirs;
- individual filter, despike, downsample, IIR, and line-actuation controls;
- polarimetry until its capability contract and validation gate are approved.

### Pointing

Normal pointing controls should be:

- a validated source-strategy enum rather than an arbitrary strategy mapping;
- source-protection radius;
- Gaussian fit on/off;
- fit radius and fit box;
- optional source-finding and filtered-map products;
- shared map geometry, product, cleaner, weighting, and diagnostic controls.

The individual header radius, coverage requirement, and center-resolution
fields should normally be profile-owned. Detailed source-fit model bounds
remain expert.

### OOF

OOF remains a first-class compact intent even while it expands to the legacy
pointing reduction type. Normal OOF controls should be:

- center policy, normally `map_center`;
- Gaussian fit on/off;
- fit radius and fit box;
- one semantic source-support radius;
- shared map, product, cleaner, weighting, and diagnostic controls.

The profile should derive the separate RTC despike protection, PTC second-pass
protection, fruit-loop adaptive support, and center-keep radii. Separate
per-stage radii remain available through expert overrides when diagnosing a
scientifically significant failure. This reduces several coupled knobs without
removing the ability to isolate source erasure or biased support.

### Beammap

Normal Beammap controls should be:

- iteration count;
- convergence tolerance and convergence radius;
- derotation;
- reference subtraction expressed as `off | auto | detector-id`;
- detector-weighting preset;
- priors expressed as `default | off | filepath`;
- detector-TOD and split-FITS product toggles;
- shared map, product, and diagnostic controls.

RFI and scan-band masks can remain common-advanced intervention toggles, but
must be labeled as science-affecting. Their thresholds and geometry remain
expert. Detector grouping should be profile-derived rather than repeated in
normal Beammap files.

### Science

Normal science controls should be:

- map method, frame, unit, and pixel size;
- coadd;
- noise maps, count, and requested noise products;
- cleaner and weighting presets;
- fruit-loop preset and iteration count;
- map-filter preset;
- optional source catalog;
- diagnostic product level;
- calibration preset.

Wiener template family or FWHM may remain common advanced when a reducer has a
scientifically justified reason to change it. Wiener convergence and
stability parameters remain expert.

## Expert And Debugging Policy

Retain both expert mechanisms, but prefer one authority in any given workflow:

- standalone compact translation may use top-level `expert:`;
- a TolTECA reduction directory should normally use a sparse
  `80_expert.yaml` under `reduce.steps.0.config.low_level`.

Do not duplicate the same expert setting in both locations. The later TolTECA
overlay is the operational convention and must appear in provenance.

Expert-only families are:

- manual interface synchronization and KIDs fitting internals;
- parallel backend, manual map geometry/grouping, coverage cut, JINC shape,
  and maximum-likelihood controls;
- raw despike thresholds and local-event morphology;
- filter bands, fixed/dynamic notches, edge guards, IIR parameters,
  downsampling factor, alt-az destriping, and raw source-kernel details;
- line-audit thresholds and every notch-actuation policy;
- detailed cleaner thresholds and PCA/MP/null/adaptive parameters;
- weight validation, caps, correlation penalties, and busy-row suppression;
- detailed processed deglitch and second-pass flagging;
- fruit-loop flux/SNR gates, local-sigma support, feedback, and interpolation
  details;
- learning and pathology thresholds;
- Beammap prior scoring/alignment, flagging thresholds, mask thresholds,
  phase strategy, fit support, sensitivity limits, split-FITS flag selection,
  and detector-TOD sampling/layout;
- map-filter edge conditioning, Gaussian fit bounds, and Wiener convergence;
- detailed diagnostic product layout and sampling.

Manual photometry, astrometry, APT selection, and calibration-item values
belong to the TolTECA target/calibration layer rather than ordinary compact
processing. Deprecated aliases remain accepted only at the compatibility
boundary, translate to one canonical requested representation, and warn with
an actionable replacement. They should not be independently writable facts.

## Diagnosis Without Restoring Hundreds Of Knobs

Ship and validate named diagnostic and isolation variants instead of requiring
reducers to discover thresholds:

- `sampled_tod`: bounded representative RTC/PTC TOD plus sidecars;
- `full_tod`: complete TOD for strict comparison;
- `no_raw_filter`;
- `no_despike`;
- `no_clean`;
- `no_fruitloops`;
- `no_map_filter`.

Pure diagnostic variants are additive and must not change signal processing.
Isolation variants intentionally change one stage at a time, must record their
delta from the selected standard profile, and require a mode-appropriate
validation result. Avoid freely composable collections of unvalidated
variants; otherwise the small surface recreates the original combinatorial
configuration problem.

The existing sparse science and Beammap validation overlays are a good model:
they make a bounded change, state the validation purpose, and can be applied
identically to OG and refactor workdirs.

Every explicitly enabled output is required. RTC TOD, PTC TOD, `rtcdiag`, and
`ptcdiag` write failures must fail the reduction. Diagnostic volume can be
controlled through presets and selection, but enabled products are not
best-effort.

## Ordered TolTECA Overlay Workflow

The target order is:

| File | Authority |
| --- | --- |
| `40_setup.yaml` | TolTECA-managed setup; never manually edited. |
| `60_citlali_profile.yaml` | Generated, versioned mode/profile defaults. |
| `70_reduce.yaml` | Sparse normal reducer choices as targeted overrides. |
| `72_target.yaml` | Input selection, APT, astrometry, photometry, and target-specific calibration. |
| `80_expert.yaml` | Optional sparse raw low-level escape hatch. |

Higher numbered files win. Until a compact-aware TolTECA boundary passes
acceptance, active `NN*.yaml` files should remain legacy-compatible TolTECA
wrappers. Do not place raw compact-only keys in the production loader and rely
on them being ignored.

Required merge semantics are:

- mappings deep-merge recursively;
- a later scalar replaces an earlier scalar;
- ordinary sequences replace atomically and never concatenate implicitly;
- indexed mapping is permitted for `steps` and `inputs`, but the target must
  exist and have the expected stable identity, such as a Citlali step named
  `citlali`;
- absence means inherit;
- null is a literal value, not deletion, because current scientific configs
  legitimately use null paths;
- deletion, if required at all, needs a separate explicit operation;
- unexpected type changes fail;
- unknown keys, nonexistent indexes, inconsistent duplicates, and ambiguous
  multiple Citlali steps fail before execution;
- only one active file may occupy a numeric tier; disabled experiments must be
  renamed so they do not match `NN*.yaml` discovery.

The overlay acceptance fixture must exercise all four modes and include expert
overrides, complete-list replacement, indexed inputs/steps, literal null,
explicit deletion policy, compatibility aliases, unknown keys, multiple
steps, and conflicting/same-tier files.

## Provenance Requirements

Every reduction must retain enough information to reconstruct both requested
authoring and realized behavior. The durable manifest or output provenance
must include:

- exact source bytes for every numbered YAML file;
- ordered source path, role, numeric precedence, and collision-safe hash;
- the selected compact schema, intent, profile name, and profile version;
- every normal and expert override, including shadowed values and the winning
  source for each final leaf;
- canonical merged low-level YAML;
- canonical immutable typed requested config;
- effective execution plan after normalization and capability suppression;
- observation-resolved and realized decisions;
- calibration/APT/astrometry/photometry sources and hashes;
- translator, TolTECA, Citlali, schema, and tool versions;
- diagnostics or processing-variant identity and its declared delta from the
  base profile.

Requested, effective, and realized state must remain distinct. Automatic
grouping, unavailable calibration, observation resolution, and fallbacks must
not overwrite the accepted request. A temporary one-way typed-to-legacy
adapter is acceptable; bidirectional synchronization is not.

## Rollout Gates

Do not replace normal TolTECA templates or make compact config authoritative
until all of the following pass:

1. The active Phase 2 config authority/provenance gates and current science
   and Beammap post-processing gates close as required by the living roadmap.
2. Scientific owners approve exact versioned normal profiles for pointing,
   OOF, Beammap, and science.
3. Current accepted validation snapshots are added as a separate profile
   equivalence suite while historical cases remain as legacy compatibility
   coverage.
4. Compact coverage and the new exposure-compliance audit both pass.
5. Unknown/unconsumed compact and low-level keys are fatal with actionable
   paths.
6. Hermetic TolTECA numbered-overlay fixtures pass for all four modes,
   including lists, indexed collections, null/deletion, aliases, expert
   precedence, multiple steps, and duplicate numeric tiers.
7. The exact final merged low-level config and complete ordered-source
   provenance match expectations.
8. Compact-expanded and existing full-schema requests pass matched Unity
   product validation appropriate to each mode: complete point TOD and
   metadata, current OOF products, current Beammap detector/fit/QC products,
   and current science coadd/noise/filter products.
9. Diagnostic-only profiles prove numerical invariance apart from requested
   outputs and metadata. Each processing/isolation variant has its own bounded
   evidence.
10. Production TolTECA/tolproj templates change only after those gates; normal
    user-facing defaults change last.

Compact deployment remains optional for closing the current typed-config
ownership phase. If it is deferred, retain these gates as explicit rollout
blockers rather than weakening them.

## Principal Risks

- A compact profile with an appealing name can silently establish scientifically
  unapproved defaults.
- Normal examples can recreate the low-level surface by restating every
  default or exposing fields classified as expert.
- A diagnostic toggle can change filtering or flagging and invalidate the
  comparison it was intended to diagnose.
- Static passthrough equivalence can be mistaken for validation of profile
  policy or real TolTECA behavior.
- List, null, and indexed-overlay ambiguity can select the wrong observations,
  calibration items, or reduction step.
- Multiple active files with the same numeric prefix can introduce an
  accidental override.
- Old compatibility fixtures can freeze historical practices as current
  defaults.
- Site-specific output, APT, or prior paths can leak into supposedly portable
  profiles.
- Enabling large diagnostics without bounded presets can cause excessive I/O,
  but treating enabled outputs as best-effort would hide scientifically
  important failures.
- Freely composable diagnostic variants can recreate an unvalidated
  combinatorial state space.

## Questions Requiring Owner Decisions

These decisions should be answered before the corresponding field becomes
operational. They should not be silently inferred by an implementation task.

1. What exact accepted validation snapshot defines each initial production
   profile, including map method, pixel frame/size, cleaner, weighting,
   fruit-loop, calibration, and product defaults?
2. Which shared fields are truly normal reducer choices versus profile policy,
   especially map method, coordinate frame, map unit, cleaner, weighting, and
   Wiener template?
3. Should normal calibration be one named preset or separate flux/extinction
   controls, and which calibration fallbacks are scientifically permitted?
4. For OOF, can one user-facing source-support radius safely derive the RTC,
   PTC, fruit-loop support, and center-keep radii? If so, what are the approved
   derivation rules and units?
5. Which Beammap controls are ordinary operations rather than interventions:
   priors, reference subtraction, detector weighting, RFI mask, scan-band mask,
   split FITS, and detector TOD?
6. What are the approved additive diagnostic levels, their product-volume
   budgets, and their required output sets for each mode?
7. Must line-audit capture and notch actuation become separate public controls,
   and which actuation profiles are approved for each mode?
8. What exact TolTECA contract governs list replacement, indexed list patches,
   null, and deletion? Is deletion required at all?
9. How should multiple reduction steps be identified: numeric index with an
   asserted name, or a stable name/key merge?
10. How long must deprecated aliases and old full-schema templates remain
    supported after compact rollout?
11. Which manual calibration and source-context changes belong in
    `72_target.yaml`, and which, if any, may appear as compact normal choices?
12. When polarimetry is eventually supported, what separate profile,
    capability rejection, provenance, and enabled-product gate are required?

The project-owner decision that explicitly enabled outputs are required is
already settled and is not reopened by this review.

## Handoff Direction

A later implementation task should start by reconciling the compact schema
with the classification policy and accepted validation snapshots, not by
wiring the current prototype into Citlali or TolTECA. The first safe artifacts
are policy/profile decisions, exposure-compliance checks, hermetic overlay
fixtures, and provenance expectations. Runtime/catalog rollout comes only
after those artifacts and the governing Phase 2 gates pass.
