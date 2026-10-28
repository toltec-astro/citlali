# Disturbance exclusion: 152418 ready for owner launch

Implemented the owner-approved first rule and compact detector-drop diagnostic.
**No reduction was launched. Grant alone runs the command below.** The previous
failed 152418 reduction and reference maps remain in their original directories.

## Scientific change

At least two distinct admitted level-shift events must affect distinct science
processing segments. After the existing whole-segment cuts, numerical output
must remain, both final strict-grid assessments must be unavailable specifically
with fixed_grid_unavailable on nonempty support, and original spectra must be
available without input/arithmetic errors. No new duration or amplitude cutoff.
The paired detector is excluded before atmospheric fitting. One frozen decision
is applied from original inputs; no cascading retry. Existing all-array and
remaining-data guards continue to determine whether products can be made.

The required detector-drops.json reports total and per-array new drops, exact
detector identities and causes, prior unavailability, and incomplete evaluation.
Unavailable is never reported as zero. Quiet mode prints the compact counts and
diagnostic path. Final map contribution remains a separate, unmeasured quantity.

## Exact prepared trial

Project root:
`/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027`

New directory: `pointings-disturbance-152418-20261028`.
Evidence and owner driver: `evidence/disturbance-152418-20261028`.
Runtime bundle: `software/configured-point-drop-4767cc144`.
Existing frozen preparation layer: `software/preparation-20261028`.

The previous 152418 numbered configuration was copied into a fresh directory,
set up through TolTECA, and changed only to select the new runtime and disturbance
policy. Existing TolProj executable snapshots, generated launcher overlay and
resource validation were reused. No sibling repository changed.

Verified final TolTECA merge: observation152418/scannum2, a1100/a1400/a2000,
observed-ALS ranks3/4/6, existing-array-lowpass-v1, 2arcsec pixels, coverage0.15,
three workers, verbose=false, feedback=false, intermediate retention=false.
The check-only driver verifies numbered-file hashes and frozen launcher identity
before a user-requested launch. The runtime launcher additionally verifies its
manifest and all bundled sources/native executable. No observational input
preparation, spectra, fits or map calculations were performed during this task.

Source: `4767cc1440d973ee883868e4d744b060883db895` (clean when frozen).
Native version contains `4767cc144`; native SHA256:
`76282265e403aff49e9e90ffbc92fa56788bc5d52cba9fa7821e4603bc85556d`.
Bundle manifest SHA256:
`0f115bbec7a1cd85ec0b84fd7e9764b81086ac281ebc04910ed37319e05e4a2c`.
TolProj snapshot/launcher-source SHA256:
`fc26882d8b09363601374dd549ee58fbb4f3e0239ef43c88d4f919c25d4d8fff`.
Full paths and configuration hashes are in evidence/executable.json and
numbered-config-bindings.json under the evidence directory above.

The absolute local runtime still depends on the existing Python environment and
native shared libraries; package contents are not fully pinned. Native version
also reports the existing `kids 04088da-dirty` dependency. This is supplemental
Mac AppleClang/Homebrew evidence, not a Spack/Unity-qualified build. Subsequent
documentation commits do not replace this reviewed frozen source.

## Checks and review

CLI build and isolated new public header compile passed. All three synthetic
native disturbance tests pass, including unaffected-detector preservation,
partial RTC completion, donor safety and foreign-plan rejection. All165 focused
Python adapter/package tests pass. Earlier broader gates on the implementation
passed1335 enabled CTest tests (one inherited disabled),207 baseline tests plus
137 subtests and the full require-all config preflight. The small review repairs
were verified with affected native/Python checks and rebuilt before freezing.

[Exact-source independent review](../validation/configured_point_2026-10-28/DISTURBANCE_REVIEW.md)
passes scientific/behavioral, architecture/ownership and repository/evidence
dispositions with the recorded local-environment and observational limitations.
Shell syntax, frozen runtime checksum/version, TolTECA merged configuration and
TolProj three-worker resource validation pass. The new reduced directory does
not exist at handoff. Source configuration hashes in the previous trial remain
unchanged. No push, canonical integration or production activation occurred.

## Grant's command

```bash
cd /Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-disturbance-152418-20261028
bash 02_redu.sh
```

The script verifies the preparation and invokes the existing TolTECA reduction
layer through TolProj's local launcher. It does not remove previous outputs.
The first run's diagnostic will be `reduced/redu00/detector-drops.json`;
successful maps/fit diagnostics will be under `reduced/redu00/152418/raw`.
Later invocations reserve the next unused reduNN. A failed run retains its
diagnostic and failure record; a scientific guard can still stop publication.

Next evidence: Grant reports completion/failure, then inspect actual drop
identities/counts, remaining-data completion and all three maps/fit diagnostics.
The actual number dropped and any improvement remain unknown until that run.
Calibration-consumer compatibility, SCIENCE/coadds and Unity qualification remain
outside this bounded trial.
