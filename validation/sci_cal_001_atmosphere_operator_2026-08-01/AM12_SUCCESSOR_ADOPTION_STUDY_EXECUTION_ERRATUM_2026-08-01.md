# SCI-CAL-001 AM12 successor adoption-study execution erratum

Date: 2026-08-01  
Scope: evidence-runner correction only; the frozen scientific protocol,
profiles, opacity coordinates, elevations, passbands, spectra, gates, and
operator candidates are unchanged.

## Disposition

The first execution attempt is invalid as a completed adoption study and is
excluded from the decision package.  Its external cache is retained read-only
at
`/private/tmp/sci_cal_001_am12_successor_adoption_v1_20260801_root`.
No artifact produced by that cache is eligible for the successor decision.
A corrected runner must use a fresh, non-overlapping external cache and must
pass a cache-only byte-for-byte replay before its results are interpreted.

## First-attempt identity and observed stop

- Runner commit: `b9f4f48b9`
- Runner SHA-256:
  `f61a8f94edb0fe0e71c96f76cff528f3aaf0cdaab3d14733999ac934c606e96f`
- First execution-context SHA-256:
  `f0acb32cd43fd0bd128a06ab8d7e354bc6a6c1389d6d0794db716753d03f85c8`
  (`17073` bytes)
- Context creation: `2026-08-01T21:53:21-0400`
- Final raw-output-directory modification observed:
  `2026-08-01T22:08:47-0400`
- Terminal error:
  `missing cached AM evidence for direct_full_grid_all_hypotheses_LMT_DJF_25_am_q50_za70_edb85ea13cb86d411e412221`
- Retained first-attempt inventory at the stop: eight scale traces, 1,025 raw
  outputs, 1,025 execution sidecars, 21,423 AM spectral-cache files, and zero
  files in `failed_attempts`.

The zero `failed_attempts` count means no AM subprocess failed its bounded
warning contract.  It does not rehabilitate the first attempt: the top-level
study failed before constructing any decision artifact.

## Exact cause

The v1 adoption runner constructed every P1 training-grid lookup with the
stage string `direct_full_grid_all_hypotheses`.  The frozen P1 scale table,
however, records `ancillary_screening_transmission_rank1=true` for
`am_q50/LMT_DJF_25` and `am_q75/LMT_DJF_75`.  The canonical P1 generator used
the stage string `direct_full_grid_selected_transmission_rank1` for those
rows.  Stage is part of the P1 cache identity even though it is not an AM
physical input.

For the first failing lookup, the canonical P1 evidence is:

- cache ID:
  `direct_full_grid_selected_transmission_rank1_LMT_DJF_25_am_q50_za70_e16c1455811e2a6191fbb7b7`
- raw-output SHA-256:
  `4c13a63e9f7b15e68cfde8995660aafa906da7b98f0c0a385739c72f072c55b9`
- sidecar SHA-256:
  `2c47499b677322223581f14aaafc3d3f2978202b7ff5a0be1b08d03e43b1e908`
- exact physical argv after the executable:
  `LMT_am_inputs/LMT_DJF_25.amc 0 GHz 500 GHz 10 MHz 70 deg 2.02963214820032256e+00`
- profile SHA-256:
  `aeeeeb48bef422f2d9392b5d7a3d62ab1887fd9e7c10322d5246d914841ba866`
- AM executable SHA-256:
  `78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb`
- canonical P1 execution-context SHA-256:
  `05148050e96e73577ec75be525b026b5bf37bbd2a8753f8e3702fc0b6dfb2bee`

Thus the evidence was not missing.  The expected and actual keys differed in
the documentary `stage` field, which also changed the derived cache-ID hash.
There is no atmospheric-profile, AM-version, H2O-scale, frequency-grid,
elevation, executable, or post-processing change in this correction.

## Bounded correction

The corrected runner must derive the P1 stage from the frozen
`ancillary_screening_transmission_rank1` field for each training row, reject
values other than literal `true` or `false`, and otherwise preserve the v1
study logic.  It must increment the study schema version and bind this
erratum's SHA-256 into the new execution context and final manifest.

Because the first cache binds the v1 runner SHA-256, it must not be resumed or
relabelled by the corrected runner.  The corrected study must execute into
`/private/tmp/sci_cal_001_am12_successor_adoption_v2_20260801_root` from an
initially absent path.  The first cache remains excluded evidence of the
failed execution attempt.
