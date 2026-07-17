# Phase 4.1 Native Validation Projects Handoff - 2026-07-17

## Decision

The Citlali validation suite must use native TolPROJ projects and the existing
TolPROJ workflow. A validation-specific implementation of raw-data staging,
APT copying, Beammap-prior copying, config installation, or immutable
`project.yaml` locking would duplicate production behavior and is not retained.

TolPROJ commit `e0754af` implements the corrected design. It supersedes the
suite implementation in commits `39f724d` and `8310c24` without rewriting
history.

## Current Shape

The portable artifact is one path-free `suite.yaml` containing the selected
observations for point, OOF, Beammap, and science. On Unity:

```bash
tolproj validation-suite init /work/toltec/citlali-validation-data/v1
tolproj validation-suite verify /work/toltec/citlali-validation-data/v1
tolproj validation-suite plan /work/toltec/citlali-validation-data/v1
```

Initialization queries the configured metadata database and creates native
projects at `point/`, `oof/`, `beammaps/`, and `science/`. The science project
contains its own eight pointing-support observations. Beammap initialization
selects only observation 148670, then uses the existing Beammap classifier to
discover source-matched pointing support.

The generated plan lists only existing TolPROJ commands for copying raw data,
reducing tunes, building cohorts, selecting and matching APTs, running pointing
calibration, estimating Beammap flux, and installing the V2 `--refactor`
numbered configs. The suite command does not execute those operations itself.

## Important Operational Detail

`validation-suite init` requires a fresh empty root. An earlier `/v1` created
with the superseded installer must be moved aside or intentionally removed on
Unity before the corrected initializer is run. Do not mix the old pseudo
projects with the new native projects.

Native `project.yaml` is live TolPROJ state. Cohorts, status flags, APT choices,
and setup results are expected to change. Suite verification protects the
portable selection and requested observation membership, not byte-for-byte
project-file identity.

## Verification

- TolPROJ full suite: 104 tests pass.
- Focused Ruff checks pass for every touched Python file.
- Python byte-compilation passes.
- Citlali compilation is not affected by this TolPROJ-only correction.

Phase 4.1 remains open until a fresh Unity root is prepared through the normal
TolPROJ machinery and point, OOF, Beammap, and science smoke reductions pass.

## Unity Setup Follow-Up

TolPROJ commit `d2c90f3` records corrections found during the first setup:

- point and OOF no longer pre-create unused nested `pointings/` directories;
- Beammap no longer pre-creates `pointings/` or `apts/`;
- science retains all four working directories because it owns supporting
  pointing reductions;
- pointing, science/OOF, and Beammap `02_redu.sh` generators all request the
  configured partition (`toltec-cpu` on Unity);
- Python workflow logs use action-specific filenames instead of the generic
  `logs/tolproj.log`, while SLURM stdout remains `<jobname>-%j.out`.

Verification after this follow-up: 105 tests, full Ruff, and Python
byte-compilation pass.

## Science Pointing-Product Ordering

The first suite science attempt exposed that `setup-science-reductions` could
write `cal_objs` paths before the science project's own pointing reduction had
created them. TolPROJ commit `9fb4c80` now fails science setup immediately when
any selected pointing-product directory is missing. It also adds
`--pointing-reduction reduNN` to science setup, matching the existing option on
`calibrate-pointing-flxscale`, so both stages can deliberately use the same
accepted pointing run. The canonical first-run expectation is
`science/pointings/reduced/redu00/<obsnum>`; the separate compact `point`
validation project is not a substitute for science's self-contained pointing
support. After this repair, 106 tests, full Ruff, and byte-compilation pass.

The first populated Unity tree exposed a more precise product-selection
contract. TolTECA's `CalObj.load_data_objs()` recursively searches a configured
directory for `ppt_*.ecsv` and requires exactly one match. An observation-level
pointing directory contains both `raw/ppt_*.ecsv` and
`filtered/ppt_*.ecsv`, so the directory exists but TolTECA rejects it as
ambiguous. TolPROJ commit `704b486` now validates exactly one raw table and
emits `science/pointings/reduced/reduNN/<obsnum>/raw` in `cal_objs`.
Beammap-generated pointing references use the same explicit raw-product
contract. After this correction, 106 tests, full Ruff, and byte-compilation
pass.
