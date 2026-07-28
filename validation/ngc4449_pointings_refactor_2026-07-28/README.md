# NGC4449 Pointing Re-reduction Setup

Date: 2026-07-28

Status: owner-run Unity setup; no Unity command has been executed by Codex.

## Scope and frozen inputs

The local project mirror inspected on 2026-07-28 identifies five NGC4449
science observations:

```text
152390 152392 152419 152431 152433
```

They are bracketed by these eight unresolved-source pointing observations:

```text
152389 152391 152393 152418 152420 152430 152432 152434
```

The existing pointing reduction binds those observations to:

```text
<project>/apts/hero/apt_<obsnum>_matched.ecsv
```

Keep the observation membership and hero APTs fixed for this side-quest. This
isolates the Citlali implementation/configuration change from an APT change.

## Reduction policy

Use TolProj's checksum-verified `phase4.1-v2.1` pointing kit with:

- the current site-selected `citlali-refactor` executable;
- ten fruit-loop iterations, all retained;
- learning and the accepted noise-map empirical products enabled;
- raw pointing maps and Gaussian source fits;
- Wiener filtering disabled; and
- raw and processed timestream products disabled.

The ten-iteration choice matches the complete 108-observation population
evidence. Every unresolved-source array trajectory met the morphology-aware
3% amplitude simulation by iteration 9, with a 1.87% P90 and 3.57% maximum
residual to the retained endpoint. This does not adopt runtime early stopping
or claim absolute photometric transfer.

## Owner-run Unity procedure

Set the project path and verify the authoritative plan:

```bash
PROJECT_ROOT="$HOME/c2025t/2025-C1-COM-01"
test -f "$PROJECT_ROOT/project.yaml"
cd "$PROJECT_ROOT"
"$HOME/tolteca/bin/python" -c \
  "from importlib.metadata import version; print(version('tolproj'))"
tolproj config preflight
```

Before changing anything, confirm that the project still names the expected
source and APTs:

```bash
grep -n -i -A8 -B3 'NGC4449' project.yaml
for obs in 152389 152391 152393 152418 152420 152430 152432 152434
do
  test -r "apts/hero/apt_${obs}_matched.ecsv" || {
    echo "MISSING APT: $obs" >&2
    exit 1
  }
done
```

Preserve the previous Citlali reduction by moving the complete old pointing
workspace out of TolProj's canonical path:

```bash
OLD_POINTINGS="$PROJECT_ROOT/pointings"
POINTINGS_ARCHIVE="$PROJECT_ROOT/pointings_legacy_citlali_20260728"

test -d "$OLD_POINTINGS"
test ! -e "$POINTINGS_ARCHIVE"
mv -- "$OLD_POINTINGS" "$POINTINGS_ARCHIVE"
```

This is a same-filesystem rename, not a deletion. Restore it with the inverse
move if setup is abandoned before the new canonical directory is used.

Generate a fresh refactor pointing workspace:

```bash
tolproj setup-pointing-reductions "$PROJECT_ROOT" \
  --source NGC4449 \
  --pointings-dir pointings \
  --apt-dir apts/hero \
  --refactor \
  --time 48:00:00 \
  --mem 64G \
  --cpus 6
```

Copy `configure_generated_pointing.py` from this bundle to Unity, then run:

```bash
"$HOME/tolteca/bin/python" ./configure_generated_pointing.py \
  "$PROJECT_ROOT/pointings"

"$HOME/tolteca/bin/python" ./configure_generated_pointing.py \
  "$PROJECT_ROOT/pointings" --write
```

The first invocation is a dry run. Both invocations fail unless the installed
kit is `phase4.1-v2.1`, the observation set is exactly the eight obsnums above,
and every observation uses its explicit `<project>/apts/hero` APT.

Validate the generated reduction and inspect the final operator-facing values:

```bash
tolproj validate-reduction "$PROJECT_ROOT/pointings"

grep -n -E \
  'path:|fruit_loops:|enabled:|max_iters:|save_all_iters:|output:' \
  "$PROJECT_ROOT/pointings/71_pointing_runtime.yaml" \
  "$PROJECT_ROOT/pointings/72_pointing_observation.yaml" \
  "$PROJECT_ROOT/pointings/81_pointing_defaults.yaml" \
  "$PROJECT_ROOT/pointings/82_pointing_products.yaml"
```

Submit through TolProj so the selected Citlali executable is frozen while the
job is queued:

```bash
tolproj submit-reduction "$PROJECT_ROOT/pointings"
```

Do not rerun the NGC4449 science reduction until the new pointing products
have passed QA and `tolproj calibrate-pointing-flxscale` has regenerated the
flux-calibrated APTs from the new canonical `pointings/reduced/redu00`
products.

## Local upload command

From the local `citlali-refactor` checkout:

```bash
rsync -av --itemize-changes \
  validation/ngc4449_pointings_refactor_2026-07-28/configure_generated_pointing.py \
  unity_toltec:~/c2025t/2025-C1-COM-01/
```
