# JINC recovery and bounded naïve-versus-JINC comparison

This local preparation records the limited recovery route after CAP-SCIENCE
failed Citlali configuration admission. It changes no application source,
does not modify the completed seven naïve MAP-001 controls, and does not use
or request a full processed-time-chunk capture.

## Decision and scope

The completed MAP-001 controls remain the seven repaired-naïve cases, in their
unchanged order: `P-SEQ`, `P-OMP`, `S-C-SEQ`, `S-C-OMP`, `S-E-SEQ`, `S-E-OMP`,
and `S-X-SEQ`. They are not rerun by this procedure.

The comparison candidates are new, isolated clones of two successful naïve
reductions using the same ordinary exact-candidate executable:

| Comparator | Baseline clone | Observations | Only intentional resolved low-level difference |
| --- | --- | --- | --- |
| `S-E-SEQ-JINC` | `S-E-SEQ` | 152390, 152392; retained pointing support 152389, 152391, 152393 | `mapmaking.method: naive -> jinc` |
| `P-SEQ-JINC` | `P-SEQ` | 152389 | `mapmaking.method: naive -> jinc` |

CAP-POINT is **not** a clean point comparator: its capture route deliberately
installed the full/all processed-time-chunk output overlay, whereas `P-SEQ`
did not. The fresh `P-SEQ-JINC` clone avoids that output difference. CAP-POINT
remains retained evidence; it is neither deleted nor replaced.

CAP-SCIENCE produced no science map. Its failure was the active fruit-loop
S/N gate combined with missing empirical-noise prerequisites. The science
overlay below explicitly retains `noise_maps.enabled=true`,
`n_noise_maps=64`, and `noise_maps.products.enabled=true`, while setting
`timestream.fruit_loops.enabled=false`. Citlali's fruit-loop activation
validation returns before inspecting the S/N gate when fruit loops are
disabled, so the inherited `sig2noise_limit` is inactive rather than a
configuration-admission defect.

## Owner-run preparation (future human action only)

These are owner commands to run only after authorization, from a Unity compute
session. They do not submit a job. Do not enter `set -euo pipefail` in an
interactive compute-node terminal.

```sh
# This is the only Unity processing root for this recovery/comparison.
PROCESSING_ROOT="/work/toltec/commissioning2025-test/2026-ENG-citlali-sci-map-001"
RUN_ROOT="$PROCESSING_ROOT/SCI-MAP-001-UNITY-001-ED2"
CONTROLS_ROOT="$PROCESSING_ROOT"

# Package staging is intentionally separate and read-only during processing.
PACKAGE_CHECKOUT="$HOME/c2025t/2026-ENG-citlali-MAP/citlali-refactor-ed2-package"
PACKAGE="$PACKAGE_CHECKOUT/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb-ed1"
CANDIDATE_BIN="/work/toltec/citlali_dev/citlali_refactor_map_ed28/build_unity_release/bin/citlali"
export TOLPROJ_CONFIG="$RUN_ROOT/tolproj-map-ed28.yaml"

NAIVE_SCIENCE="$CONTROLS_ROOT/SCI-MAP-001-SCIENCE-SOURCE/S-E-SEQ"
NAIVE_POINT="$CONTROLS_ROOT/SCI-MAP-001-POINT-SOURCE/P-SEQ"
JINC_ROOT="$RUN_ROOT/comparators"
JINC_SCIENCE="$JINC_ROOT/S-E-SEQ-JINC"
JINC_POINT="$JINC_ROOT/P-SEQ-JINC"

test -x "$CANDIDATE_BIN"
test -d "$PROCESSING_ROOT"
test -d "$PACKAGE"
test -d "$NAIVE_SCIENCE"
test -d "$NAIVE_POINT"
test ! -e "$JINC_SCIENCE"
test ! -e "$JINC_POINT"
mkdir "$JINC_ROOT"
```

Create each clean clone by copying only its numbered configuration, launcher,
and small support files—not its prior `reduced/` products, logs, or captures.
The numbered fragments are the input authority. The loop rejects an unexpected
top-level regular file instead of silently guessing what belongs in a clone.

```sh
clone_reduction() {
  source_dir=$1
  destination_dir=$2
  mkdir "$destination_dir"
  for item in "$source_dir"/[0-9][0-9]_*.yaml "$source_dir"/02_redu.sh "$source_dir"/README.md; do
    test -f "$item" || { printf 'missing expected source file: %s\n' "$item" >&2; return 2; }
    cp -a "$item" "$destination_dir/"
  done
  for name in bin cal doc; do
    test -e "$source_dir/$name" || continue
    cp -a "$source_dir/$name" "$destination_dir/"
  done
}

clone_reduction "$NAIVE_SCIENCE" "$JINC_SCIENCE"
clone_reduction "$NAIVE_POINT" "$JINC_POINT"
```

Install the final fragments, then reinstall TolProj's local launcher in each
clone. The launcher operation is local metadata preparation only; it does not
freeze or submit a Slurm job. It resolves the candidate executable from the
copied numbered configuration, so stop if either resolved source differs from
the one candidate binary above.

```sh
cp "$PACKAGE/jinc-science-comparator-overlay.yaml" \
  "$JINC_SCIENCE/99_zzzz_jinc_comparator.yaml"
cp "$PACKAGE/jinc-point-comparator-overlay.yaml" \
  "$JINC_POINT/99_zzzz_jinc_comparator.yaml"

python -c 'from pathlib import Path; from tolproj.reduction_runtime import install_reduction_launcher; import sys; [install_reduction_launcher(Path(p)) for p in sys.argv[1:]]' \
  "$JINC_SCIENCE" "$JINC_POINT"

python -c 'from pathlib import Path; from tolproj.reduction_runtime import requested_executable; import sys; [print(requested_executable(Path(p))[0]) for p in sys.argv[1:]]' \
  "$JINC_SCIENCE" "$JINC_POINT"
```

The last command must print `CANDIDATE_BIN` twice. If it does not, stop; do
not edit another numbered file to compensate.

## Post-merge, pre-submit admission and structural-diff check

The checker merges the numbered fragments in their filename order, compares
the resulting `reduce.steps[0].config.low_level` mappings, and writes a
non-overwritable JSON proof. It accepts exactly the one method leaf difference.
For science it also enforces the 64-realization empirical-noise and disabled
fruit-loop requirements that prevent the CAP-SCIENCE error. `tolproj
validate-reduction` separately checks the requested Slurm CPU allocation
against `runtime.n_threads`; it never submits a job.

```sh
mkdir "$JINC_ROOT/proofs"

python "$PACKAGE/scripts/jinc-comparator-check.py" \
  --science \
  --baseline-directory "$NAIVE_SCIENCE" \
  --candidate-directory "$JINC_SCIENCE" \
  --output "$JINC_ROOT/proofs/S-E-SEQ-JINC.config-proof.json"

python "$PACKAGE/scripts/jinc-comparator-check.py" \
  --baseline-directory "$NAIVE_POINT" \
  --candidate-directory "$JINC_POINT" \
  --output "$JINC_ROOT/proofs/P-SEQ-JINC.config-proof.json"

tolproj validate-reduction "$JINC_SCIENCE"
tolproj validate-reduction "$JINC_POINT"
```

This is deliberately a structural/configuration admission check, not a
simulation of a reduction. Citlali has no standalone config-only mode that
validates an observation without entering the pipeline. The first reduction,
if later approved, remains the application-level admission check. A failure at
that point stops this bounded comparison; it does not authorize configuration
experimentation.

## CAP-POINT inspection record

Before treating CAP-POINT as any descriptive auxiliary reference, run the same
checker on its already saved resolved config and the saved `P-SEQ` resolved
config. It will record the full diff; a passing method-only result would be
required before calling CAP-POINT clean. The known full/all PTC output overlay
already means this condition is not expected to pass.

```sh
CAP_POINT="$RUN_ROOT/captures/CAP-POINT/capture-pointing"
PSEQ_MERGED=$(find "$NAIVE_POINT/reduced" -name citlali_merged_config.yaml -type f -print)
CAP_POINT_MERGED=$(find "$CAP_POINT/reduced" -name citlali_merged_config.yaml -type f -print)
test "$(printf '%s\n' "$PSEQ_MERGED" | sed '/^$/d' | wc -l)" -eq 1
test "$(printf '%s\n' "$CAP_POINT_MERGED" | sed '/^$/d' | wc -l)" -eq 1

python "$PACKAGE/scripts/jinc-comparator-check.py" \
  --report-nonconformant \
  --baseline-merged "$PSEQ_MERGED" \
  --candidate-merged "$CAP_POINT_MERGED" \
  --output "$JINC_ROOT/proofs/CAP-POINT-vs-P-SEQ.config-diff.json"
```

That command is expected to exit nonzero because it is deliberately a
method-only comparator checker, while CAP-POINT contains capture-output
differences. Preserve its stderr alongside the two config files as the
inspection record; do not modify CAP-POINT.

## Submission is intentionally not prepared here

No `processed-time-chunk-full-overlay.yaml` is installed in either comparator,
and this preparation does not call `tolproj submit-reduction`, `sbatch`, or
`tolteca reduce`. A separate owner/coordinator decision is required before any
submission.

## Comparison observables and limits

If both comparator reductions are later authorized and exit cleanly, compare
each JINC result only with its named naïve clone:

- raw observation/APT/PPT identity and the frozen candidate executable digest;
- resolved low-level configuration proof and numbered-fragment inventory;
- map WCS, dimensions, coverage/weight support, signal, weight, and empirical
  noise products for arrays a1100, a1400, and a2000; and
- runtime/exit logs and product cardinality.

This is a bounded naïve-versus-JINC behavioral comparison. It does not replace
the seven naïve MAP-001 acceptance cases, establish JINC-contract conformance,
validate a JINC repair, close a finding, or justify MAP-002 closure. MAP-002
has owner-contract/audit work but no JINC repair to validate.
