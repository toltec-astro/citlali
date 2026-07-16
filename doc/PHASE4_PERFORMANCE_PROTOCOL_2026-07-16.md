# Phase 4 Controlled Performance Protocol

## Purpose

This protocol provides a controlled runtime and peak-memory investigation path
without changing Citlali or its deferred build infrastructure. If a campaign is
triggered, Beammap observation 148670 is the first target because Beammap is the
longest supported mode.

The campaign answers a narrow question: on one Unity node, with matched input,
configuration, thread policy, dependencies, and storage, is the candidate
within the accepted runtime and peak-memory budgets relative to the baseline?
It is not an optimization exercise and does not attribute storage incidents or
uncontrolled historical timing differences to code.

## Operational Decision

On 2026-07-16 the project owner accepted the existing Beammap history as
proportionate operational evidence and deferred a dedicated campaign. Twelve
accepted refactor checkpoints span 3,397.522 to 4,215.296 seconds with a median
of 3,594.693 seconds, move in both directions, and show only a 1.9% increase in
the latest adjacent comparison. One earlier 13.0% total-time increase occurred
while mapmaking was 1.3% faster and the change was concentrated in PTC and
diagnostic I/O. Repeated scientific-product validation also passes.

These observations do not establish a controlled OG/refactor runtime ratio,
but they do not show a sustained performance regression. Serializing only
Citlali jobs cannot control unrelated traffic on the shared VAST filesystem.
Requiring eight additional hour-scale reductions without a problem signal
would therefore cost more than the evidence is worth for this deployment.

The wrapper remains the preferred launcher for the next naturally required
Beammap validation so peak RSS and complete performance provenance accumulate
without a dedicated reduction. Run the controlled campaign below only after a
sustained runtime regression, unexplained stage-level slowdown, memory failure,
peak RSS near node capacity, or a material change to a measured hot path. A
suspicious opportunistic result should be confirmed before it is attributed to
code.

## Planned Beammap Corpus Census

After the refactor, the project owner plans to re-reduce the approximately 50
existing Beammap observations with one frozen new-code release. This corpus is
the preferred broad performance characterization because it spans the real
variation in Beammap scan structure, detector population, observing conditions,
and output workload. It is not a prerequisite for refactor closeout.

Before launching the corpus, add a lightweight census manifest and analyzer
that reuse the existing evidence extractor. Retain for every observation:

- observation and input identity, effective-config digest, executable and
  dependency revisions, node, storage identity, and runtime policy;
- external and Citlali wall time, peak RSS, filesystem counters, stage totals,
  success/failure state, and serious log counts;
- available workload descriptors such as scans, samples, active detectors,
  generated maps, and retained products.

Report the distribution and its relationship to workload rather than treating
unlike observations as repeated timing trials. Where a comparable historical
or future reduction exists for the same observation, preserve that pairing.
The resulting corpus becomes the operational baseline for later Citlali
releases and can identify outliers for targeted profiling. Do not add census
machinery to the runtime hot path solely for this purpose.

## Campaign Design When Triggered

- Use one Unity node for the entire campaign. Do not allow the scheduler to
  move individual runs to different nodes.
- Run no baseline and candidate reductions concurrently.
- Use the same Release builds, dependency revisions, data, numbered TolTECA
  overlays, thread count, parallel policy, and output products throughout.
- Confirm the generated low-level configs differ only in the allowlisted
  baseline/candidate input-copy and output paths. In particular,
  `mapmaking.grouping`, detector TOD output, and Beammap iterations must match.
- Run one unmeasured warmup for each role.
- Run at least three measured pairs; five pairs are preferred. Alternate which
  role runs first: baseline/candidate, candidate/baseline, then
  baseline/candidate.
- Retain every measurement. Do not discard an outlier after seeing the result.
  A storage or node incident may invalidate the whole campaign if the reason is
  independently recorded.
- Validate the candidate scientific products with the active Beammap profile in
  addition to this performance campaign.

## Captured Evidence

`tools/baseline/run_performance_case.py` wraps one `tolteca reduce` invocation
with GNU Time and writes `performance_run.json` plus `performance_time.txt` into
the resulting `reduNN` directory. The portable JSON includes:

- command, cwd, UTC start/end, role, pair, warmup/measured status, and exit code;
- executable path, binary SHA-256, `--version` output, and discoverable
  kidscpp/Tula checkout revisions from that executable's build tree;
- host, platform, available CPU affinity, selected OpenMP/SLURM environment;
- external wall time, user/system time, CPU use, filesystem counters, page
  faults, context switches, and peak RSS;
- Citlali version and internal log wall time;
- exact normalized low-level config leaves;
- input paths, sizes, and SHA-256 for files no larger than 100 MB;
- requested/effective/realized runtime policy when provenance is available;
- serious log counts and profile-sidecar stage totals.

The internal Citlali log interval is the primary runtime metric. External wall
time measures TolTECA plus Citlali and is retained as operational context. GNU
Time peak RSS measures the launched TolTECA/Citlali process tree. Filesystem
counters are comparative indicators, not bytes.

## Unity Command

Run the wrapper with the Python from the active TolTECA environment. The tool
must come from the candidate checkout, but it can launch either workspace's
configured `tolteca reduce` command.

```bash
python "$HOME/work_toltec/citlali_dev/citlali_refactor/tools/baseline/run_performance_case.py" \
  --campaign-id phase4-beammap-148670-controlled-v1 \
  --case-id baseline-warmup \
  --role baseline \
  --phase warmup \
  --pair-index 0 \
  --build-type Release \
  --citlali-executable /work/toltec/citlali_dev/citlali/build/bin/citlali \
  --reduced-root reduced \
  --output performance/baseline-warmup.json \
  -- tolteca reduce
```

Invoke it from the appropriate `beammap/citlali` or `beammap/refactor`
directory. Change `case-id`, `role`, `phase`, `pair-index`, and output for each
run. The required order is:

| Sequence | Role | Phase | Pair |
| ---: | --- | --- | ---: |
| 1 | baseline | warmup | 0 |
| 2 | candidate | warmup | 0 |
| 3 | baseline | measured | 0 |
| 4 | candidate | measured | 0 |
| 5 | candidate | measured | 1 |
| 6 | baseline | measured | 1 |
| 7 | baseline | measured | 2 |
| 8 | candidate | measured | 2 |

Pairs 3 and 4 may be added in the opposite/alternating order for the preferred
five-pair campaign.

The wrapper refuses to overwrite evidence, requires exactly one new reduction
by default, and returns failure when the reduction command, directory claim, or
GNU Time measurement is incomplete. The original metadata also remains under
the requested `performance/` path so failed launches retain evidence.

## Offline Analysis

Copy `validation/performance/beammap_campaign_template.json` beside the
downloaded Beammap workspaces. Replace both version tokens and add each attached
metadata path to `runs`, for example:

```json
{
  "metadata": "citlali/reduced/redu00/performance_run.json"
}
```

Then run:

```bash
$HOME/tolteca/bin/python tools/baseline/analyze_performance_campaign.py \
  /path/to/beammap_campaign.json \
  --json-out /tmp/beammap_performance.json \
  --report-out /tmp/beammap_performance.md
```

The analyzer fails an incomplete campaign, mismatched input/config/thread
policy, non-alternating order, missing/failed run, serious log message, wrong
binary version, or absent wall/RSS metric. For a complete protocol it reports
each paired ratio, median and IQR, filesystem ratios, and the largest candidate
profile stages.

## Campaign Budgets

The established wall-time budget is a maximum 5% median candidate regression.
Peak RSS remains a required measurement, but the campaign template deliberately
leaves its limit unset. The analyzer reports a `pending_policy` verdict after a
complete, otherwise passing campaign until that limit is chosen from actual
measurements. The memory decision will consider both relative growth and
absolute growth against the available memory on the supported cluster nodes;
it will not assume that a 5% change is inherently material. A known runtime
budget failure still rejects a campaign even while memory policy is pending.

The current run-owned profiler has no enable/disable control, so nested stage
totals cannot measure its own cost. Campaign evidence records sidecar size and
record counts but does not pretend they are overhead measurements. No profiling
control or dedicated A/B run is required without a performance signal; a future
performance investigation may add one after that interface is approved.

## Local Verification

- All 86 baseline-tool tests pass, including 12 focused campaign/parser tests.
- Full config preflight passes with all 96 config tests and eight compatibility
  fixtures.
- The evidence extractor was exercised against accepted Beammap `redu06`. It
  recovered the Citlali/kidscpp/Tula identities, 529 config leaves, 14 input
  references, the 16-thread runtime policy, 43 profile records across 29 stage
  names, and the 4215.296-second internal interval.
- The analyzer accepts a complete synthetic three-pair campaign, rejects an
  over-budget campaign, and marks missing pairs or mismatched configs/inputs as
  incomplete.
- The live wrapper passed its Unity integration pilot with point `redu67` at
  `7ca0be50c`: both evidence copies were identical, the command and structural
  checks passed, and the record captured 131.08 seconds external wall time,
  110.477 seconds Citlali time, and 908,316 KB peak RSS. The active point
  profile then accepted all three gates with zero config or product changes.
  This pilot qualifies the wrapper; no dedicated Beammap campaign is currently
  required.
