# Reproduction and boundaries

Worktree: `/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor`.
Branch: `codex/point-speed-pca-20260927`. Parent completed work: `11b0c313cffe6cc0583b7a9e70dcfa98e3137746`.

The authorized campaign is finished. These commands only analyze saved evidence;
they do not authorize another campaign or restore deleted native arrays.

```sh
POINT_REPO=/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor
POINT_RECORD="$POINT_REPO/validation/point_speed_pca_2026-09-27"
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point-speed-mpl XDG_CACHE_HOME=/private/tmp/point-speed-cache \
  "$HOME/tolteca/bin/python" "$POINT_REPO/tools/timestream_successor/report_point_speed_science.py" \
  --root "$POINT_RECORD/compact" \
  --retained "$POINT_REPO/validation/ptc_rank_response_2026-09-26/compact" \
  --report-dir /private/tmp/point-speed-saved-check
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point-speed-mpl XDG_CACHE_HOME=/private/tmp/point-speed-cache \
  "$HOME/tolteca/bin/python" "$POINT_REPO/tools/timestream_successor/plot_point_speed.py" \
  --root "$POINT_RECORD/compact" --figure-dir /private/tmp/point-speed-saved-figures
```

The science reader loads retained checkpoint maps, all-cycle measurements,
M0 noise operands and already-completed conditional endpoint scales. It evaluates
only the existing ideal-addition measurement where needed. It does not learn PTC,
reprocess original inputs, or run new uncertainty companions. `extract_point_speed_native.py`
was used before compaction to extract signed spectra, support identities, objectives,
basis-projector differences, acquisition duration and stage timing. Its output is
preserved; it cannot and need not rerun after removal of native arrays.

The owner-facing assessment is rendered from the saved machine-readable records.
`render_point_speed_report.py` can rebuild the compact report in a disposable copy
of the compact root; do not mutate the sealed record in place. Its Phase C assertion
prevents changing the completed disposition. Timing intervals are not synthesized
from solver time. First-run stage comparisons and all repeated raw timings remain
separate. The full-campaign storage-walk confound is material and must accompany
any use of these timings.

A science used e28f6630fffaba77f78fc2c2cb883c7345ad92f4, tree
 a4ede0dabf25b69f61a0a0e52e00424511963297. B science used
08f81382d96afb0aad6bc1ed40543322887729b9, tree
09005e5f5d36be6ae00267e3341adab9ee080ac6. Clean exact-SHA CLI versions and independent
preflight dispositions are retained. The A timing report was committed at
b7746e3ca6f1615039a2b9d4a3aabe2df62a2a7f before B implementation.

Local supplemental gates: 60 focused Python tests, 37 focused native CTest cases,
207 baseline tests plus137 subtests, full config preflight. Independent B reviewer
reproduced22 driver,12 PTC and8 FRUIT tests. The local dependency realization is
AppleClang/Homebrew; Unity/Spack qualification was not attempted. The unrestricted
CTest invocation from Phase A is explicitly not a pass: optional excluded targets
were not built. The established focused test selection passed. The initially
mistyped optional native target during B verification was corrected before science.
The launch-setup-only incident in Phase A consumed no trajectory or parent attempts.

The original frozen tag `timestream-successor-bright-point-base-v1-20260926` and
`data/development/bright_point_base_v1/` remain unchanged. The original user's dirty
checkout was not edited. No push, merge, RTC change, automatic stopping or production
promotion occurred.

## Preservation and cleanup

`COMPACT_MANIFEST.json` binds each retained file and original encoding hash.
`HASH_ONLY.json.gz` binds redundant strict-regression-equivalent products and
non-checkpoint products. Every cycle retains numerical summaries; necessary
N0/1/3/5/10 science maps and response maps are compressed without loss. Native
receipts preserve signed spectra and support identities. Old rank controls are
referenced in their already committed record rather than copied.

`NATIVE_REMOVALS.json.gz` is an exact enumerated plan for detector-time payloads
within this campaign alone. `PROTECTED_INPUTS.json` binds all unique original,
tune, APT, calibration and profile dependencies. Cleanup requires a clean exact
committed tree, independent final-review pass at that SHA, no active campaign
process, reverified compact hashes, reverified payload hashes and protected-input
hashes. The cleanup code deletes only those files, never directories or inputs.
`CLEANUP.json` records the applied plan and post-deletion protection checks.
Non-array metadata outside the compact record is left in place; this is not a
repository-wide purge.

Saved-only isolation passed: the OS sandbox denied the original campaign tree and
all31 protected dependencies, a positive denial control confirmed enforcement,
and the regenerated scientific records matched exactly. All four figures rendered.
Compact evidence totals163,803,552bytes (164MB decimal); the exact native removal
plan covers244,212files /13,223,564,416bytes. Six synthetic cleanup guard tests pass.
Final exact-SHA review passed at cec7917ef; the exact native plan was then fully
rehash-verified and removed. All31protected dependencies remain unchanged.
