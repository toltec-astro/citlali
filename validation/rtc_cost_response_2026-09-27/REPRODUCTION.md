# RTC campaign reproduction and evidence

Completed source: A first REF at `3bb6444ab5e97265cbcb8f197994a52603b481ae`;
remaining A at `d897dc0014da3c8d007fb74dff1e259d9abdf8d9` (comparator repair only).
B at `a0f7345fc0c0501e34e877f91104c4a78e61daf3`, tree
`af90af9bfc7915bf612ac25cd0a85ca7face78e0`. Native executable version embeds B's
revision; KIDs reports `04088da-dirty`, unchanged. Local AppleClang/Homebrew
Release is supplemental evidence, not Unity qualification.

## Saved-only report

```sh
cd /Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/rtc-report-mpl XDG_CACHE_HOME=/private/tmp/rtc-report-cache \
  $HOME/tolteca/bin/python tools/timestream_successor/report_point_rtc.py \
  --root validation/rtc_cost_response_2026-09-27/compact \
  --output /private/tmp/rtc-saved-report
$HOME/tolteca/bin/python validation/rtc_cost_response_2026-09-27/verify_saved_report.py
```

The report reads only compact JSON/NPZ, performs no reduction, new fit, or endpoint
ensemble. The isolation check denies the original campaign tree and all 31
protected inputs with a positive denial control, and reproduces report text,
JSON and two figures. Full-population native diagnostics were extracted before
cleanup by `point_rtc_diagnostics.py` and `point_rtc_response.py`; their NPZ/JSON,
recipe, source masks, denominator counts, windows, events and coherence remain
in `compact/report/A`. Native extraction is not a saved-only reporting step and
cannot run after scratch cleanup. Do not regenerate science automatically.

## Evidence index and execution identity

- `compact/FAMILY-A.json`, `FAMILY-B.json`, `FAMILY-C.json`: completed versus
  deliberately unselected disposition. Exactly 9 parents/trajectories/endpoints,
  45 feedback applications, 1 separate RTC capture, no replacement trajectories.
- Each run's `CAMPAIGN.json`, timing/storage/attempt/command records and frozen
  dependencies bind original inputs, source, configuration and executable.
  Parent and branch state is private; r/input validity is unchanged by injection.
- `compact/report/timing.json` and `timing-recipe.json`: all repeat values,
  surrounding phases, overlap bounds, process-wide CPU and sampled RSS. No sum
  of parallel wall times is labeled latency. Fresh processes; OS caches not
  flushed; no concurrent science, build or report during timed repetitions.
- `compact/report/resources.json`: actual footprint separately from conservative
  allocation charge and cumulative writes. Limit 20 GB, minimum free 2 GB.
- `compact/report/A/source-responses.json`, `native-source-accounting.json`:
  exact native CAL-only inverse mapping, own-REF ideal support, M0/T5 measurements,
  conditional RTC-projected fits, raw/background aperture terms and support.
- `compact/report/postRTC-controls.json`: prior downstream control only.
- `compact/B-REF-rep*/NATIVE_REGRESSION.json` and `REGRESSION.json`: all three
  byte-exact RTC/CAL/PTC/discrete gates and exact downstream planes/noise/endpoints.
  B source response is inherited, not independently remeasured.
- `COMPARISON_REPAIR.md`, `PREFIX_RECOVERY.json`, sealed `prior-prefix`:
  failed whole-FITS comparison was preserved; authenticated T2/T4 originals
  prove checksum metadata differences with all scientific planes exact.
  Continuation reused completed first REF. No relaxed tolerance or science retry.
  Historical comparator scope text says T10; actual recorded prefix ends T5.
- `UNCERTAINTY_WARNING_CHECK.json`: retained NumPy warnings, independently exact
  finite saved operators/scales without the warned contraction. No asserted
  root-cause diagnosis or general library qualification. Native error count zero.

## Validation and lifecycle

Preflight repairs, independent reviews and their exact-source bindings are kept
beside this file. B gates: 77 focused native tests, 39 Python tests, 207 baseline
tests and complete config preflight. Six cleanup fixtures check ownership,
protected paths and rejection behavior. Final saved-report isolation and visual
inspection supplement those gates; no new native change followed B execution.

`COMPACT_MANIFEST.json` retains 1,608 objects / 90,615,958 bytes. Hash-only records
cover redundant verified repeats and intermediate maps; all-cycle measurements
remain. `NATIVE_REMOVALS.json.gz` enumerates only this campaign's 117,673 inactive
native payloads / 7,907,658,328 bytes. `compact_record.py clean --expected-head`
requires a clean committed exact review candidate, explicit final-review pass,
committed manifests, preserved compact hashes and unchanged protected inputs.
It verifies the native deletion inventory once, then removes exactly those files
and rechecks protection. Review must not duplicate that exhaustive scratch hash.
`CLEANUP.json` records completion; the immutable deletion plan remains unapplied
as the original plan, with actual execution recorded separately.

The old cold-rank10/T10 profile and tag are unchanged. No push, merge, Unity,
input transfer, new observation, production activation or automatic follow-on.
