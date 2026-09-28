# Reproduce the saved partial assessment

Use the existing local venv and headless rendering. No original input, native
scratch, fitting, uncertainty learning, or observational run is required:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/retained-mpl XDG_CACHE_HOME=/private/tmp/retained-cache OPENBLAS_NUM_THREADS=1 /Users/gwilson/tolteca/bin/python /Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor/validation/rtc_retained_residuals_2026-09-28/retained_report.py --compact /Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor/validation/rtc_retained_residuals_2026-09-28/compact --output /private/tmp/retained-report-reproduction
```

`verify_saved_report.py` verifies byte-identical regeneration of the seven saved
artifacts (six figures), with OS read denial of raw campaigns, protected original
inputs and prior compact records. The new compact copy alone is sufficient.
`test_saved_report.py` checks complete population accounting, unavailable semantics,
operator projection, exact partial B comparators, execution counts and no fictitiousT5.

`retained_compare.py`, `retained_population.py`, `retained_windows.py`,
`retained_direct_map.py` and `retained_old_windows.py` document one-time extraction
from completed M0 and prior savedT5, not a post-cleanup reproduction requirement.
They must not be confused with `run_campaign.py`, whose sole B attempt has failed
and must not be rerun under this order. No candidate preparation exists.

Authority/binding chain: OWNER_DIRECTIVE → PREFLIGHT/RECOVERY → science source
c517444861022f70c95dde95ed13e6a6a259249b → compact prospective/generated/original
bindings and executable digest → LAUNCH/EXECUTION/logs → partial REGRESSION and
NATIVE_REGRESSION → packed report → deterministic renderer. Prior T5 operator
and support are bound to previous NATIVE_REMOVALS hashes. Original raw/tune/APT/
calibration and all previous compact records are protected by PROTECTED_INPUTS.

Cleanup remains unapplied. The exact manifest enumerates only newly generated
native scratch under the one new campaign root. Its guarded `clean` mode requires
explicit new owner approval bound to the manifest, a clean reviewed exact head,
committed preservation, saved-only regeneration and verified originals. No prior
approval may be reused; metadata/FITS/logs remain outside that native deletion.
