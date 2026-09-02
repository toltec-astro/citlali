# SCI-FRUIT EL-F2 — Text Fit-Report Correction r0.3

Decision candidate: `SCI-FRUIT-EL-F2-INDEPENDENT-POINTING-EARLY-STOP-R0.3`

Status: **owner-review correction; no further run is authorized**

## What happened

The r0.2 files were the right observation and tune scan, but the wrong file
type. They are processed tune NetCDFs. This Citlali build does not read those
files as fit reports; it looks for one matching `.txt` table per detector
network and parses that table as ECSV or ASCII.

The r0.2 replacement therefore stopped before iteration 0. It produced no map,
checkpoint, or scientific result. Its log and partial setup output are
preserved, and both environmental replacements allowed by r0.1 are now used.

## The verified text files

A read-only search found all 12 required text tables in an existing local
development directory. For every network:

- Citlali's observed filename pattern selects exactly one file;
- the file parses as ECSV with the expected 14 model columns;
- its metadata says observation 123424, sub-observation 0, tune scan 1, and
  the expected network; and
- its row count exactly matches the number of tones in the corresponding tune
  NetCDF.

Their paths, sizes, hashes, identities, and row counts are frozen in
`TEXT_FITREPORT_INPUT_INVENTORY_R0.3.md`. Their earlier acquisition provenance
beyond the existing local path is unavailable. That limitation is acceptable
only for this development screen; it would not be acceptable evidence for
method, APT, or production qualification.

The proposed overlay changes only:

```yaml
kids:
  solver:
    fitreportdir: /Users/gwilson/work_toltec/local_data/tone-match-lab/c2025t_tune_text/beammaps/data/
```

It would again be applied after `COMMON_LOCAL.yaml` and before the
trajectory-specific overlay.

## What does and does not change

The scientific question, executable, recurrence, raw data, telescope data,
APT, alpha values, injection, terminal iterations, metrics, thresholds,
single-thread rule, BAAB order, restart condition, and output limits do not
change.

Choice A would explicitly add one final environmental replacement, raising the
total allowance from two to three. It may replace only the still-unexecuted
first valid trajectory. Any further pre-iteration failure ends EL-F2 as
invalid, and no unfavorable scientific outcome may be rerun.

The two failed starts lasted 1.22 and 0.82 seconds and did not reach mapmaking.
They may have warmed file metadata. This will be disclosed with the timing
result. The valid reference pair still follows a complete candidate control,
and the candidate injected trajectory remains last, so the original BAAB
pair-mean timing comparison is retained; no exact cache-equality claim is
made.

## Owner choices

### Choice A — Use the verified local text tables and make one final attempt (recommended)

Approve `SCI-FRUIT-EL-F2-INDEPENDENT-POINTING-EARLY-STOP-R0.3` exactly as
bound by `EL_F2_BUNDLE_MANIFEST_R0.3.md`. This authorizes one final replacement
and, only if it succeeds, completion of the original four valid primary
trajectories and their already frozen analysis. The conditional restart replay
remains allowed only after a promising primary result.

### Choice B — Stop EL-F2 as invalid

Retain both input failures and do not run observation 123424.

### Choice C — Require newly sourced text tables

Pause EL-F2 until the owner supplies or identifies a preferred set of the 12
text tables with stronger acquisition provenance. A new exact inventory and
approval packet would then be required.

No choice here qualifies a method or APT, changes production defaults, starts
Gate D or Stage B, establishes historical superiority, or authorizes Unity
work.
