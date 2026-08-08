# SCI-ALIGN-001 PTC scan-metadata defect — 2026-08-08

## Classification and scope

This is a distinct confirmed Citlali engineering defect in processed-TOD
product metadata. It is not evidence for or against the approximately 12-ms
SCI-ALIGN-001 timing offset, and it does not alter the completed Beammap maps,
APTs, or split-direction fits.

The affected lifecycle is appending variable-length processed time chunks to
the required PTC TOD NetCDF product in the supported `full` or `mini` output
modes. The appended signal, flags, pointing, and telescope arrays are present;
the persisted scan bounds do not describe their variable-length chunk
extents.

## Unity trigger

The owner-run ObsNum 150819 product
`sci_align_001_naive_full_ptc_singlepass_150819_2026-08-07_retry1`
completed successfully and retained 153,360 PTC samples across 199 scans. A
same-scan join against the naive map reduction failed twice:

```text
ERROR: scan 23 selected/full PTC length mismatch: selected=775 full=606
ERROR: scan 23 selected/full-PTC outer length mismatch: selected=775 full=606
```

The owner then established:

- every full-PTC `scan_indices` row has length 606;
- `raw_scan_indices` also reports the same 606-sample inner support;
- the map run's selected detector TOD and direction registry agree on
  variable scan lengths from 606 through at least 800 samples;
- their one-based scan identities agree; and
- the full-PTC `n_pts` count is consistent with appending the complete
  variable-length chunks, not 199 copies of 606 samples.

Processed PTC output supports only `full` and `mini`; unlike raw RTC output,
there is no supported `full_outer` mode. A diagnostic attempt to request that
mode failed configuration validation before output creation.

## Root cause

`PTCProc::append_to_netcdf` appends every row of `in.scans.data`, beginning at
the current unlimited `n_pts` dimension. It calls
`TCProc::append_base_to_netcdf` with `output_outer_scan=false`.

For every scan after the first, the non-outer metadata branch previously read
the preceding two-element `scan_indices` row and added the current chunk
length to both elements. That recurrence preserves the first scan's length
even when later appended chunks have different lengths. Consequently the
unlimited arrays and scan metadata diverge.

## Narrow repair

For a non-outer append, the persisted bounds are now defined by the same
values that control the NetCDF write:

```text
start = n_pts before append
end   = start + appended row count - 1
```

The outer-output branch is unchanged. The shared helper rejects empty or
overflowing appends. This is a metadata repair; it does not change signal,
flags, pointing, weights, cleaning, mapmaking, or scan selection.

## Local validation

The focused regression uses variable chunk lengths 606, 775, 751, and 767 and
requires exact contiguous append extents. The previous PTC
`FRUITLOOPS_ITER` schema/finalization regression is run alongside it.

```text
citlali::safety::tod_output_scan_metadata.follows_variable_length_append_extents: PASS
citlali::safety::ptc_tod_schema.iteration_field_exists_before_final_header_and_updates: PASS
citlali_cli build: PASS
```

## Remaining Unity validation

One owner-run validation remains:

1. rebuild Citlali at the repair commit;
2. rerun the accepted single-pass `mode: full` PTC configuration in a fresh
   output root;
3. require `scan_indices` and `raw_scan_indices` lengths to equal the selected
   detector-TOD length for each retained scan identity; and
4. rerun the read-only selected-sampling join.

The prior completed full-PTC file remains useful defect evidence but is not an
accepted scan-bound authority.

## Routing

Preserve this record in SCI-ALIGN-001 closeout and route it to the future
SCI-BEAM-001 inbox as required-product metadata debt. Keep it separate from
the earlier `FRUITLOOPS_ITER` creation-order defect and from all physical
timing interpretations.
