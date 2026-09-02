# FRUIT EL-F2 second-attempt input failure

Status: **invalid pre-iteration attempt; replacement allowance exhausted**

After exact r0.2 approval and input reverification, the authorized replacement
of the first scheduled trajectory began at 16:35:44 local time on 2026-09-02.
It stopped before iteration 0 after 0.82 seconds.

The r0.2 inventory correctly identified the files' observation, tune scan,
networks, hashes, and NetCDF dimensions, but incorrectly classified the
processed tune NetCDFs as loadable Citlali fit reports. This executable derives
a per-network `cal_file` regular expression from the raw metadata and searches
for a matching `.txt` file. It found none in the r0.2 directory, emitted one
critical message for each of the 12 networks, and then failed closed because
the gap-aligned RTC KIDs input contained NaN values.

No map FITS product and no restart checkpoint was created. The partial output
and complete outer log are preserved at:

`/Users/gwilson/work_toltec/local_data/fruit-development/point-123424/fruit-injection-development/el-f2-early-stop-r0.1/attempts/attempt-02-missing-text-fitreports`

Evidence:

- outer log bytes: 25,812;
- outer log SHA-256:
  `87abc6b8915ced7c82477b2c7ff17008dc975f5e9fbabd5286f0a83431615476`;
- error/critical messages: 12, all missing text-fit-report messages;
- maximum resident set size: 265,895,936 bytes; and
- retained attempt tree: 9,972 KiB at inspection time.

The source behavior is visible in
`include/citlali/core/engine/detail/kidsproc_metadata_reduce_impl.h`: the
loader uses `cal_file` to search `fitreportdir`, then parses the selected file
as ECSV or ASCII. The failed file-type assumption was in the experiment
packet, not in the executable.

This is an environmental input-binding failure, not a scientific result. It
uses the second and last replacement allowed by r0.1. All four valid primary
trajectories remain unexecuted, and no further run is authorized.
