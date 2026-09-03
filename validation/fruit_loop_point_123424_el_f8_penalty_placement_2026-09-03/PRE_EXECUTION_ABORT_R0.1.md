# EL-F8 R0.1 Pre-execution Abort

## Status

`REGISTRATION_R0.1.yaml` did not produce an EL-F8 scientific replay.  The
first scheduled trajectory (`c5-current`) stopped during exact-restart policy
validation, before iteration 5 began and before any map product was written.
The remaining three trajectories were not started.

This is retained as a pre-execution compatibility failure.  It is not an
environmental replacement, a scientific result, or evidence about penalty
placement.

## Failure identity

- attempted trajectory: `c5-current`
- registered source commit: `ccb67a99257fc9fba82d25346e85503363673651`
- registered executable SHA-256:
  `7190abe12c092cc11314a89673a2840f810fd906915e2636e41ffe196b8754a0`
- exit status: `1`
- wall time reported by `/usr/bin/time -l`: `0.77 s`
- peak resident set size reported by `/usr/bin/time -l`: `47,677,440 bytes`
- log path:
  `/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f8-penalty-placement-r0.1/logs/c5-current.log`
- log size: `14,563 bytes`
- log SHA-256:
  `207a26366bc53a5fee7808e7edaaf62c34d49ad184fe9e07d9ff39187abaf80f`

The output root contains only the acquired `.citlali-reduction.lock`; it has
no `redu05` directory or scientific product.

## Cause and bounded repair

EL-F8 added
`map_pixel_outlier_detector_exclusion_application` to the serialized learning
policy.  The frozen iteration-4 checkpoint predates that field.  The R0.1
executable compared serialized policy strings exactly and therefore rejected
the historical checkpoint even though the requested `pre_cleaning` value is
the field's explicit legacy default.

The bounded repair may normalize absence of this one field to `pre_cleaning`
when checking restart compatibility.  An explicit or requested
`pre_mapmaking` value must remain a policy mismatch.  No recurrence,
selection threshold, detector record, mapmaking operation, or registered
scientific comparison may change.

R0.1 remains immutable evidence of the abort.  A new registration revision
must identify the repaired source and executable before staging or execution.
