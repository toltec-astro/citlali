# SCI-FRUIT EL-F2 early-stop screen result

Primary classification: **does_not_replicate**

This is development evidence only and is not a method qualification.

## Terminal comparison

| Array | Reference recovery | Candidate recovery | Reference annular residual | Candidate annular residual | Checks |
| --- | ---: | ---: | ---: | ---: | --- |
| a1100 | 0.867388 | 0.869745 | 0.00025877239 | 0.00030476192 | FAIL — terminal_major_width_within_limit, terminal_minor_width_within_limit, terminal_annular_residual_within_limit |
| a1400 | 0.896659 | 0.822828 | 0.0029607378 | 0.021474106 | FAIL — terminal_recovery_error_within_allowance, terminal_major_width_within_limit, terminal_minor_width_within_limit, terminal_centroid_within_limit, terminal_annular_residual_within_limit, terminal_kernel_residual_within_limit |
| a2000 | 0.731317 | 0.734453 | 0.0034916855 | 0.0041373778 | FAIL — terminal_major_width_within_limit, terminal_minor_width_within_limit, terminal_centroid_within_limit, terminal_annular_residual_within_limit |

## Performance

- Reference pair-mean wall time: 201.610 s
- Candidate pair-mean wall time: 172.175 s
- Candidate wall-time improvement: 14.600%
- Ten-percent target: PASS

## Follow-up

- Exact restart replay required: no
