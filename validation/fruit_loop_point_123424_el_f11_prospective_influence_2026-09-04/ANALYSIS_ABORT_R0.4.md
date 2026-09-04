# SCI-FRUIT EL-F11 analysis abort r0.4

Test ID:
`SCI-FRUIT-EL-F11-PROSPECTIVE-INFLUENCE-PERSISTENCE-R0.1`

Status: **compatibility failure; no prospective accounting result**

The frozen analysis validated all 35 registered file identities and then
reproduced all twelve ordinary output planes before stopping at the registered
whole-file learning-output gate:

- all nine a1100/a1400/a2000 `signal_I`, `kernel_I`, and `weight_I` planes
  are bitwise identical to the retained EL-F5 iteration-4 reference;
- all three `weight_formal_I` planes are bitwise identical; and
- `learning_iter_4.csv` is not byte-for-byte identical.

The analyzer raised:

```text
ValueError: iteration-4 learning output is not byte-identical
```

It stopped before opening the EL-F11 JINC accounting receipt, target ledger,
or any prospective-persistence value. It wrote no derived result product.

## Read-only diagnosis after the stop

The reference learning file is a cumulative ledger. It contains 4,703 data
rows distributed as follows:

| Absolute iteration | Rows |
|---:|---:|
| 0 | 1,709 |
| 1 | 1,680 |
| 2 | 437 |
| 3 | 438 |
| 4 | 439 |

The restarted replay's learning file contains 439 data rows, all for absolute
iteration 4. The two files have the same ordered CSV header. Selecting the
reference rows whose literal `iter` value is `4` gives exactly 439 rows, and
those rows equal the replay rows in the same order and in every raw CSV field.
The SHA-256 of the canonical ordered row sequence is
`80519723e6edea8b5cf2a88ecbbc3216de38737990a1411139f719fd507948c7`
for both selections.

This establishes that the failed bytewise comparison is caused by the
reference file retaining earlier-iteration rows that a checkpoint restart
does not rewrite. It does not establish that whole-file byte identity passed,
and the registered rule may not be changed without owner approval.

Additional read-only compatibility diagnosis did not open the JINC receipt or
target ledger. It found the reference and replay map-diagnostic NetCDF files
identical in structure, attributes, masks, and values (their whole-file hashes
are also identical). Checkpoint structure and scientific values also match,
with exactly the two registered differences: `creator_version` and
`learning_policy_yaml`; the latter becomes equal under the already registered
historical-default normalization.

The one authorized Citlali replay remains consumed. No rerun is needed to
resolve this ledger-retention-scope question.
