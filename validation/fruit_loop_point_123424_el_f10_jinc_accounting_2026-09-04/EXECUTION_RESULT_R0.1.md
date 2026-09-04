# FRUIT EL-F10 execution result r0.1

Test ID: `SCI-FRUIT-EL-F10-TARGETED-JINC-ACCOUNTING-R0.1`

Result: **compatibility gate failed; target accounting was not interpreted**

## Plain answer

The new diagnostic did not change any ordinary map values we tested. All nine
registered signal, kernel, and empirical-coefficient planes, plus all three
unscaled formal-coefficient planes, match the historical EL-F6 N5 products
bit for bit.

However, the pre-registered checkpoint rule was stricter than that. It allowed
only the executable-version field to differ. The new checkpoint also writes
one learning-policy field that the older EL-F6 checkpoint omitted:

```yaml
map_pixel_outlier_detector_exclusion_application: pre_cleaning
```

The older absence and the new explicit value describe the same historical
placement according to the current restart compatibility rule and the prior
EL-F8 evidence. But EL-F10 r0.1 did not register that normalization before the
replay. The analysis therefore stopped exactly where the approved protocol
said it must stop. We have not opened or interpreted the target UID's `N`,
`C`, or `Q` values, and we cannot yet answer the scientific leverage question.

## What passed before the stop

- All 19 registered input, software, method, and comparison hashes passed.
- All three arrays matched EL-F6 N5 bitwise in `signal_I`, `kernel_I`, and
  `weight_I`: nine of nine ordinary science planes.
- All three `weight_formal_I` planes matched bitwise.
- The FITS grids and WCS identities used by the exact comparison matched.
- The replay completed one iteration, returned exit code zero, and emitted no
  error or critical log records.
- The checkpoint has the same global attributes, dimensions, 77-variable set,
  variable structures, and values except for `creator_version` and
  `learning_policy_yaml`.
- The entire `learning_policy_yaml` difference is the one explicit
  `pre_cleaning` line shown above.

These facts are encouraging evidence that the diagnostic is observational,
but they do not override the failed registered gate.

## Execution cost and retained products

The local sequential replay took 33.29 seconds and reported a maximum resident
set size of 922,501,120 bytes. The isolated reduction tree is 63,848 KiB and
the external log directory is 516 KiB, comfortably inside the registered
one-hour, 64-GiB, and 8-GiB limits.

The diagnostic receipt and target-sample ledger were written successfully and
are preserved unopened for a possible separately approved repair analysis:

- `toltec_commissioning_pointing_123424_jinc_accounting.nc`: 5,485,571 bytes,
  SHA-256
  `010d8cbac8b4031223b84b3ef4e6a2e77d52a5a9d0b8e673a735e0e97e1c9cfc`;
- `toltec_commissioning_pointing_123424_jinc_accounting_target_samples.ecsv`:
  100,681 bytes, SHA-256
  `8e9e94178bbe1099344ca9be9e1e3e7ba8048dbc016ebd2ace83c885cca5d08c`.

They remain diagnostic-only, non-science, and non-checkpoint products. Their
existence and file hashes are not scientific interpretation.

## Why this is a real stop rather than a failed algorithm

The mismatch is not evidence that the JINC diagnostic changed the science
maps: the exact map tests passed. It is a registration defect caused by using
a newer executable against an older checkpoint without admitting a known,
semantically default serialization difference.

The same issue was encountered and explicitly repaired during EL-F8. Current
restart loading normalizes historical absence of this one field to
`pre_cleaning`, and the current enum default is `pre_cleaning`. Those are
implementation and prior-development evidence, not permission to relax a
gate after seeing an output. The approved EL-F10 owner review says a
compatibility failure requires a revised packet.

## Recommended next decision

Prepare a no-replay compatibility-repair packet. It should bind the exact
retained EL-F10 output hashes before opening accounting values and change only
the checkpoint comparison so that absence of
`map_pixel_outlier_detector_exclusion_application` is equivalent to an
explicit `pre_cleaning` value. Every other checkpoint value must retain the
original exact-equality requirement; all map-neutrality, accumulator-closure,
sample-ledger, binary64, support, scope, and claim gates must remain unchanged.

If the scientific owner approves that exact repair, the frozen analyzer can
continue on this already completed replay. No additional reduction is needed
or justified.

## Boundaries

This result does not determine UID 4460 leverage, explain the arcs, judge a
detector, select a safeguard, change a penalty or threshold, alter FRUIT or
JINC, qualify a method, authorize another replay, launch Gate D or Stage B, or
authorize Unity activity. The external EL-F6 and EL-F8 products were not
modified.
