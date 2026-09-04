# FRUIT EL-F10-R1 execution result r0.2

Test ID: `SCI-FRUIT-EL-F10-R1-COMPATIBILITY-NORMALIZATION-R0.1`

Result: **repaired compatibility passed; frozen analysis reader failed before
accounting values were read**

## Plain answer

The owner-approved checkpoint repair worked exactly as intended. The 21
registered file hashes passed, the only observed checkpoint differences were
`creator_version` and `learning_policy_yaml`, and the two learning policies
became exactly equal after the single authorized missing-key normalization to
`pre_cleaning`. The twelve ordinary map planes again matched bitwise.

The frozen analyzer then opened the diagnostic receipt and tried to verify its
schema name. The NetCDF library returned that name as an ordinary Python
`str`. The helper handled byte strings and NumPy scalar objects, but attempted
`.item()` on this already-native string and raised:

```text
AttributeError: 'str' object has no attribute 'item'
```

The exception occurred on `schema_identity`, before the analyzer read any
total or target `N`, `C`, or `Q` plane and before it opened the target-sample
ledger. It wrote no result, table, component map, or figure. Under R1's exact
stop rule, the analysis was not patched and rerun.

## Gates completed

- The R1 owner authorization was recorded against bundle SHA-256
  `ee353ac67630a77ab1727084475eb6db86ca46d583bd43c16d06e3fe52d6a217`.
- `REGISTRATION_R0.2.yaml` froze 21 files after the replay and before the
  accounting files were opened; all 21 size/hash checks passed.
- An independent pre-accounting check required and observed exactly
  `creator_version` and `learning_policy_yaml` as the changed checkpoint
  values.
- The normalized old and new learning policies matched exactly.
- The frozen analyzer passed all nine ordinary science-map comparisons, all
  three formal-coefficient comparisons, and the repaired checkpoint gate.

No scientific accounting, sample-ledger, reconstruction, error-bound,
support, regional, trigger-pixel, or descriptive result exists yet.

## Bounded repair

The required code repair is mechanical and does not change a scientific gate:

```python
if isinstance(value, bytes):
    return value.decode()
if hasattr(value, "item"):
    return value.item()
return value
```

A focused test must cover native `str`, `bytes`, and NumPy numeric scalar
returns. The repaired tool and its tests must pass before a new SHA-256 is
frozen. A new registration must rebind the unchanged external files and the
repaired analyzer before one analysis retry.

No Citlali replay is needed or authorized. The receipt and target ledger
remain byte-identical to the R1 registration.

## Boundaries

This result does not determine UID 4460 leverage, explain the arcs, judge a
detector, select a safeguard, change a penalty or threshold, alter FRUIT or
JINC, qualify a method, authorize another reduction, launch Gate D or Stage B,
or authorize Unity activity.
