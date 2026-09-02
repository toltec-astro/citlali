# SCI-FRUIT EL-F2 r0.2 — Owner Approval

Scientific owner: Grant Wilson

Decision date: `2026-09-02`

Decision ID: `SCI-FRUIT-EL-F2-INDEPENDENT-POINTING-EARLY-STOP-R0.2`

Status: **Choice A approved; corrected local comparison authorized**

## Exact approval

The scientific owner stated:

> I approve `SCI-FRUIT-EL-F2-INDEPENDENT-POINTING-EARLY-STOP-R0.2` against
> the exact `EL_F2_BUNDLE_MANIFEST_R0.2.md`.

Remote push state was not stated or independently queried and is not
scientific authority.

## Approved object

The manifest at local correction commit `382b2ca7e` is:

| Object | Bytes | SHA-256 |
| --- | ---: | --- |
| `EL_F2_BUNDLE_MANIFEST_R0.2.md` | 2529 | `0ffc2446568b4e70696291c2c46aad2545e0e99be73cb3bd439ddfbfaf8acb88` |

Its seven payload entries retain the exact byte counts and SHA-256 identities
recorded in that manifest. All seven members and the 12 corrected KIDs
fit-report inputs must be reverified before the replacement trajectory.

## Post-approval execution finding

The required reverification passed, but the authorized replacement then
showed that the 12 files had been misclassified. They are processed tune
NetCDFs, while this executable searches for and reads per-network ECSV/ASCII
text fit reports. The replacement stopped before iteration 0. This later
finding does not alter the approved object's identity; it invalidates r0.2 as
an executable input correction.

## Authorized effect

This selects Choice A. It authorizes only:

- replacing the failed pre-iteration-0 first trajectory with the corrected
  local fit-report binding;
- completing the original four valid primary trajectories in the frozen BAAB
  order, with one environmental replacement remaining;
- applying the already frozen scientific and performance analysis; and
- one conditional exact restart replay only if the primary result is
  promising.

## Preserved non-effects

This does not qualify a method or APT, change production defaults, alter the
approved science question or thresholds, permit rerunning an unfavorable
scientific outcome, establish historical superiority, launch Gate D or Stage
B, or authorize Unity work.
