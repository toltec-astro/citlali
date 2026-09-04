# SCI-FRUIT EL-F11 setup abort r0.1

Test ID:
`SCI-FRUIT-EL-F11-PROSPECTIVE-INFLUENCE-PERSISTENCE-R0.1`

Status: **pre-iteration setup failure; no scientific result and no replay
consumed**

The first local launch exited during restart validation in 0.91 seconds. It
reported that injected-source `start_iteration: 1` did not equal the copied
checkpoint's `next_iteration: 4`. Citlali never entered the FRUIT iteration
loop and produced no iteration directory, map, checkpoint, learning output,
JINC receipt, or target ledger.

The complete failed log and empty reduction lock are preserved at:

`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f11-prospective-influence-r0.1/attempts/r0.1-preflight-failure`

The method-preserving correction is recorded in
`EL_F11_R1_ROUTINE_RESTART_INJECTION_START_REPAIR_2026-09-04.md`. A successor
registration must bind the corrected override and preserved failure evidence
before another launch. No scientific value was opened.
