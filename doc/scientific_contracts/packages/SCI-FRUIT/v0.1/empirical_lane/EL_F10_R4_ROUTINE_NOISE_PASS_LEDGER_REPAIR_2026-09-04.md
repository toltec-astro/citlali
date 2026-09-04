# SCI-FRUIT EL-F10-R4 — delegated noise-pass ledger repair

Repair identity:
`SCI-FRUIT-EL-F10-R4-NOISE-PASS-LEDGER-REPAIR-R0.1`

Date: `2026-09-04`

Authority:
`SCIENTIFIC_OWNER_ROUTINE_DEFECT_REPAIR_DIRECTION_2026-09-04.md`

The R0.4 analyzer passed exact map neutrality, checkpoint compatibility, and
total-accumulator closure. It then stopped because the target ledger contained
610 rows rather than the registered 305 final-PTC occurrences.

Read-only diagnosis established two complete, contiguous sample-index
sequences from 0 through 304. The first came from the FRUIT noise-only JINC
pass (`run_omb=false`) and contained 271 artificial `center_outside_map`
records plus 34 final-flagged records. The second came from final observation
mapmaking and contained exactly the registered 271 admitted and 34
final-flagged occurrences. The APT contains exactly one a1400 row for UID 4460.

The source defect is that `populate_maps_jinc` created and retained target
sample records even when observation-map population was disabled. Target
`N_t`, `C_t`, and `Q_t` were not contaminated because contribution accounting
is already inside the observation-map branch.

This record delegates only the following diagnostic repair:

1. require `run_omb=true` when identifying a ledger target occurrence;
2. add a focused unit test proving that the same detector/scan/array/map tuple
   is selected for an observation-map pass and rejected for a noise-only pass;
3. run the complete local verification;
4. retain the defective replay and create an isolated replacement root from
   the same iteration-4 checkpoint and unchanged science configuration;
5. freeze the replacement executable, setup, and output registration before
   analysis; and
6. continue the unchanged accounting gates.

The repair changes diagnostic bookkeeping only. It changes no map
accumulator, sample flag, detector weight, FRUIT state, science map,
configuration, gate, numerical bound, region, trigger, or interpretation. The
ordinary products must again match EL-F6 N5 bitwise. Any scientific gate
failure after replacement remains a significant decision point.
