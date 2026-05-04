# REDU40 RTCDiag Triage

This note converts the lightweight `rtcdiag` survey in
`/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu40/rtcdiag_survey_report`
into a concrete obsnum triage for the next contamination/debugging pass.

Scope:
- Array: `a1100`
- Obsnums: full 13-observation GOODS-N survey
- Inputs used:
  - `RTCDIAG_SURVEY_REPORT.md`
  - `rtcdiag_survey_by_obsnum.csv`
  - `rtcdiag_survey_by_obsnum_network.csv`

## Reading The Table

- `max severity`: worst scan-network row severity in the obsnum.
- `step/coherent`: driven mainly by high `max_step_det_frac` and high `max_cm_lowmid`.
- `impulsive`: driven mainly by high `top_slot_event_score` and/or high impulsive detector fraction.
- `masked nw-scans`: how often the current `network_step_mask` already activated.

The intent here is not to freeze a rejection policy yet. It is to rank where the next
human review and next code changes should focus.

## Triage

### Priority 1: Review First

These are the strongest or most diagnostic cases for continued glitch/step development.

| obsnum | dominant family | why it matters | dominant networks |
|---:|:---|:---|:---|
| `151103` | mixed severe | worst case in survey; extreme step/coherent plus real impulsive activity | `nw3`, `nw1`, `nw2` |
| `150784` | step/coherent | strong step-family stress test with limited impulsive contamination | `nw2`, `nw3`, `nw4` |
| `152526` | impulsive | cleanest strong impulsive case; top slot score is extreme | `nw2`, `nw5`, `nw4` |
| `151096` | mixed severe | strong mixed case with high step fraction and large impulsive slots | `nw3`, `nw2`, `nw1` |

### Priority 2: Review Next

These are important but slightly less clean as first-pass stress tests.

| obsnum | dominant family | dominant networks |
|---:|:---|:---|
| `152524` | mixed moderate, impulsive-leaning | `nw1`, `nw3`, `nw2` |
| `150792` | step/coherent | `nw2`, `nw4`, `nw3` |
| `151928` | impulsive | `nw1`, `nw5`, `nw4` |
| `151094` | mixed severe | `nw2`, `nw4`, `nw5` |
| `152286` | mixed severe | `nw2`, `nw3`, `nw4` |

### Priority 3: Control / Lower Priority

These still contain contamination, but they are better suited as controls than as first stress tests.

| obsnum | role | dominant networks |
|---:|:---|:---|
| `151930` | familiar moderate control | `nw5`, `nw4`, `nw0` |
| `151937` | milder step/coherent | `nw0`, `nw3`, `nw4` |
| `152294` | milder mixed | `nw4`, `nw1`, `nw5` |
| `152533` | moderate but less diagnostic than the cases above | `nw3`, `nw5`, `nw4` |

## Family Notes

### Step / Coherent Branch

The strongest current step/coherent branch members are:
- `151103`: especially `nw3`, then `nw1`
- `150784`: especially `nw2`, then `nw3`
- `150792`: especially `nw2`
- `152286`: especially `nw2` and `nw3`

These are the best obsnums for:
- testing step/slow-jump classification
- testing whether the current `network_step_mask` is firing on the right networks
- testing whether network-window masking is enough, or whether detector-local slow-jump handling is needed

### Impulsive Branch

The strongest impulsive branch members are:
- `152526`: dominated by `nw2`
- `151928`: dominated by `nw1` and `nw5`
- `152524`: dominated by `nw1`

These are the best obsnums for:
- evaluating compact event detection and event-action policy
- deciding whether the next runtime change should be impulsive interpolation, masking, or weighting

## Immediate Next Work

1. Use `redu49` and upcoming `redu50` map products to check whether the worst `rtcdiag`
   cases are also the ones producing the clutteriest filtered maps.
2. Keep the first contamination-debugging focus on:
   - `151103`
   - `150784`
   - `152526`
   - `151930` as control
3. If the next runtime change targets step-like contamination, use:
   - `151103`
   - `150784`
   - `152286`
4. If the next runtime change targets impulsive contamination, use:
   - `152526`
   - `151928`
   - `152524`

## Policy Read (Tentative)

Current survey evidence suggests:
- whole-obs rejection should remain a last resort
- step-dominated cases are better handled by masking/subsegmentation than by discarding the obs
- impulsive-dominated cases should be attacked detector/event-locally, not by whole-scan rejection

That policy should still be confirmed against the coadded map products from the new faster-WF run.
