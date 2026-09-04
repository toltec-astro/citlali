# SCI-FRUIT EL-F11-R1 — Routine restart injection-start repair

Date: `2026-09-04`

Status: **narrow setup-defect repair under standing owner direction; no
scientific method or scope change**

The first EL-F11 launch stopped before entering any FRUIT iteration. The
restart guard reported:

> fruit-loop injected-source restart requires start_iteration 1 to equal
> checkpoint next_iteration 4

The copied completed iteration-3 checkpoint correctly identifies absolute
iteration 4 as the next iteration. The initial EL-F11 override had repeated
the historical fresh-run activation value `start_iteration: 1`. For a restart,
Citlali requires that field to name the next absolute iteration so the already
active injected source is applied to the resumed step. The accepted EL-F10
restart used the same convention (`5` for a completed iteration-4 checkpoint).

The repair changes only the EL-F11 override's `start_iteration` from `1` to
`4`. It does not change whether the source is enabled, its position or
amplitude, the incoming checkpoint, any scientific or learning state, RTC,
PTC, mapmaking, masks, weights, filters, target accounting, gate, bound,
metric, interpretation, resource limit, or claim limit.

The failed attempt produced no iteration directory or scientific product. Its
log and empty reduction lock are preserved under:

`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f11-prospective-influence-r0.1/attempts/r0.1-preflight-failure`

This is a simple configuration-binding defect within the already authorized
test. The owner's standing routine-defect direction authorizes correction,
documentation, refreezing, and continuation without another approval. The
single authorized scientific replay remains unused.
