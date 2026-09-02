# FRUIT EL-F2 independent-pointing early-stop proposal

Status: **prepared for owner review; not authorized for execution**

Test ID: `SCI-FRUIT-EL-F2-INDEPENDENT-POINTING-EARLY-STOP-R0.1`

The proposed comparison uses observation 123424 as a date-independent
replication of the early-stop hypothesis suggested by EL-F1 on observation
152389.

## Frozen prospective comparison

- reference: `alpha = 1.00`, iterations 0 through 6;
- candidate: `alpha = 1.25`, iterations 0 through 5;
- one injection-disabled and one centered 100 mJy/beam injection-enabled
  trajectory for each method;
- injection begins at absolute iteration 1 in all arrays;
- candidate iteration 5 is compared with reference iteration 6;
- execution is sequential with one configured thread; and
- all new products are confined to
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-123424/fruit-injection-development/el-f2-early-stop-r0.1`.

The base configuration is
`../fruit_loop_population_stage_b_2026-07-26/citlali_rc1_fruitloops10_o123424.yaml`.
`INPUTS_LOCAL.yaml` replaces only its Unity input paths with the exact local
files in `INPUT_INVENTORY_R0.1.md`. `COMMON_LOCAL.yaml` suppresses optional
timestream output and fixes one thread. The four variant overlays differ only
in output path, alpha, injection switch, and the predeclared terminal
iteration.

No file in this directory authorizes a run. Authorization requires owner
approval against the exact bundle manifest in the SCI-FRUIT empirical lane.
