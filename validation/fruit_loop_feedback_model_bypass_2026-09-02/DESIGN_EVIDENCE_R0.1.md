# FRUIT EL-F4 design evidence

Status: **post-EL-F3 hypothesis evidence, not a result of EL-F4**

EL-F3 established that one carried factor-zero a1400 penalty caused the
observation-123424 iteration-5 collapse. Source inspection then established
this ordering in the current development implementation:

1. the exact carried feedback map is projected into the processed timestream;
2. that model bypasses PTC cleaning and is restored before final mapmaking;
3. the final complete raw-observation map is populated; and
4. `mapdiag:raw_obs` uses that complete map to learn a hard penalty for a
   future iteration.

The relevant implementation evidence is in
`include/citlali/core/engine/detail/pointing_run_impl.h`,
`include/citlali/core/engine/detail/pointing_fruitloop_impl.h`,
`include/citlali/core/pipeline/observation_output_execution.h`, and
`include/citlali/core/engine/detail/mapdiag_output_impl.h`. It explains current
behavior but is not scientific authority.

For the four a1400 pixels that made UID 4460 cross the hard threshold at
iteration 4, the exact EL-F1 state update permits recovery of the complete-map
value `Q4` from frozen `F3` and `F4`:

`Q4 = F3 + (F4 - F3) / 1.25`.

The recovered `Q4` agrees with the saved FITS signal plane, after its documented
column orientation, to a maximum absolute difference of
`5.684341886080802e-14`. At the four penalty pixels:

| Internal row | Internal col | Q4 | F3 | Q4 − F3 | (Q4 − F3) sqrt(weight) |
|---:|---:|---:|---:|---:|---:|
| 142 | 280 | 136.190210 | 131.630684 | 4.559526 | 0.745445 |
| 144 | 280 | 117.136401 | 111.273400 | 5.863001 | 1.101863 |
| 142 | 281 | 127.200338 | 119.012586 | 8.187752 | 1.416806 |
| 144 | 281 | 111.970943 | 102.665273 | 9.305671 | 1.759137 |

Thus the large complete-map values are mostly already present in the accepted
feedback state. This motivates testing a feedback-model bypass in the penalty
learner without selecting a new threshold: the existing minimum absolute
robust z of 8 remains fixed.

This calculation does not establish that `Qk − F(k−1)` is the literal
sample-domain residual after projection, response, weighting, and mapmaking.
EL-F4 deliberately names it the **feedback-excluded map-domain diagnostic
view** and treats it as an intentional candidate operation whose scientific
behavior must be measured.
