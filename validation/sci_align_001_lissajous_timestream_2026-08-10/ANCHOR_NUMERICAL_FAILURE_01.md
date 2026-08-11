# Real-anchor numerical gate failure 01

ObsNum 150818 passed the frozen input, common-support, and zero-lag
coordinate-reconstruction gates. All 12 scans, 2,294 eligible detector UIDs,
six a1100 networks, 3,372 common-support rows, and 789,241 scored values were
retained. The maximum absolute zero-lag coordinate residual was exactly
0 arcsec.

The first frozen implementation nevertheless failed its numerical-stability
gate. Both the scalar-lag and joint optimizers returned their initial
`tau=0` exactly. This was inconsistent with two value-independent numerical
diagnostics:

- the objective at the same fixed source position changed from
  110684.096885 at 0 ms to 110428.287096 at +4 ms; and
- the separately projected first-order derivative estimate was +4.174 ms.

The failure was caused by placing tau in seconds (a bounded coordinate of
order 0.01) beside position and hysteresis coordinates in arcseconds (order
1). The finite-difference L-BFGS-B calculation did not move tau from its
starts on the real product. The repair changes only the optimizer coordinate
for tau from seconds to milliseconds. `parameter_dict` converts it back to
seconds before any coordinate interpolation, support selection, source-model
evaluation, or result reporting. No real fitted value was used as an
acceptance target, and no mask, source window, beam, baseline, bound, or input
identity was changed.

Before the anchor may be examined again, the complete synthetic suite must
pass and the repaired implementation, test, and protocol identities must be
refrozen.
