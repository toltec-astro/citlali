# Bounded diagnosis of existing blank-fit failures

Owner request (2026-09-25): "Carry out a bounded diagnosis of the existing
blank-fit failures if you haven't done so already." No completed diagnosis was
found; the sealed preceding campaign recommends this work but did not run it.

Base: 2aae3c5608e6fff7ce0e55337942a5ea25c51998, clean at entry.
Owned branch codex/map-successor-first-forward-001, existing worktree
/private/tmp/citlali-point-additive-20260923. Main checkout remains untouched.
Governance read: ENGINEERING_GOVERNANCE, TIMESTREAM_SUCCESSOR_GOVERNANCE and
REVIEW_AND_CONFORMANCE, effective via accepted06a3ade51 and77507836325.
Current status, predecessor directive/method/assessment/seal, original fitter,
admission failure rules and trial generation are the authorities. This is a
Tier2 diagnostic evidence follow-up, no application algorithm modification.

Read every archived calibration/validation/stress outcome; replay exactly the
first three failed calibration blank cases per array plus first successful
constrained/unconstrained blank controls and the first failure's paired core
control:18fits. On only the first failed blank case per array, run one isolated
max_nfev6000 probe from the same original start:3additional diagnostic fits.
No tuning based on outcomes, new random fields or full upstream rerun.

Intercept the original solver call to retain its termination, evaluation
counts, finite-state checks, Jacobian singular values, active bounds and sparse
residual-call cost trace. Preserve objective, starts and all original settings;
only the three named probes change the evaluation cap. Probe outcomes are
diagnostic and never replace original failures or enter thresholds/admission.
Keep original pixels, units, FK5 J2000 frame and source/noise models unchanged.

Products: reproducible diagnostic command, authenticated plan/census/cases,
compact numerical assessment and figure. Scientific owner remains existing
POINT diagnostic measurement; there are no new stage interfaces or Engine
state. Read original products, publish a fresh separate diagnosis bundle.
Verify every consumed input against the existing sealed manifest before and
after execution. Stop on mismatch or incomplete exact replay.

Gates: original fit outcome replay, deterministic fixed selection and seeds,
finite-input accounting, no archived data mutation; focused successor tests.
No native/build/config changes, so prior broad baseline/native/config evidence
is retained and no new Unity/Spack/build claim is made. Fresh-context independent
exact-SHA read-only review is required for numerical interpretation and evidence
closure. Commit verified diagnostic/evidence changes locally; no integration,
push, new source-absence rule, fitter repair, threshold recalibration, candidate,
FRUIT application, additional experiment, production change or cleanup.
