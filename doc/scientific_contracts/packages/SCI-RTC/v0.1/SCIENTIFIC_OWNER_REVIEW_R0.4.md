# SCI-RTC v0.1 scientific-owner review and r0.4 decisions

Date: `2026-08-18`

Status: binding targeted correction direction. The supplied review source has
SHA-256
`2298d9b801a2213bf8327f83abe5e0a6aeb3eca2c1398bafb4ec106a9972eba4`.

## Review disposition

The scientific owner approved the r0.3 bounded iterative notch-plan
architecture and rejected another broad authorship round. R0.4 is confined to:

- separating refinement attempts from accepted plans and defining the initial
  evaluation product;
- correcting phase-zero selection to read the final pre-decimation stream;
- using Learn--Resolve--Apply in the title;
- making `x`/`xs` uniformly raw, uncalibrated detector `Delta f/f` and giving
  calibrated detector signal a distinct downstream identity;
- adding the role-specific RTC-plan matrix; and
- applying the two owner decisions below.

## Approved scientific-owner decisions

1. RTC remains raw `Delta f/f` through compatible donor replacement, temporal
   conditioning, and phase-zero sampling. A valid
   `flxscale_q/flxscale_d` ratio is raw donor convention transfer, not absolute
   calibration. SCI-CAL applies absolute `flxscale` and target-atmosphere
   correction only after the complete RTC bundle.
2. Only directly selected ALIGN-synthesized or RTC-replaced occurrences are
   universally excluded as independent detector measurements. RTC preserves
   cause-bearing transitive influence over full support; each downstream
   consumer owns its declared eligibility policy for noncenter influence.

No implementation, audit, test, reduction, or production evidence is an
authorized r0.4 input. The review and decisions do not establish conformity,
validation, science qualification, or readiness.
