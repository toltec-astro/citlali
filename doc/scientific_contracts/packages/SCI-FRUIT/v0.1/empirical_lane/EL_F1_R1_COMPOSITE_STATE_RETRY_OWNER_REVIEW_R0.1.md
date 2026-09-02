# SCI-FRUIT EL-F1 — Technical Repair And Fresh Retry r0.1

Decision candidate: `SCI-FRUIT-EL-F1-R1-COMPOSITE-STATE-RETRY-R0.1`

Status: **owner-review proposal; no repair or new run is authorized**

## What happened

The first EL-F1 screen is invalid. Alpha 1.00 completed, but alpha 1.25
stopped before its first feedback-processed iteration. The two permitted
replacement attempts were used to identify state bookkeeping mismatches. No
candidate was scientifically tested, so the result does not count for or
against over-relaxation.

The final stop was caused by copying `MEDRMS` into two places and requiring the
copies to be bit-for-bit equal. One copy was the in-memory value before output;
the other had been written as decimal FITS-header text and read back. They
differed only at about one part in 10^15, but they were not the same bits.

## Recommended repair

Use the existing saved complete map `Q_k` as the single authority for the
newest weights and RMS. Keep the separate relaxed feedback state `F_k` limited
to:

- relaxed signal and kernel planes;
- method, alpha, observation, and completed-iteration identity; and
- exact two-dimensional spatial WCS, map grouping, ordered-plane count, grid,
  and finite support.

On the next iteration, Citlali first reloads the checkpoint-bound ordinary
`Q_k`, including its weights and RMS, exactly as the ordinary recurrence does.
It then replaces only the signal and kernel used as the accepted feedback
model with `F_k`. Selection and all later processing continue to see the
reloaded `Q_k` weights/RMS. The learned operational state remains unchanged.

For restart, the complete causal state is the existing checkpoint-bound
`Q_k`, the separate signal/kernel `F_k`, and the existing learned-state
checkpoint. This removes duplicate RMS authority; it does not discard causal
state or weaken spatial identity checks.

## Required tests before another real-data run

The repair must prove that:

1. alpha 1.00 still uses the unmodified numerical path bit-for-bit;
2. non-unity alpha changes only feedback signal and kernel;
3. the reloaded complete map remains authoritative for weights and RMS;
4. a deliberate spatial WCS, grid, grouping, iteration, support, method, or
   alpha mismatch still fails closed;
5. a FITS `MEDRMS` decimal round trip does not create duplicate-state failure;
6. ordinary non-experimental checkpoints remain byte-schema v3; and
7. experimental restart reproduces three subsequent iterations bit-for-bit
   whenever a candidate reaches the prospective scientific screen.

## Fresh retry

If the tests pass, build once, record the executable SHA-256 before opening
the data, and run the original six trajectories from iteration 0. The earlier
alpha-one products are retained as diagnostic evidence but are not reused,
because the executable was rebuilt after they were created.

The alpha values, injected source, data, operator settings, metrics,
thresholds, sequential one-thread policy, stop rules, and scientific
limitations remain exactly those in the original approved EL-F1 packet. The
fresh retry gets the same maximum six primary trajectories, two genuine
environmental/interruption replacements, 42 iteration passes, 4 hours per
trajectory, 24 hours total, 64 GiB per trajectory, and 250 GiB retained-output
limit. A scientific failure may not be rerun.

## Choices

### Choice A — Repair and rerun all six trajectories (recommended)

Approve the narrow state-authority repair, its focused tests, one frozen
binary, and a fresh execution of the original six-trajectory screen.

### Choice B — Repair and test only

Approve the code and synthetic/FITS round-trip tests, but require a new owner
review before any real-data trajectory.

### Choice C — Stop this recurrence family

Retain the invalid result and do not spend more development runs on these
fixed alpha candidates.

No choice qualifies a method, changes production defaults, launches Gate D or
Stage B, authorizes Unity, or establishes historical superiority.
