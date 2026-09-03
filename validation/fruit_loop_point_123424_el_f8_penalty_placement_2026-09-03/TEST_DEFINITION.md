# FRUIT EL-F8 penalty-placement decomposition

Test ID:
`SCI-FRUIT-EL-F8-PENALTY-PLACEMENT-DECOMPOSITION-R0.1`

Status: **registered before external staging or execution**

## Question

Does the large a1400 response associated with the carried UID 4460 hard
penalty arise mainly because the detector is removed before shared RTC/PTC
cleaning, or from the detector's direct contribution to the final map?

## Intervention

The same factor-zero, scan-local, map-diagnostic detector records are applied
in one of two places:

- `pre_cleaning`: the existing behavior and default, before RTC and PTC; or
- `pre_mapmaking`: after residual-only shared cleaning and FRUIT model
  add-back, immediately before final map accumulation.

Only records with producer `mapdiag:raw_obs` and reason
`map_pixel_outlier_detector_dominance` move. Busy-detector exclusions,
network exclusions, sample masks, weights, filters, recurrence, incoming
feedback state, and injection remain unchanged.

## Four ordered replays

Each trajectory is a single iteration, absolute iteration 4 to 5:

1. `C5-current`: control C4, `pre_cleaning`;
2. `A5-current`: injected A4, `pre_cleaning`;
3. `C5-map`: control C4, `pre_mapmaking`;
4. `A5-map`: injected A4, `pre_mapmaking`.

The first two must reproduce the existing EL-F5 C5/A5 signal, kernel, and
weight planes bitwise. All checkpoint values other than creator identity and
the normalized new configuration provenance must match. Failure stops the
experiment before scientific interpretation.

## Fixed measurements

For each array retain source recovery, centroid, width, fixed-kernel
projection and residual, Neptune-region RMS, registered annular RMS, complete
maps, cross terms, checkpoint differences, and execution resources.

For a1400:

\[
T_{current}=A5-C5,\quad T_{map}=A5_{map}-C5_{map},
\]

\[
D_{current}=A5-N5,\quad D_{map}=A5_{map}-N5,
\]

\[
Q=D_{current}-D_{map}=A5-A5_{map}.
\]

The exact closure of the two definitions of `Q` is required within the
registered floating-point roundoff bound. Direct UID-4460 interpretation is
restricted to a1400. The a2000 response is a side-effect measurement because
two map-diagnostic penalties move in both branches. The a1100 busy-detector
penalty does not move.

At the four frozen a1400 trigger pixels `(row, col) = (142,280), (144,280),
(142,281), (144,281)`, report A5-current and A5-map signal and weight together
with the inherited UID 4460 leading-contributor and leave-one-out evidence.
Verify that scan-local UID 4460 is not hard-excluded before RTC/PTC in the map
placement and is fully excluded before mapmaking.

No numerical dominance threshold is registered. The mechanism result is
reported from the complete continuous component measurements and maps as
early-exclusion amplification, direct mapped contribution, or mixed.

## Bounds

One configured thread, `--grppiex seq`, at most one environmental replacement
per trajectory, at most one hour and 64 GiB per replay, and at most six hours
and 8 GiB retained in aggregate. Stop after the four replays and registered
analysis. This is descriptive development evidence only.
