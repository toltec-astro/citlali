# FRUIT EL-F8 penalty-placement result

Result: **valid bounded decomposition; descriptive development evidence only**

Test ID: `SCI-FRUIT-EL-F8-PENALTY-PLACEMENT-DECOMPOSITION-R0.1`

## Validity

- current-placement control and injection maps: bitwise compatible;
- scientific checkpoint values: compatible after the two registered provenance allowances;
- paired placement-only configuration checks: `PASS`;
- units, WCS/grid, normalization, support, and Q closure: `PASS`; and
- unexpected error/critical messages: zero.

## Compact response

| Array | Current central recovery | Map-only central recovery | Current width major/minor | Map-only width major/minor |
|---|---:|---:|---:|---:|
| a1100 | 0.981291 | 0.981291 | 0.9981 / 1.0052 | 0.9981 / 1.0052 |
| a1400 | 1.037055 | 1.037009 | 1.0147 / 1.0165 | 1.0139 / 1.0155 |
| a2000 | 0.967963 | 0.967870 | 1.0085 / 0.9858 | 1.0087 / 0.9859 |

## a1400 placement components

RMS values are mJy/beam. No numerical dominance threshold was registered.

| Region | D current | D map-only | Q early-placement increment |
|---|---:|---:|---:|
| injected source r<20 | 0.262866 | 0 | 0.262866 |
| Neptune r<20 | 2.43696 | 2.13653 | 1.20638 |
| annulus 40-120 excl. Neptune | 2.2736 | 2.05442 | 0.9634 |

The complete component maps, fixed-kernel residuals, signed cross terms, trigger-pixel table, application evidence, and execution resources are retained with this report.

Direct UID 4460 interpretation is limited to a1400. The a2000 measurement includes two moved map-diagnostic exclusions; a1100 retains its busy-detector placement.

This result does not judge UID 4460, establish a generic mechanism, select a safeguard or production policy, qualify FRUIT, launch Gate D, or begin Stage B.
