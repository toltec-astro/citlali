# SCI-MAP v0.2/r0.3 WCS comparison-domain specification

Status: **CANDIDATE representation-test specification**. The shared authority
defines the science. This record does not claim that a representation has been
tested or has passed.

## Exact finite population

For an immutable grid of `n_x` by `n_y` cells, preregister the population from
the full-precision typed WCS, exact grid shape, and scientific row domain
before generating the physical FITS WCS. In one-based FITS pixel coordinates,
the population is the union of:

1. every scientifically addressable pixel center `(j+1,k+1)` for
   `0 <= j < n_x` and `0 <= k < n_y`; and
2. every outer-footprint cell vertex:
   `(h+1/2,1/2)` and `(h+1/2,n_y+1/2)` for `0 <= h <= n_x`, together with
   `(1/2,l+1/2)` and `(n_x+1/2,l+1/2)` for `0 <= l <= n_y`.

The second set includes all four corners and every required outer-boundary
vertex. A center is absent only if its scientific row was already absent when
the population was generated. Every listed boundary vertex remains required.

## Bound comparison

| Item | Exact rule |
| --- | --- |
| Representations | Full-precision typed WCS and its physical FITS WCS. |
| Identity | Same WCS-generation identity, celestial frame, axis order, signs, handedness, orientation, grid shape, and centered-integer reference-pixel relation. |
| Index mapping | Zero-based memory center `(j,k)` corresponds exactly to one-based FITS center `(j+1,k+1)`; boundary coordinates use the explicit half-pixel values above. |
| Metric | Great-circle angle `atan2(||u x v||, u dot v)` between unit direction vectors in the exact declared frame. |
| Precision | IEEE-754 binary64 operations, round-to-nearest ties-to-even; separation reported in arcseconds. |
| Bound | Maximum separation over the complete preregistered finite population is at most 0.1 arcsec. |
| Invalid point | If either WCS yields an invalid or non-finite direction at any required population member, the representation fails; the point is not dropped. |

The finite population establishes only the stated representation bound on
that population. It authorizes no inference about a smaller representative
subset, a continuum between points, or a global sky maximum. The tolerance
does not authorize a different grid, crop, shift, interpolation, reprojection,
or mosaic.

Prospective fixtures include both one-cell and multi-cell grids, every corner,
all four boundary runs, an addressable-center subset fixed before FITS
generation, axis-sign/handedness reversal, centered-integer mismatch, and one
invalid projection point. Results remain `not_assessed`.
