# SCI-ALIGN-001 unthresholded full-PTC evidence - 2026-08-08

## Scope and identity

This note records a bounded read-only review of the owner-run ObsNum 150819
full-PTC reconstruction for a1100 detector UID 199 (network 0). It tests
whether Citlali's final map-support threshold or retained-pixel selection is
the primary cause of the previously measured split-direction displacement.
It does not identify the physical timestamp event, prescribe a timing
correction, or generalize one detector to the full instrument.

The owner-returned evidence directory was:

```text
/Users/gwilson/foo/review_uid199_unthresholded_ptc_v1
```

The external evidence remains outside the repository. Its checksum manifest
has SHA-256
`4596f5e3b60380ac4e546b93914441f58e388752d09f409dd627c08abb6bf457`.
All five listed artifacts verified:

| Artifact | SHA-256 |
| --- | --- |
| `displacement_comparison.ecsv` | `01decffbcde347e77f5bf2ce4cfaf4697eb2bd3a949aa05dbc11faeea990122e` |
| `manifest.json` | `cdeb93f60d869238ccba60e9372b45b7232738bc45ae47ea81cc11d801612a23` |
| `reconstruction_metrics.ecsv` | `7dbdb4b8a40b283a91c480c6d12f0c9d9d1813ccb3b6a5ef643bc68650b5e176` |
| `unthresholded_maps.npz` | `b9e6139cf506be2c45c00ed572f4c51145e3ac765460bd81b129b4d4af4f613b` |
| `unthresholded_ptc_maps_o150819_uid199.pdf` | `7389e490774edbaec65f880c0fdd63e78392bd79fca56415ba21a32c061d661f` |

The manifest schema is
`sci-align-001-unthresholded-full-ptc-map-reconstruction-v1`. It binds ObsNum
150819, a1100, UID 199, 199 scans, 153,360 selected samples, every input path,
size, and digest, and the pointing contract used by the reconstruction. The
stored physical detector pointing includes the elevation-rotated APT offset;
detector-grouped mapmaking instead uses telescope tangent pointing plus the
configured pointing offsets and suppresses the physical detector APT offset.
The observed and expected physical-pointing offsets both equal
69.3669607790 arcsec, with a maximum model residual of
`3.63e-14` arcsec.

## Independent artifact review

The ECSV rows agree exactly with the JSON manifest. The NPZ contains 301 by
301 maps on a 1-arcsec grid from -150 through +150 arcsec for standard, left,
and right signal, weight, and hit count. Its internal accumulation identities
hold:

| Mode | Selected samples | Accepted samples | Sum of hit counts | Positive-weight pixels |
| --- | ---: | ---: | ---: | ---: |
| standard | 153,360 | 140,624 | 140,624 | 53,439 |
| left | 77,364 | 70,964 | 70,964 | 31,495 |
| right | 75,996 | 69,660 | 69,660 | 31,393 |

The standard hit-count and weight maps equal left plus right exactly. The
standard weighted signal equals the sum of the left and right weighted signals
with a maximum absolute residual of `6.35e-22` in native signal-weight units.
There are no non-finite accepted signal samples and no outside-map samples.

Both PDF pages were rendered and inspected. Page 1 shows the unthresholded
positive-weight reconstructions above the retained Citlali maps on the same
coordinates. Sparse horizontal gaps are already present before the final
support threshold, consistent with raster hit geometry and nearest-pixel
projection rather than a detector-timestream discontinuity. Page 2 shows the
recentered standard, left, and right core profiles and the exact directional
displacement comparison. The profiles retain broadly similar core morphology;
modest width and wing differences remain compatible with sparse directional
coverage and the mostly cross-scan 3C273 jet.

## Result

| Product family | Right-minus-left parallel | Perpendicular | Timing equivalent |
| --- | ---: | ---: | ---: |
| Unthresholded full-PTC reconstruction | -2.698480468 arcsec | -0.706284456 arcsec | -28.703167985 ms |
| Thresholded retained Citlali APT | -2.538003907 arcsec | -0.521129923 arcsec | -26.996212620 ms |
| Unthresholded minus retained | -0.160476562 arcsec | -0.185154534 arcsec | -1.706955366 ms |

The parallel displacement has the same sign and similar magnitude before and
after final support thresholding. Removing the threshold changes its magnitude
by 0.160 arcsec (about 6.3 percent of the retained result), rather than
removing the 2.5-2.7 arcsec effect.

### What this falsifies

For ObsNum 150819 a1100 UID 199, final support thresholding or retained-pixel
selection is strongly disfavored as the primary cause of the split-direction
displacement. The sparse white map structure is also not created primarily by
that threshold; it is already present in the positive-weight hit geometry.

### What this does not prove

- The full PTC is a separate single-pass replay, not a reversible pre-threshold
  buffer from the retained multi-iteration map. The 0.160-arcsec difference
  must not be interpreted as a pure threshold bias.
- One detector in one observation does not establish a universal offset or a
  correction applicable to other detectors, networks, sessions, or sources.
- The result does not distinguish correctly timestamped native detector phase,
  detector-frame quantization of PPS observation, adjacent or non-atomic
  metadata association, or start/end/centroid timestamp semantics.
- It does not locate an error in the ROACH, FPGA, UDP metadata, telescope
  producer, or Citlali, and it does not authorize row reassociation or a
  physical clock correction.

## Smallest next diagnostic

Before requesting any new Unity work, perform one local same-T0 cadence-lattice
comparison using the already downloaded 11-map corpus aggregate. The frozen
T0-vector group `roach-t0:44cf69da97d473965ef6` contains the three enhanced
maps 148670, 150819, and 151126. Join their per-network timing estimates and
retained raw-phase fields by stable network ID, then compute pairwise changes
in:

1. measured timing;
2. native detector-frame phase;
3. native-to-assigned-slot residual; and
4. same-row versus adjacent-row PPS/PpsTime association and anomaly class.

Evaluate the residual after the preregistered slope `-1` slot prediction on
the 8.192-ms cadence lattice and its 4.096-ms half-step. This single local
comparison most directly separates the remaining hypotheses:

- stable within-T0 native phase predicts little unexplained pairwise change;
- an adjacent-frame association predicts discrete approximately one-row
  changes tied to association class;
- start/end/centroid semantics predicts a common half-row component rather
  than map-varying network changes; and
- a non-lattice, association-variable residual keeps non-atomic metadata
  capture or another upstream timing-semantic cause in scope.

This is a proposed read-only diagnostic only. No new reduction, Unity request,
application change, or correction is authorized before owner review.
