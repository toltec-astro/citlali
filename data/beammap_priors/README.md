# Beammap Priors

This directory holds git-tracked prior artifacts for beammap source identification.

## Current artifact

- `beammap_network_priors_v1.ecsv`: Network-level geometric priors derived from historical measured APT files.
- `beammap_slot_priors_soft_v1.ecsv`: Soft within-network slot priors (quantile slots along each network stripe).

## How it is built

Run:

```bash
$HOME/tolteca/bin/python data/beammap_priors/build_beammap_network_priors.py \
  --input-glob "/Users/gwilson/GitHub/apt_sandbox/data/measured_apt/apt_commissioning_beammap_*_citlali.ecsv" \
  --output data/beammap_priors/beammap_network_priors_v1.ecsv
```

Soft slot prior builder:

```bash
$HOME/tolteca/bin/python data/beammap_priors/build_beammap_slot_priors_soft.py \
  --input-glob "/Users/gwilson/GitHub/apt_sandbox/data/measured_apt/apt_commissioning_beammap_*_citlali.ecsv" \
  --output data/beammap_priors/beammap_slot_priors_soft_v1.ecsv
```

The builder uses `flag==0` detectors, recenters each observation by per-array median `(x_t, y_t)`,
and then summarizes each `(array, nw)` cloud.

## Column notes

- `x_rel_*`, `y_rel_*`: Centered derotated coordinates in arcsec.
- `x_raw_rel_*`, `y_raw_rel_*`: Centered raw coordinates in arcsec.
- `cov_*`, `pca_*`: Network-shape statistics for gating and candidate ranking.
- Soft slot priors include `x_rel_sigma_soft_arcsec` and `y_rel_sigma_soft_arcsec`, which are inflated/floored spreads for gentle gating.
