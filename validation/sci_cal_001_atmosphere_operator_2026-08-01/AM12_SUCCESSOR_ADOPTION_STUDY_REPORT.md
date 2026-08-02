# SCI-CAL-001 AM-12.2 successor adoption study

Status: **numerical_adoption_evidence_fail**.

This study is bounded to q0--q75 and elevations 20--80 degrees. It does not evaluate q95.
The one-percent threshold is numerical representation fidelity, not observational photometric accuracy.

## Frozen candidates

- `fixed_djf25_v1`: `LMT_DJF_25` independently H2O-scaled to q25, q50, and q75.
- `conditioned_djf_v1`: `LMT_DJF_25@q25`, `LMT_DJF_50@q50`, and `LMT_DJF_75@q75`.
- Shape-preserving PCHIP in line-of-sight optical depth versus elevation for every nonzero anchor; analytic linear q0--q25 followed by either piecewise-linear or PCHIP opacity interpolation.
- TolTECA v1 ECSV passbands are primary. FTS spectra are challengers, not replacements.
- Source spectra use `S_nu` proportional to `nu^alpha` for alpha -1, 0, 2, and 4.

## Evidence

- Canonical P1 direct grids validated and integrated: 155.
- Frozen training grids: five unique target/profile constructions at 31 even elevations.
- Independent band-integrated candidate holdout rows: 23040 from all 240 direct AM truth grids.
- Digest-bound holdout run inventory: 1025 total (785 scale-search anchors plus 240 full grids).
- G8 expanded-row key coverage: 23,040/23,040 with zero missing, unexpected, or duplicate keys.
- Positivity, opacity/elevation monotonicity, continuity, fail-closed support, low-segment identity, and exact-anchor contract: PASS.
- G7 challenger disposition maximum direct-truth difference: 3.474613%.
- Machine decision status: `numerical_adoption_evidence_fail`.

## Interpretation boundary

FTS-versus-primary sensitivity follows the frozen three-state G7 disposition and is not charged to interpolation error. A numerical recommendation is not owner selection or observational authorization; calibrator repeatability and absolute-flux gates remain separate.

## Machine recommendation and conditional primary ranking

```json
{
  "conditional_primary_ranking": [],
  "recommendation": null
}
```
