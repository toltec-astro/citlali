# Beammap Configuration

Review and ordinary edits should start with:

1. `81_beammap_defaults.yaml` for iterations, convergence, reference detector,
   detector weighting, priors, mask activation, mapmaking, cleaning, and fruit
   loops;
2. `82_beammap_products.yaml` for detector TOD, split FITS, line-audit, and
   retained RTC/PTC TOD products; and
3. `71_beammap_runtime.yaml` for executable and CPU allocation.

TolPROJ generates `72_beammap_observation.yaml`, including the calibrator fluxes
owned upstream. Do not edit `60_beammap_internal_policy.yaml`; it is the
complete accepted compatibility policy. Detailed flagging, prior scoring,
mask thresholds, fit controls, detector selection, and split-flag policy remain
available only through advanced or expert overrides.

The unchanged files merge exactly to accepted Beammap `redu06`.
