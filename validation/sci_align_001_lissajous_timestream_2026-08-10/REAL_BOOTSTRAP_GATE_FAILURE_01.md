# Real bootstrap gate failure 01

The first two frozen corpus batches were run only after the anchor passed all
coordinate, numerical, residual, sensitivity, profile, paired-bootstrap, and
synthetic gates. ObsNum 136280 completed 500 exact whole-scan bootstrap
realizations with the scalar interval-change metrics inside their declared
limits, but `bootstrap_summary` classified the distribution as multimodal.

The frozen stop rule requires a bootstrap distribution to be extended beyond
500 when convergence requires it and permits acceptance only if multimodality
resolves; persistent multimodality at 1,500 is an observation-level failure.
The implementation incorrectly allowed scalar interval convergence to end the
bootstrap without also requiring `multimodal=false`.

This is a statistical gate defect independent of the fitted tau value. The
repair adds unimodality to `bootstrap_is_converged`, preserving the existing
250-realization increment and 1,500 maximum. It also adds a checksum-valid
extension command that preserves the original 500-realization result and
checksum manifest, reauthenticates all inputs, and resumes the existing exact
whole-scan draws without repeating or changing deterministic fits.
The extension accepts only the exact revision-4 predecessor protocol digest
recorded by the initial result or the current revision-5 digest, and records
both the point-fit and extension protocol identities in the successor result.

No estimator, coordinate model, sample support, source window, nuisance model,
search bound, seed, resampling unit, or paired-map arithmetic changes. The
complete synthetic suite and a successor checksum freeze are required before
ObsNum 136280 is extended or another real observation is examined.
