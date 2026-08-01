# Evidence boundaries and named limitations

The campaign verifier is intentionally stricter than an output-only FITS
inventory. These limits must remain visible in every result and handoff.

1. Observation geometric hits, contribution hits, upstream exposure, retained
   exposure, and pre-normalization thresholds cannot be independently
   reconstructed from normalized output maps. Exact processed per-term state
   and realization signs are required through `sample-ledger-contract.json`.
   Missing ledgers are an evidence gap.
2. Coadd-enabled cases do not publish observation realization FITS. A direct
   reconstruction needs approved processed intermediates; an S-E cross-case
   proxy is allowed only after exact non-noise identity and byte equality are
   proved, and it must remain labelled as a proxy.
3. The scan-farm `2*gamma_n*sum(abs(per_scan_value))` bound needs exact
   run-produced pre-normalization binary64 per-scan planes and commit order.
   Normalized aggregate maps cannot establish it. The Unity analyzer records
   that exact lane as neutral N/A, makes no external gamma claim, and still
   performs the registered exact-topology/WCS/inventory plus `atol=2e-8`,
   `rtol=1e-10` seq/OpenMP comparisons. The candidate local F011 truth suite
   remains the exact policy gate.
4. Response identity includes configured kernel type/path, sigma/FWHM/limit,
   grouping, extension/source vectors, and source images. `kernel_I` alone is
   insufficient; all source facts must be preflight-hashed.
5. Exact exposure provenance sums use Eigen's native floating reduction. A
   portable external sum can characterize numerical agreement but is not a
   byte-exact native-sum authority without a frozen native conformance helper.
6. Filtering is disabled in all seven cases, so filtered raw-parent carriage
   is not exercised.
7. The campaign does not exercise JINC, detector grouping, non-array grouping,
   polarimetry, general reprojection/interpolation, or covariance/precision
   claims.
8. Successful production cases do not exercise F009 failure atomicity. Local
   mismatch, tamper, and failure-injection tests remain its evidence.
9. Sign-randomized realizations do not establish calibrated significance,
   precision, covariance, physical flux calibration, astrometric truth, or
   upstream-eligibility correctness.
10. The campaign cannot close SCI-ALIGN-001, SCI-CAL-001, SCI-AST-001,
    SCI-PTC-001, or SCI-VAL-001. F013 remains conditioned.
11. Typed frame/projection and persisted WCS relationships are checked for
    exact internal consistency. This campaign does not independently bind
    `source_epoch`/`RADESYS` to a J2000 raw authority, so absolute epoch and
    astrometric acceptance remain conditioned on SCI-AST-001 through F013.

An unavailable required input causes a named evidence gap. The analyzer must
never replace missing authority with a finite output value, weight, coverage,
support alias, filename inference, or old-run artifact.
