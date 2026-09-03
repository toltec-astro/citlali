# SCI-BEAM v0.1 — Final Scientific-Owner Review r0.3

Status: approved bounded corrections and stopping rule

Scientific owner: Grant Wilson

Review date: 2026-08-17

## Judgment

The r0.2 science-team rationale has the correct structure, scientific
narrative, effective-core interpretation, nominal-beam convention, full
tensor model, coordinate architecture, BEAM ownership, independent-state
model, APT contents, and validation program. None of those elements is
reopened.

## Authorized final corrections

- Define sens=abs(flxscale) n_off; preserve the physical sign on flxscale,
  and require available sens to be finite and strictly positive.
- Separate local map-fit Jacobian/covariance from the derived-product
  Jacobians that propagate calibrator, nominal-beam, atmosphere, scan-noise,
  exposure, and related uncertainties into flxscale and sens.
- Require an explicit fitted-centroid-to-detector sign/frame transformation;
  derive detector effective rotation from parent-sample contribution support
  propagated through the fitted map support.
- Require the same immutable APT artifact for bracketing pointing and the
  associated science observation unless a separately authorized transform
  proves equivalence.
- Restore one concise scientist-facing explanation of the soft prior and
  scientific convergence.
- Give the three open-decision registers stable document-facing IDs
  SCI-BEAM-OD-001--003, mapped to the nine atomic ledger questions.
- Compress the rationale contents into one compact list.
- State directly that the calibrator source flux is required in
  top-of-atmosphere mJy per fixed nominal beam; remove defensive discussion
  of the earlier mJy-versus-mJy-per-beam dispute and duplicate factors.
- Carry both v0.1 and r0.3 in final artifact filenames.

## Stopping rule

After the corrections, cross-artifact consistency check, compilation, and
rendered-page review pass, the SCI-BEAM v0.1 scientific authority is frozen.
Future changes require a formally resolved open decision, a normative
contract change, new validation evidence that changes evidentiary status, or
a genuine scientific inconsistency.

No implementation conformance, representation fidelity, observational
performance, science-impact qualification, or production readiness is
established by this review.
