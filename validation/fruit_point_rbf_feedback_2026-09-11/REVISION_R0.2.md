# One targeted empirical revision: stronger noise smoothing

2026-09-11. SCI-FRUIT-POINT-RBF@r0.2.

The first frozen candidate completed all 189 passes without numerical failure.
It rejected every synthetic source, as well as every null/background case.
Even the compact-source fold cosine was only about 0.58–0.66 on seed 20260911:
the flexible fitted fields were dominated by incoherent noise. The comatic
case did not get feedback in either R or G. This is an admission/estimation
failure, not a failed representation basis or implementation defect.

Use the owner's one allowed targeted revision to change only lambda from 0.01
to 1.0 in the same quadratic penalty. This makes the normalized squared
neighbor-difference term commensurate with the normalized residual term, rather
than treating it as a small correction. This is one deliberately stronger
smoothing choice, not a search for the best lambda. It may suppress real narrow
structure; report that cost and retain the existing recovery gates.

The 3-arcsec centers, 4-arcsec widths, screened-floor coefficient, source domain,
plane, nonnegative coefficients, parity split, two scores >=5 and cosine >=0.8
remain unchanged. All truths, brightnesses, nuisance realizations, parent,
rank, recurrence, timing conventions and tolerances stay unchanged. No
concentration value is used to select the revision. The unchanged raw basis
and newly penalized noiseless fits are recorded separately before trajectories.

Run the same nine cases, all three arms, seven passes (189 additional passes)
to retain matched timing. Freeze before execution. Preserve r0.1. No additional
candidate, input adjustment or automatic sweep follows this revision; close the
experiment from these results. 129081 remains reserved for new comparisons,
with its historical exposure disclosed.

The revised noiseless fit shows a substantial smoothing cost: compact model
image error is 41.4–44.6%, diffraction 52.6–55.4%, and coma 22.9–24.7%.
These are fitted-model errors, not output-map errors. The unchanged raw basis
passes. This makes the revision a stringent final test of the noise/fidelity
tradeoff, not a presumed improvement. Do not lower the output recovery gates or
pick another penalty after these results. The primary decision remains the
reconstructed output fidelity and admission behavior.
