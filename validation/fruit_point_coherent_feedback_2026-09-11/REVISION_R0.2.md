# One targeted revision: coherent numerical initialization

2026-09-11. SCI-FRUIT-POINT-COHERENT-SCREEN@r0.2.
Initial results at `/private/tmp/sci-fruit-point-coherent-feedback-20260911-r0.1`
are preserved. The first campaign completed all 84 passes in 106.08 seconds,
without a trajectory failure, using peak RSS 2.02 GB. Its source and protocol
freeze at 0f8ccd9d8 remain unchanged.

The Gaussian candidate met joint injected amplitude/width/centroid targets in
2–3 passes in a1100 and a2000, while the pixelwise reference did not meet all
three within seven. But in a1400 the candidate lost 21.8–23.1% amplitude in both
Gaussian cases and failed the all-array nondegradation requirement. Inspection
of its recorded fit states identifies one precise weakness: all initial-width
fits start from the single brightest pixel, converge at approximately (−80,−18)
arcsec with a width at the 4-arcsec bound, and reject feedback at every pass.
The injected source is at (17,−11); that truth identifies the failure for
assessment only and is never passed to revised fitting.

Use the owner's single allowed revision to change only numerical start centers.
For each of the existing widths 6,18,42 arcsec, convolve the plane-subtracted
image with that Gaussian template and divide its cross-product by the square
root of template squared support. Search only existing admitted fit pixels
strictly inside the existing centroid bounds. The largest positive coherent
score supplies that width's start center and least-squares amplitude. The same
three nonlinear least-squares fits then run; lowest converged SSE wins. Neither
this initialization score nor its width becomes an amplitude/shape prior or a
feedback admission rule. Final centroid, amplitude and both widths remain free.
The final source model, background exclusion, fit bounds, 3R admission, scatter
scale, PTC, masks, rank 5, grouping, mapmaker, all six cases and seven-pass bound
remain unchanged. There is no threshold/rank/grouping sweep.

Apply this improved numerical solution of the same Gaussian evaluator to both
arms; P's feedback never consumes that fit, so P's numerical maps must reproduce
the initial reference. Keep the original evaluator results as evidence; record
any changed real-data fit interpretation. Timing still removes P's evaluation-only
fit cost and charges G's inference. Repeat the same 84-pass comparison set once,
then stop revising. Retain both attempts, both source freezes and all metrics.
129081 remains unopened; a candidate freeze and development disposition precede
its evaluation. This revision is authorized by the bounded owner direction,
not by the results alone, and exhausts that revision allowance.
