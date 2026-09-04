# FRUIT EL-F11 scientific interpretation r0.5

Test ID:
`SCI-FRUIT-EL-F11-PROSPECTIVE-INFLUENCE-PERSISTENCE-R0.1`

Interpretation: **the harmful iteration-5 map consequence was already visible
with nearly the same shape and amplitude in iteration 4, before the carried
hard exclusion was applied**

## Plain answer

EL-F11 answers its prerequisite question decisively for this case. If we had
calculated UID 4460 scan 5's full JINC deletion response after iteration 4, we
would have seen essentially the same arc-shaped response that applying its
hard exclusion produced in iteration 5.

The comparison is much stronger than a loose correlation. The response maps
have identical conditioned support, their normalized alignment is `0.99964`,
their fitted amplitude ratio is only `0.4%` from unity, and their strongest 1%
of pixels are exactly the same set. After rescaling the iteration-4 response,
only `2.69%` of the iteration-5 response norm remains unexplained.

This means the large map effect was not created unexpectedly by a new
iteration-5 geometry or support change. The relevant signed leverage and
processed-signal contrast were already present before the decision took
effect.

## What has become plausible

A response-aware decision is now scientifically worth testing. The existing
rule acts because four map pixels cross a repeat-count threshold. It does not
ask what excluding the entire detector/scan will do elsewhere in the map.
EL-F11 shows that, at least in this example, that consequence could have been
evaluated from the preceding iteration rather than discovered after the
damage.

That is a feasibility result, not yet a method. The target was chosen because
we already knew it caused trouble in iteration 5. A deployable procedure still
needs a causal way to identify which candidate detector/scans deserve an
influence calculation, a bounded action when their predicted response is
large, and controls showing that ordinary helpful exclusions are not lost.

## Important limitation at the injected source

UID 4460 scan 5 contributes no conditioned target samples inside the
20-arcsec aperture around the injected source. Consequently this test says
nothing directly about preserving injected-source flux at that position. Its
strong evidence concerns prediction of the real-field and scan-trajectory map
response. Any intervention screen must therefore retain the ordinary
source-recovery, width, residual-leakage, convergence, and performance metrics;
it cannot use deletion-response agreement as its sole measure of success.

## Recommended next milestone

This completes the diagnosis-only branch. I recommend one separately reviewed
response-aware intervention screen, not another explanatory study of UID
4460. Its first obligation is to define a causal candidate-selection rule
using only state available before a carried penalty is applied. It should then
compare the unchanged historical hard action with a small, predeclared set of
bounded alternatives while retaining the historical recurrence and complete
incoming feedback state.

The screen must measure ordinary source recovery and morphology, real-field
and annular leakage, false-action behavior on benign candidates, support
changes, convergence, runtime, and memory. A favorable development result must
replicate on an independent pointing before any policy recommendation. Only a
then-frozen method may proceed toward the broader and more expensive science-
field angular-scale and flux-transfer qualification.

If a causal selector cannot be defined without using later outcomes, or if the
bounded alternatives merely exchange this failure for losses elsewhere, this
branch should stop and the historical recurrence should remain the control.
