# FRUIT EL-F10 scientific interpretation r0.6

Test ID: `SCI-FRUIT-EL-F10-R4-NOISE-PASS-LEDGER-REPAIR-R0.1`

Result: **the large localized arcs are quantitatively explained by a modest
local signed coefficient share acting on an extreme processed-signal
contrast**

## Plain answer

The result resolves the apparent paradox without showing that one detector
simply dominates the map.

UID 4460's ordinary scan-5 detector weight is elevated but not at the
pipeline's high-weight limit: `5.69737e-5`, about three times the scan median
of `1.912e-5`, but well below the logged upper limit of `2.103e-4`. It is
therefore unsurprising that ordinary scalar weighting does not reject it.

JINC's local map response is not determined by that scalar weight alone. It
depends on a signed, spatially varying kernel coefficient and on how different
the detector/scan's processed signal is from the signal supplied by the rest
of the data at the same pixel. At the worst arc pixel, UID 4460 scan 5 supplies
only 4.87% of the signed JINC denominator, while its target-only signal differs
from the without-target signal by about `2032 mJy/beam`. Their product is the
observed `-98.97 mJy/beam` response. The accounting closes to rounding error.

Thus the large imprint is a localized tail of **leverage times contrast**, not
evidence that UID 4460 globally outweighs all other detectors. Hundreds of
detectors can contribute at the same pixel and still permit this response
because the JINC denominator is signed, kernel contributions partly cancel,
and the target signal can be extremely different from the local remainder.

## The exact identity

For total JINC numerator and signed denominator `N,C`, and target UID/scan
components `N_t,C_t`, define

\[
M = N/C,\qquad M_t=N_t/C_t,\qquad
M_{-t}=(N-N_t)/(C-C_t).
\]

Then, wherever the registered conditioning rules admit the quantities,

\[
M_{-t}-M = \frac{C_t}{C}\left(M_{-t}-M_t\right).
\]

The left side is the counterfactual map response. The first factor on the
right is signed local leverage, and the second is processed-signal contrast.
The maximum absolute residual in this identity is
`2.44360e-13 mJy/beam` over 4,229 target-footprint pixels.

## How localized is the effect?

Most target-footprint pixels have negligible signed leverage: the median is
`0.000168%`. The 95th percentile is `1.70%`, and the maximum is `19.15%`.
The corresponding deletion response has median
`0.0000749 mJy/beam`, but a long tail: 745 footprint pixels exceed
`1 mJy/beam` in absolute response, 266 exceed `5`, 146 exceed `10`, 56 exceed
`20`, and 22 exceed `50 mJy/beam`.

The relationship strengthens sharply in the high-leverage tail. In the
highest of ten equal-count signed-leverage bins, the leverage range is
0.827%--19.15%, its median is 1.71%, and the response RMS is
`19.842 mJy/beam`. Lower and middle leverage bins have much smaller response.

The real Neptune region contains 284--324 unique contributing detectors. Its
95th-percentile signed leverage is 0.979%, maximum leverage is 5.73%, and
response RMS is `2.537 mJy/beam`. In the 40--120 arcsec annulus about the
injected position, excluding Neptune, the corresponding values are 1.74%,
18.87%, and `7.281 mJy/beam`.

## Relation to the earlier FRUIT experiments

This accounting confirms EL-F8's direct mapmaking component rather than
changing it. Exact subtraction of UID 4460 scan 5's JINC components
reconstructs the previously retained map-only counterfactual with no support
changes and errors far below the registered bounds.

It also explains EL-F9's finding that the four trigger pixels and the response
arcs are spatially separated. All four trigger pixels have zero target
occurrences and zero deletion response. They selected a scan-local hard
action; the response occurs elsewhere along the full scan/JINC footprint.

UID 4460 scan 5 has no direct target footprint in the 20-arcsec aperture around
the off-source injected compact source. This is consistent with EL-F8's zero
direct mapmaking term there. The strong direct response is tied to the real
field and scan geometry in this observation, not to direct overlap with the
injected source.

## What this does not establish

The evidence is exact for this counterfactual, but narrow. It does not show
that UID 4460 is intrinsically defective, that all similar arcs share this
cause, or that a threshold on `C_t/C` alone would be scientifically safe.
Signed leverage, signal contrast, support, and the earlier shared-cleaning
component all matter. Nor can the already calculated leave-one-out response
be used blindly as a production decision statistic, because a practical rule
must be computable before taking the action it is meant to choose.

## Recommended next significant decision

The next study should not tune the ordinary detector-weight cutoff. That
quantity does not measure the interaction found here.

I recommend an EL-F11 **prospective influence test**. Its first task would be
to define a quantity available before the hard exclusion that predicts the
map consequence of a detector/scan using local signed leverage and processed-
signal contrast. It should then compare, without choosing a production rule:

1. the existing repeat-count hard exclusion;
2. a response-aware hard decision; and
3. a bounded soft or map-local response-aware action.

The comparison should retain source recovery, real-field residual leakage,
false-action rate, support changes, convergence, runtime, and memory, and it
should require replication on an independent pointing before any safeguard
selection. Choosing to open that study is a scientific-method decision and is
the next point requiring owner involvement.
