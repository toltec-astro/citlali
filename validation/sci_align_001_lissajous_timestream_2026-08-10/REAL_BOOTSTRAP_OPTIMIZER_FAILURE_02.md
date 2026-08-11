# Real bootstrap optimizer failure 02

## Trigger and authenticated evidence

The owner judged the standard ObsNum 136280 a1100 pointing map to be only a
modest compact detection with substantial off-peak structure. That assessment
is consistent with the retained standard fit (`S/N=30.799`, fitted FWHM
`7.50 x 9.59 arcsec`) and the visible asymmetric residual field. The exact
standard FITS digest is
`fbb01c6165901c6e7b16d0cf32624bdf98dc8581f45b48945dc291f04fdf89a6`;
the companion PPT digest is
`5d9d042d315946c6b37f70c8ef3a9e439e4eb6bba74723794c600b937dc21c9e`.

A read-only audit used the already completed 1,500-draw bootstrap only. It did
not refit the estimator, alter the source model, inspect another observation,
or access Unity. The audited result digest is
`065c9cb932692485bff62663e22088666706ca4006bc34ab5225c6a4eecec017`;
its result-manifest digest is
`bbd5e6ad6c6370f7e0b911ea3901a77a6e039ab504afad181494bb69c92a1e47`.
The scan-audit checksum manifest is
`fbb8544a1618ffc6417a4d621b0c0b56df83f37cf6bc78174dd9c43c516dde29`;
the optimizer-probe manifest is
`223cb46c7bb0ea851d8054b352eae45318ee7de087e4876b16be54c006650c75`.

## Scan and morphology audit

The 361 solutions exactly equal to the inherited point-fit start cannot be
explained by a gross loss of scan diversity or timing information:

- pile-up and moved draws both have median 8 unique scans and scan effective
  sample size 6.545;
- median directional imbalance is 0.225 for pile-up draws and 0.243 for moved
  draws;
- median profiled tau information is higher, not lower, in pile-up draws
  (`25.28e6` versus `22.09e6`);
- tau is strongly associated with resampled residual MSE (Spearman
  `rho=-0.413`, `p=6.7e-63`).

Individual scan content matters materially. Scan rows 6 and 7 have nearly
opposing mean velocities, `(-10.36,-20.47)` and `(+11.83,+24.30) arcsec/s`,
and their multiplicities pull tau in opposite directions (Spearman `+0.446`
and `-0.454`). Relative scan-6 minus scan-7 multiplicity moves the bootstrap
median monotonically from about `+1.23 ms` at -3 through `+11.73 ms` at zero
to `+22.05 ms` at +3. This is compatible with the visibly structured source
or background field interacting with scan direction. It is not evidence for
a universal timing state.

## Confirmed numerical defect

Four deterministic pile-up replicates spanning different scan-6/scan-7
balances and residual MSE were rerun without changing their draws. In every
case, the inherited single-start L-BFGS-B fit returned zero iterations,
`success=false`, and message `ABNORMAL`, but the diagnostic accepted the
finite objective and recorded `status=success` at exactly `+11.728437 ms`.
The already frozen three-start lag fit converged to `+7.637389`, `+15.064359`,
`+14.627600`, and `+10.255707 ms`, with objective improvements of `56.81`,
`101.89`, `72.84`, and `4.96` respectively.

This is a confirmed diagnostic optimizer-control defect independent of the
fitted value of tau. The 361-point spike is therefore not a physical or
statistical mode. The old ObsNum 136280 bootstrap interval, KDE mode count,
paired difference interval, covariance, and correlation are contaminated and
must not be interpreted. The full-data point fit is not invalidated by this
audit, but the modest structured map makes a single clean-source lag
interpretation scientifically weak even after numerical repair.

## Narrow repair and stop discipline

The repair changes only bootstrap optimizer control flow. A bootstrap fit may
start from the authenticated point solution as before. If that single attempt
returns no successful finite optimizer result, it is discarded and retried
with the pre-existing deterministic lag starts `{-25,0,+25} ms`. Established
multistart selection, coordinates, support, samples, weights, baselines,
source model, bounds, seed, and scan draws are unchanged. Regression tests
require both the failure-triggered fallback and recovery of every existing
synthetic regime.

The historical early stop remains protective: it prevented propagation of a
bad bootstrap into a corpus result. Its immediate classification is refined
from unresolved persistent multimodality to a confirmed numerical bootstrap
failure plus real scan-dependent residual sensitivity. No unopened
observation may be inspected. After a checksum-bound successor freeze, the
smallest next owner decision is whether to rerun ObsNum 136280 alone in a new
output root; the old bootstrap checkpoint must not be resumed.
