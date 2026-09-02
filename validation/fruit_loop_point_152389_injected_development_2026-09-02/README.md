# FRUIT known-source development check: pointing 152389

Status: **exploratory development evidence only; not qualification, candidate
ranking, or a stopping-rule decision**

## The simple question

If we add a known 100 mJy/beam point source to this real pointing observation,
how much of it does the present FRUIT path recover after one and two feedback
passes?

The test starts from one exactly identified iteration-0 checkpoint. A control
run and an injected run are otherwise identical. Subtracting the control map
from the injected map isolates the response to the synthetic source, including
any nonlinear interaction with the real source, atmosphere, weights, and
learned state. The restarted control reproduces uninterrupted iteration 1
bit-for-bit in signal, kernel, and weight for every array.

## What happened

The main compact-source number fits the known central response and divides its
peak by both the injected 100 mJy/beam and the fitted peak of the same run's
processed kernel. A second, more global number projects the complete
injected-minus-control map onto that kernel. They answer slightly different
questions, so both are retained.

| Array | Central peak recovery, iter 1 | iter 2 | Full-kernel projection, iter 1 | iter 2 | Shape/kernel by iter 2 | Centroid error, iter 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a1100` | 87.4% | 92.6% | 81.8% | 88.3% | 95.7–96.0% | 0.017 arcsec |
| `a1400` | 85.6% | 92.5% | 79.0% | 88.7% | 95.3–97.3% | 0.049 arcsec |
| `a2000` | 78.1% | 89.5% | 71.5% | 85.4% | 95.4–97.4% | 0.016 arcsec |

The current method therefore does recover most of this compact injected
signal, and the recovery increases in all three arrays on the second feedback
pass. By iteration 2 the fitted source axes are within about 3–5 percent of the
processed-kernel axes, and the recovered centroid agrees with the kernel to
better than 0.05 arcsec.

It is not yet a convergence result. From iteration 1 to iteration 2, the full
synthetic-source response changes by 37%, 40%, and 51% of its preceding RMS in
`a1100`, `a1400`, and `a2000`. The central recovery also rises by roughly 5,
7, and 11 percentage points. These two saved feedback passes do not identify a
scientifically justified stopping iteration.

The Gaussian and full-kernel estimates differ by about 4–7 percentage points,
and the control/injected kernels and weights are not exactly equal. Median
positive weights differ by less than 0.6 percent, but the nonzero differences
confirm that this is a response of a nonlinear iterative reduction, not a
perfectly additive source pasted onto a fixed map.

## What the percentages do and do not mean

The raw fitted response peaks are only about 65–80 mJy/beam because the
processed `kernel_I` peak is itself below one. Dividing by that realized kernel
is the more useful compact-response comparison here. It does not turn the
kernel into independent diffraction truth.

The synthetic source enters after RTC processing, despiking, calibration, and
the initial learned-mask and detector-selection steps. It then passes through
PTC cleaning, FRUIT subtraction/add-back, weighting, mapmaking, and the next
iteration's carried state. Consequently, this test says nothing about flux
lost before that insertion point. It also tests only one positive, centered,
compact source co-located with the real pointing source. It does not establish
off-center response, dynamic range, angular-scale recovery, faint extended
emission recovery, atmosphere leakage, or superiority over historical Citlali.

## Analysis repair disclosed

The first report initialized its Gaussian fit from the brightest pixel in the
whole difference map. Two fits locked onto distant subtraction artifacts more
than 100 arcsec from the known injected position. Those entries were rejected
as a diagnostic error. The repaired analysis searches and constrains the fit
within 25 arcsec of the injection position, which was fixed at map center
before the run. A regression test now covers a brighter distant artifact. No
data, injected amplitude, iteration, threshold, or scientific conclusion was
retuned.

## Review and reproduction

- [`TEST_DEFINITION.md`](TEST_DEFINITION.md) records the frozen question,
  attempts, operator boundary, and analysis repair.
- [`injected_source_iteration_metrics.csv`](injected_source_iteration_metrics.csv)
  contains all numerical rows.
- [`injected_source_iteration_metrics.md`](injected_source_iteration_metrics.md)
  is the compact generated table.
- [`injected_source_iteration_metrics.png`](injected_source_iteration_metrics.png)
  plots amplitude, shape, centroid, and remaining iteration change.
- [`manifest.json`](manifest.json) records the exact executable, configuration,
  restart, maps, and SHA-256 hashes.

From the repository root, the compact products can be regenerated with:

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR=/tmp/fruit-inject-mpl \
XDG_CACHE_HOME=/tmp/fruit-inject-xdg \
$HOME/tolteca/bin/python \
  tools/fruit_loops/compare_injected_source_pair.py \
  --control /Users/gwilson/work_toltec/local_data/fruit-development/point-152389/fruit-injection-development/centered-100mjy-r0.1/attempt-04/pair/control/reduced \
  --injected /Users/gwilson/work_toltec/local_data/fruit-development/point-152389/fruit-injection-development/centered-100mjy-r0.1/attempt-04/pair/injected/reduced \
  --manifest /Users/gwilson/work_toltec/local_data/fruit-development/point-152389/fruit-injection-development/centered-100mjy-r0.1/attempt-04/pair/setup/manifest.yaml \
  --continuation-reference /Users/gwilson/work_toltec/local_data/fruit-development/point-152389/fruit-injection-development/centered-100mjy-r0.1/attempt-04/reference/reduced/redu01 \
  --obsnum 152389 \
  --output validation/fruit_loop_point_152389_injected_development_2026-09-02/injected_source_iteration_metrics.csv \
  --plot validation/fruit_loop_point_152389_injected_development_2026-09-02/injected_source_iteration_metrics.png \
  --provenance-output validation/fruit_loop_point_152389_injected_development_2026-09-02/manifest.json \
  --executable build/bin/citlali \
  --software-id sci-noi-v0.1-stage-a-22-g92d174630 \
  --test-id SCI-FRUIT-POINT-152389-INJECT-CENTER-100MJY-R0.1
```
