# Separate optical response from temporal stability — candidate v0.1

Status: authorized proposal and synthetic study; not wired into the pipeline.
See [work order](WORK_ORDER.md) and [recovery](PRIOR_WORK.md). The current frozen
trial method and detector selections remain unchanged.

**Synthetic disposition: revise before activation.** The seed sweep found that
the R2 non-overlap veto flags genuinely stationary synthetic detectors in 2 of
16 fixed noise realizations. The exact proposed rule is retained below so the
failure is reproducible. R1/full-response separation is supported by the cases;
the R2 veto is **not validated for automatic scientific rejection**. Do not use
this prototype to restore or newly reject observational detectors. See REPORT.md.

## Meaning and inputs

The question “does this detector respond to the shared optical signal?” uses the
whole observation. The question “is there evidence its response changes with
time?” is reported separately. Lack of precision in a short subset is not by
itself evidence of bad response.

Keep configurable `f = 0.10`, array-relative normalization, existing protected
source domain, held-out-network reference, two peer panels and required a2000
cross-array corroboration. Response is dimensionless, not a calibration gain.
Use the same six conditional intervals (full A/B, early A/B, late A/B) and their
availability facts. Missing is not zero. No duration/correlation/noise cut is
introduced. Membership, once eventually applied, remains observation-local and
frozen before PTC/FRUIT; prior exclusions cannot be undone by this rule.

## R1: optical response

With both full-observation panel intervals available, take their envelope `[L,U]`:

| Condition | Optical state |
| --- | --- |
| `L > f` | responsive |
| `-f < L` and `U < f` | weak full-observation response |
| `U < -f` | reversed full-observation response |
| Otherwise, including exact boundary contact | unresolved |

Unavailable full evidence or required reference means unresolved and withheld.
Prior unavailability and disturbance exclusion retain precedence. Correlation
alone is never an admission rule. These optical labels describe the full domain,
not a claim that all temporal subsets agree or that a detector is physically dead.

## R2: temporal evidence

Evaluate temporal causes without replacing the full-observation optical state:

- Missing any required half interval: `unassessed`; withhold. A finite wide
  interval is different from an unavailable estimate.
- Any available half interval with `upper <= f`: positive evidence of a subset
  failing the required response; withhold. Exact upper-bound contact is held
  conservatively; no epsilon or tuned tolerance is applied.
- Early and late intervals do not overlap **within either same peer panel**:
  `difference_requires_review`; withhold for review, even if both are positive.
  Disjointness is `max(lower) > min(upper)`; touching endpoints overlap.
  This is a provisional review flag, not a calibrated significance test or a
  declaration that the observed difference damages maps. No material-change
  tolerance has been approved.
- Otherwise, if all half lower bounds exceed `f`, report
  `positive_subsets_no_detected_difference`. This does not prove stationarity.
- Otherwise report `subset_uncertainty_only`: adequate full evidence, available
  but imprecise short-subset evidence, and no above contradiction flag.

Keep all causes if several apply. Display priority is temporal difference,
subset failing requirement, missing interval, then the two non-veto states.
Missingness must not conceal a known contradiction elsewhere.

## R3: study eligibility and diagnostics

Only `responsive` plus either non-veto temporal state is eligible. The uncertainty
case is explicitly `eligible_with_temporal_uncertainty`, not “stable.” All other
cases remain withheld. Thus weakening the precision demand does not admit dark,
reversed, full-domain ambiguous, known temporally inconsistent, or previously
excluded detectors.

Future compact diagnostics must count optical and temporal states separately,
with final eligibility/reasons and exact detector identities. Existing and newly
withheld populations must not be double-counted. Do not change the APT, apply a
response gain, bypass source protection, or fabricate a noise coefficient.
Normal inverse-noise learning still decides usable segment weights after health;
it is not a guarantee against coherent contamination or unusually high leverage.

## Validation interpretation

Synthetic tests must include: healthy quiet/noisy response; weak and reversed
response; limited within-block excitation with a normal full response; actual
positive gain changes; late weak/reversed response; absent excitation, singular
draws and missing halves; near-cut boundaries; peer disagreement; and prior
disturbance exclusion. Hand-specified intervals test decisions; independent
synthetic timestreams test whether the statistical procedure can produce the
distinction. Never use only intervals crafted to match the implementation.

Synthetic estimator evidence here is an independent, source-free reference
experiment with frozen Huber weights and the exact multinomial block-bootstrap
distribution. It is not native learner conformance (reference construction,
binning, source masks and finite 2000-draw RNG differ in scope). Do not import it
into application code or silently change native interval semantics.

No nominal 99% coverage, stationarity, unbiased flux, improved map sensitivity or
real-detector eligibility is established. A future native implementation needs
its own method identity, saved-input validation, synthetic conformance and
owner-run comparison. Non-overlap/missing-half dispositions remain proposed
until the owner accepts this exact candidate; this study does not activate them.
