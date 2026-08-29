# SCI-NOI v0.1 — Ownership And Boundary Classification

Status: proposed Stage A classification; awaiting scientific-owner review

| Fact or operation | Owner | SCI-NOI role | Prohibited promotion |
| --- | --- | --- | --- |
| RTC conditioning and learned state | SCI-RTC | Bound fixed or named-relearned input state | NOI cannot redefine RTC cleaning or call an unrerun state relearned |
| CAL quantity/unit/calibration state | SCI-CAL | Immutable inherited parent fact | NOI cannot recalibrate the signal by relabeling an empirical scale |
| PTC transformed sample, retention, segment, cleaning state | SCI-PTC | Immutable inherited parent or explicitly rerun state | NOI cannot rewrite retention/validity or invent PTC authority |
| PTC/MAP analysis/gridding coefficient | SCI-PTC / SCI-MAP boundary | Fixed operator input identity | Empirical NOI weight is not this coefficient |
| AST occurrence coordinate and WCS binding | SCI-AST | Fixed parent fact unless a named method reruns AST | Numerical coordinate equality is not identity |
| Producer validity/cause | Producing package; profile evaluated by SCI-VAL | Preserved input fact | A sign, finite output, or realization QC cannot clear producer invalidity |
| MAP ordinary estimator, base/coadd validity, response/covariance disclosure | SCI-MAP | Immutable parent and possible fixed operator | NOI cannot rewrite MAP claims or use an empirical result as retroactive MAP authority |
| JINC signed estimator and product limits | SCI-JINC | Immutable parent when an exact realized route exists | NOI cannot infer missing JINC coefficient, response, or covariance state |
| Randomization unit, sign/assignment law, balance, seed/key, membership | `NOI-GEN` | Owned | These facts do not establish uncertainty |
| Realization availability, support, QC, persistence/reconstruction | `NOI-GEN` | Owned | Availability is not sample validity or covariance adequacy |
| Target law, centering, finite correction, covariance domain, calibration | `NOI-UNC` | Owned | “Empirical” does not mean physical or calibrated |
| Empirical variance/covariance | `NOI-UNC` | Owned versioned companion | It does not become a formal MAP/JINC covariance retroactively |
| Marginal inverse variance, precision, consumer-effective weight | `NOI-UNC`, each as a distinct role | Owned when separately authorized | They are not interchangeable and are not PTC/MAP coefficients |
| Standardized signal | `NOI-STD` | Owned derived companion | It is not an uncertainty estimate, significance, or detection probability by itself |
| Deterministic filter transfer/edge/response | SCI-FLT | Bound operator dependency | NOI cannot define filter science |
| Wiener/noise-inferred filter operator | SCI-FLT plus its noise-model boundary | Fixed or named re-estimated dependency | Same ensemble cannot automatically fit and validate the operator |
| Beammap, source fitting, Pointing, OOF interpretation | SCI-BEAM / future SCI-SRC or SCI-MODE | Named consumer dependency | Their historical ratios cannot be relabeled generic NOI S/N |
| FRUIT subtraction/add-back, learning, recurrence, stopping | SCI-FRUIT | Fixed residual state or full-replay dependency | NOI cannot authorize adaptive inference or convergence |
| Policy registration/evaluation | SCI-VAL | Evaluate owner-authored exact profiles | VAL does not author NOI target, adequacy, or publication policy |

## Boundary Consequences

1. A `NOI-GEN` output may exist with `NOI-UNC` unavailable.
2. A `NOI-UNC` output may be valid for one projection and unavailable for
   another.
3. A `NOI-STD` output requires a valid `NOI-UNC` scale but gains no stronger
   probabilistic claim than that scale and its separately authorized null law.
4. Fixed-state and relearned generation cannot be members of one ensemble.
5. A parent product remains immutable even when a later NOI companion reveals
   a limitation or supplies a better empirical estimate.
6. Persistence, reconstruction, and statistical adequacy are separate axes.
7. An exact route boundary never creates an unavailable MAP, JINC, PTC, FLT,
   or FRUIT numerical parent.
