# Coherent-I/Q sidecar validation — 2026-07-31

## Outcome

The corrected all-network production observer is operational, but its mode
scores are not yet equivalent to the offline reference. Candidate-time
refinement is the next repair gate. This result does not authorize masking,
subtraction, detector exclusion, weighting changes, or a network allow-list.

The validated Unity product is observation 152433 from Citlali `41ff5d64`:

- observer execution completed without exceeding its workload budget;
- 1,128 unique RTC-seeded candidate times were retained;
- all 12,408 candidate/network pairs across 11 raw networks were scored;
- no candidate failed template or tone-coordinate compatibility.

## Independent event overlap

At the frozen 0.35 s matching tolerance:

- 16/19 curated raw-I/Q event-vector clusters are recovered (84.2%);
- 250/328 primary events from the independently detected continuous catalog
  are recovered (76.2%);
- 322/443 events of all continuous-catalog quality tiers are recovered
  (72.7%);
- the primary-catalog overlap exceeds 1,000 circular-shift null trials
  (`p = 0.000999`).

The three unmatched curated events are c0046, c0047, and c0049. Unmatched
sidecar candidates are not labeled false positives: the production sidecar is
RTC-seeded across all available networks, while the continuous catalog was
selected for a learned six-network step-mode family.

## Timing and score transfer

For the 16 matched curated events, the sidecar candidate center precedes the
offline event-vector center by a median 0.221 s. Comparing all 176 matched
event/network score pairs gives:

- signed-amplitude Pearson correlation: 0.741;
- signed-amplitude Spearman correlation: 0.849;
- amplitude-sign agreement: 84.7%;
- absolute-cosine Pearson correlation: 0.425;
- absolute-cosine Spearman correlation: 0.391;
- median absolute amplitude difference: 1.814 mrad;
- median absolute cosine difference: 0.160.

At the diagnostic-only operating point of absolute cosine at least 0.6 and
absolute amplitude at least 5 mrad, the runtime sidecar selects 25 responses,
the offline analysis selects 61, and 24 are common. Runtime recall of the
offline-selected responses is therefore 39.3%, while its precision against
that descriptive reference is 96.0%. Absolute timing residual and
absolute-cosine error have Spearman correlation 0.443.

The score suppression is not explained by using alternating-half templates
offline versus full-corpus templates at runtime: a separate fold-all offline
recalculation gives essentially the same discrepancy. The evidence instead
points to candidate-window alignment. Both implementations use the same
0.20 s pre-window, 0.05 s guard, and 0.20 s post-window, but center those
windows on different candidate-time estimates.

## Candidate-time refinement implementation

An opt-in, observe-only refinement is now implemented on
`codex/coherent-iq-sidecar-validation`. It preserves the RTC seed and original
seed-centered `mode_score`. For every compatible network it projects a bounded
raw-I/Q window onto the versioned template modes, smooths the projected phase,
and finds the strongest local absolute derivative. It rejects boundary peaks,
weak derivatives, and separated comparable peaks. A shared time is accepted
only when at least the configured number of networks agree within the
configured tolerance. The sidecar records each local result, the shared
consensus, and a separate `refined_mode_score` evaluated at that shared time.

The feature is disabled by default and does not contain a network allow-list.
It does not change samples, flags, weights, maps, or learning state. The
version-one schema remains available as
`validation/coherent_iq_mode_sidecar_v1.schema.json`; new output uses the
version-two schema.

Local validation passed the complete 540-test CTest suite (539 enabled, one
pre-existing disabled), the 123-test configuration preflight, focused C++ and
Python tests, public-header syntax, and v1/v2 schema checks. The exact Unity
science corpus is not available locally, so scientific acceptance remains
pending.

## Required next gate

Run the bounded observation-152433 Unity smoke with refinement enabled and:

1. require completed execution without exceeding the existing global work
   budget;
2. inspect the distribution of local rejection reasons, shared support, time
   displacement, and network-time span;
3. compare the preserved seed scores and refined scores independently against
   the same frozen curated and continuous catalogs;
4. require materially improved timing and score transfer without widening the
   matching or descriptive score thresholds; and
5. confirm that the added second network pass remains operationally bounded.

Shape score and absolute amplitude remain separate quantities. Displaced-state
occupancy, settling, and stable dwell boundaries are not supplied by this
change and remain the next design layer before masking is considered.

A success here still advances only observe-only diagnostics. Stable dwell
boundaries and astronomical signal-bias tests are separate gates before any
masking or correction.

## Reproducible artifacts

Reusable validator:

`tools/diagnostics/coherent_iq_sidecar_validation.py`

Project evidence directory:

`~/work_toltec/local_data/2025-C1-COM-01/docs/coherent-iq-sidecar-validation-20260731`

The directory contains the report, checksummed manifest, deduplicated sidecar
candidates, curated and continuous-catalog matches, matched network-score
pairs, and per-network summaries. The manifest records exact input paths,
SHA-256 identities, thresholds, null-test seeds, execution summary, metrics,
and limitations.
