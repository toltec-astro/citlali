# SCI-FRUIT v0.1 — EL-F10 Targeted JINC Accounting r0.1

Decision candidate:
`SCI-FRUIT-EL-F10-TARGETED-JINC-ACCOUNTING-R0.1`

Status: **owner-review proposal; no implementation, build, replay, or
analysis execution is authorized**

## The short version

EL-F9 established why the existing output is insufficient: a JINC map stores
the normalized answer and a nonlinear coefficient, not the additive pieces
needed to remove one detector mathematically.

EL-F10 proposes one short diagnostic replay of the already studied
iteration-5 case. It would keep the missing total and UID-4460 JINC accounting
and the final target samples. Before those diagnostics may be interpreted,
the replay's ordinary science maps must match the existing no-penalty result
bit for bit.

This is an explanation test. It does not change FRUIT, change a detector
decision, or try a safeguard.

## Exact question

For observation 123424, a1400, UID 4460, zero-based scan 5, and the existing
iteration-4-to-5 transition:

> Is the large direct map response associated mainly with high local JINC
> leverage, unusually different processed signal, signed-kernel cancellation,
> low local detector redundancy, or a measured combination of these?

The result remains local to this one exposed trajectory. It cannot establish
that UID 4460 is bad or that another observation will behave the same way.

## Bounded implementation

Add one disabled-by-default, typed `mapmaking.jinc_accounting` diagnostic.
For one exact `(array, uid, zero-based scan)` target it retains:

- total and target `N`, `C`, and `Q` JINC accumulator planes;
- total and target absolute-term sums, contribution counts, and unique-
  detector counts;
- realized support masks, thresholds, grid/WCS, units, and algorithm
  identities; and
- a compact final-PTC target-sample table with positions, processed values,
  weights, flags/admission reasons, and JINC placement facts.

The exact algebra and interpretation limits are in
`EL_F10_JINC_ACCOUNTING_DESIGN_R0.1.md`. The sidecars are diagnostic receipts,
not calibrated detector sky maps, new science products, or restart state.

The setting must be generic. Production code may not hard-code observation
123424, UID 4460, scan 5, or a1400. Enabling it outside JINC raw-observation
mapmaking, selecting an absent/ambiguous target, or failing a required output
must fail closed. Diagnostics disabled remains the default and allocates no
diagnostic state.

## One replay only

Use an exact copy of the EL-F6 `injected-without-uid4460` iteration-4 restart
source:

`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f6-off-source-penalty-counterfactual-r0.1/injected-without-uid4460/restart-source/redu04`

The required checkpoint is 6,508,179 bytes with SHA-256
`9f8faf73fc759202258ba58109ba499bd73d8f513d93ea763df75069ae78f942`.
It contains the injected trajectory with the UID 4460 hard record removed.

Advance exactly once, from absolute iteration 4 to 5, with the existing
off-source 100 mJy/beam injection at map-world `(AZOFFSET, ELOFFSET) =
(0, -60)` arcsec and all existing RTC, PTC, masks, weights, filters, feedback,
learning, mapmaking, and stopping settings unchanged. Enable only the new
diagnostic and point output/restart paths to isolated copies under:

`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1`

Use one configured thread and `--grppiex seq`. Preserve every EL-F5--F9 input
and product byte-for-byte.

## Compatibility gate before interpretation

The diagnostic-on replay is compared with the existing EL-F6 no-record `N5`
reference:

- a1400 science-map FITS: 7,191,360 bytes, SHA-256
  `5d8594aa566d3bd30f00e4ca3beecef69e3c69f26503f57ce4f0c7834670b0cd`;
- iteration-5 checkpoint: 6,506,519 bytes, SHA-256
  `d7df0ee480ad99ab3e1b51bb9f311c69e5ae9ab7104a525490dc2fe32ff37faa`.

All nine a1100/a1400/a2000 signal, kernel, and coefficient planes must be
bitwise identical. Every scientific checkpoint value must be identical,
excluding only registered creator-version and diagnostic-configuration
provenance fields. Units, WCS/grid, map cardinality, normalization, support,
injection, learning state, and sample participation must match.

If this gate fails, stop. The new accounting products may be debugged but may
not be scientifically interpreted.

## Accounting and reconstruction gates

After compatibility passes:

1. verify that the retained total `N`, `C`, and `Q` re-finalize to the
   diagnostic-on a1400 signal and unscaled formal-coefficient planes exactly;
2. verify the target-sample ledger against the expected 305 proposed, 34
   already unavailable, and 271 otherwise admitted occurrences, reporting any
   additional JINC geometry/support exclusions separately;
3. subtract target `N`, `C`, and `Q`, apply the exact existing finalization
   rules, and compare the result with the existing EL-F8 `A5-map` a1400 FITS
   (7,191,360 bytes, SHA-256
   `ce1633261dbd8bdb3836f9db8eb731e7e46d7f817e5fe3951f73e8ffef81468c`);
4. require every common-support signal and unscaled formal-coefficient
   difference to lie within the pre-registered per-pixel binary64 bound; and
5. report support agreement and any bound-explainable threshold-edge changes
   separately. An unexplained support change or out-of-bound pixel stops the
   interpretation.

Bitwise equality is required for diagnostic neutrality. It is not required
for `total - target` reconstruction because subtracting two separately
rounded accumulations need not reproduce the summation order of the existing
without-target run. That comparison instead has a fixed forward-error bound;
no tolerance may be chosen after the result is seen.

## What will be measured

For the complete common-support map, the existing injected-source aperture,
the fitted-Neptune aperture, and the existing 40--120 arcsec annulus with
Neptune excluded, report:

- signed normalization share `C_t/C`;
- absolute coefficient-mass share `B_t/B`;
- quadratic-support share `Q_t/Q`;
- total and target signed-cancellation statistics;
- exact hit and unique-detector support;
- target-only and without-target normalized signals where both are
  conditioned;
- direct deletion response and the closed leverage-times-contrast identity;
- distributions and binned response RMS versus leverage, contrast,
  cancellation, and redundancy; and
- the same facts at the four original trigger pixels and along the observed
  scan-shaped response.

Retain complete component maps and signed cross terms. Correlations are
descriptive. No dominance cutoff, detector-quality threshold, or safeguard
rule may be invented from this single case.

Report wall time, peak memory when available, and diagnostic-product size
against the existing short N5 run. These are development costs, not a FRUIT
performance qualification.

## Verification and bounds

- Source parent: Git commit `2a2962409` on
  `codex/sci-fruit-v0.1-empirical-lane`.
- Add focused tests for exact total/target accounting, signed kernels,
  zero/negative/cancelled normalization, target selection, sample-ledger
  reasons, disabled allocation, required-output failure, and science-map
  neutrality.
- Pass the local Citlali build, all enabled CTest cases, baseline and FRUIT
  Python tests, Ruff/byte compilation for new Python, and the complete config
  preflight before freezing the executable.
- Freeze the executable, configuration, copied checkpoint, reference products,
  and analysis registration before opening the new accounting values.
- Permit at most one replacement replay, only for an environmental
  interruption. A scientific or compatibility failure requires a revised
  packet.
- Bound the replay to 1 hour and 64 GiB, and retained new products to 8 GiB.
- Preserve the original reduction products and perform no Unity activity.

## Stop boundary and claim limits

Stop after the one replay, registered closure analysis, and result record. Do
not change a penalty factor, placement, threshold, confirmation rule,
recurrence, feedback state, source, observation, or iteration count.

EL-F10 cannot judge UID 4460, approve a safeguard, validate a detector policy,
establish generic map leverage, qualify FRUIT or JINC, establish a TolTEC JINC
numerical route, select a recurrence, launch Gate D or Stage B, change a
production default, or authorize Unity use. A successful result would explain
this case and support the design of a separately reviewed candidate safeguard
test.

## Owner choices

### Choice A — Approve the targeted accounting replay (recommended)

Approve `SCI-FRUIT-EL-F10-TARGETED-JINC-ACCOUNTING-R0.1` exactly against its
bundle manifest. This authorizes only the bounded diagnostic implementation,
tests, local build, one copied-checkpoint replay, registered analysis, and
result record above.

### Choice B — Retain EL-F9's present stopping point

Keep exact leverage and signal contrast unresolved. Make no implementation or
new-run change.

### Choice C — Revise the diagnostic test

Return a new packet with different retained quantities, compatibility gates,
resource bounds, or target scope. Nothing in Choice A is authorized.

General agreement to prepare this packet is not exact approval of Choice A.
