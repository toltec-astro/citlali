# OOF a2000 JINC readiness

The comparison is three new JINC maps of the existing cold-M0 samples for
152385/0/1,152386/0/1,152387/0/1, with historical R and ordinary S maps retained.
JINC parameters are fixed prospectively in WORK_ORDER.md: 1″, scale2mm/45m,
shape1.1/3.17/2, rmax1.5, one phase. The accepted exact-root and ceil support
remain in effect (29×29, versus historical27×27). No kernel sweep or fitted gain.
Historical R remains AltAz with its original weights/processing; new S remains
FK5 and uniform-weighted. Gridding alone cannot establish morphology equivalence
or permit paired R−S pixel subtraction.

Calibration and sample support remain bound to the completed f79db1faa OOF run:
canonical148670 matched APT-v2, Tune152384/0/1, a2000 networks11/12, native CAL,
rank3 PTC, no spatial POINT gate, no mean restoration, no feedback. Saved samples,
causes, detector occurrences, slots, native continuous AST and uniform coefficients
are authenticated. Ordinary membership, normalization and signal must replay
before any JINC application. The original scan2 publication defect is disclosed;
new identity is bound to the actual scan1 inputs and CAL/AST, never by editing
old receipts. Future native ordinary MAP publications derive the validated scope.

The OOF-approved arithmetic certificate uses each original central32″ peak as
fixed P (exact values in PREPARED.json), never a new JINC peak. Signal bound is
10^-6 P + 10^-6 |computed N/C|; denominator must be separated from zero; Q/T relative
bounds10^-6. Unavailable values remain NaN with explicit local causes. Retain
five base roles only; certificate planes are separate diagnostics. Neither Q nor
coefficient-squared time establishes variance, precision or physical exposure.

Storage reserves1GB for this campaign, within the standing shared30GB cap plus
2GB free disk reserve. Prepared dense geometry/value inputs total about334MB;
three 1″ full-map bundles and diagnostics, logs, cache and predecessor binaries
fit the remainder. Rows run serially with one numeric worker. Exact actual and
reserved bytes are checked before freeze and each application.

Launch is gated by exact committed clean source, rebuilt binary hashes, all
required gates, saved ordinary replays and independent review. The external
campaign root is
/Users/gwilson/work_toltec/local_data/citlali-validation/successor-oof-a2000-jinc-20261002.
PREPARED.json, GATES.json, review.json and FREEZE.json bind the final candidate.
No map is launched by preparation, fixtures, review, freeze or check.

After freeze/check pass, run once in an owner Mac terminal:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/gwilson/tolteca/bin/python /Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor/validation/oof_jinc_2026-10-02/campaign.py run
```

Keep that foreground terminal open, then say **done**. Codex assesses saved maps
on manual resume: common raw brightness scales, central32″/100″ and full context,
with support gaps/negative lobes preserved and native axes labeled. Any normalized
display retains its divisor and raw parent separately. No optical inference,
noise/sensitivity claim, extra observations, retry, feedback, remote access,
push, merge or POINT reopening. No remaining scientific decision is required
for this bounded mapmaking test. This is development evidence, not production
or representative Spack/Unity qualification.
