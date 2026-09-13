# Six-branch POINT feedback-sensitivity experiment

2026-09-13 · r0.1 · Prospective diagnostic protocol.

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md) and
[reviewed recovery](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md).
Adopt the accepted immutable-parent, residual-relearning, total-sky replacement
recurrence and existing PTC/MAP responsibilities. Preserve the
[evaluator repair and failed stopping qualification](../fruit_point_bounded_repair_2026-09-12/SCIENTIFIC_REPORT.md).
This diagnostic remains outside the frozen scientific author channel.

## Authority and question

Decision `SCI-FRUIT-POINT-PAIRED-FEEDBACK-SENSITIVITY-2026-09-13` records the
owner's [exact supplied directive](OWNER_DIRECTION.md): examine saved solution
differences, their constrained evidence and actual feedback projection, then
perform **six one-step branch evaluations on three states**. No new starlet
optimization, regularizer, estimator, acceptance gate or full trajectory follows.
The earlier qualification remains failed. Diagnostic use of its unavailable
models is authorized only here. No production, Unity, 129081 or policy approval.

Question: do the differences between nominal (first 1e-4 stop) and tighter
(first 1e-6 stop on the same optimizer path) models materially affect PTC
learning and the next reconstructed-total source measurement? Neither model is
astronomical truth. A large image peak change alone does not diagnose a null
space, need for regularization, or source-recovery failure.

## Saved-pair screen, with no new learning or optimization

Use every available pair from the prior 144 saved problems: 141 pairs, including
84 exact-zero pairs, plus an explicit unavailable record for the three missing
tighter comparators. These cover all 48 retained candidate map states. Preserve
their array identities; placeholder zeros for a missing array in a projection
batch are never interpreted as an available pair. Replay states require all
three arrays to have both solutions.

For delta = tighter minus nominal, record both maxima's coordinates, coverage,
distance from commanded origin and from the original total-map fit, and complete
wavelet support. Partition squared image difference into a data-only source
core (the original total-map Gaussian >=10% of fitted positive peak) and its
complement. Also report fixed rim (52<r<=60), low-coverage stratum, concentration
in the top 1% of domain pixels, and connected components of |delta|>=10% of
its maximum. These overlapping descriptions are not gates or an energy partition.
Core definitions cannot set the centroid-usability label.

Apply the existing five starlet analysis bands to delta for scale diagnostics;
their squared norms are not orthogonal energy fractions. Record the fixed
selected coefficients, their uncertainty-scaled target and change norms,
nominal/tighter objective, remaining-residual reduction and reduction relative
to the zero-model objective. A small objective change relative to the data norm
can coexist with a large fractional reduction in a small remaining residual;
report both. No new thresholds classify underconstraint automatically.

## Existing feedback projection

Read only the already-bound 123424 parent export and the existing geometry and
flags. Use the unmodified `Data.project` from the frozen experimental base:
2-arcsec containing-pixel map values sampled at each valid detector occurrence,
with inherited calibration/units, no extra smoothing or weights. Compute both
projected models and subtract them, verifying linearity against projection of
the difference. This is the actual feedback path of this experimental method,
not a claim of production conformance under the frozen generic contract.

For every available pair, report valid-sample difference energy, amplitude and
nonzero count, source-crossing energy, and detector-group summaries. For every
inherited network/chunk group, report difference before/after the usual
per-detector mean removal, the shared time-varying mean across valid detectors,
and the analogous mean across source-crossing detectors. Weight mean-waveform
energy by its contributing detector count. Retain the group waveform and
contributor counts. Record coupling of the centered difference to the saved
preceding removed subspace as a diagnostic. None of these is a noise-clearance
criterion, and no PTC basis is learned in this screen.

## Three deterministic replay states

Freeze this selection rule before the screen, then record the chosen identities
and their source hashes before any branch evaluation:

1. Ordinary compact: H, nuisance seed 20260911, saved pass 0.
2. Among other saved H/H-shift/D/T states having all three pairs, choose the
   greatest **absolute projected squared difference inside the data-only source
   core in any array**. This measures an actually sampled source-region change;
   full-field and group effects remain visible separately. Ties use case name
   then pass index in ascending order. No next-map measurement or truth error
   is available to this selection. H chosen in item 1 is excluded to keep three
   distinct states.
3. Real data: real123424, saved pass 0, fixed without inspecting replay outcomes.

These are three-array states, not three selected detectors. Each receives one
nominal and one tighter branch: exactly six `Data.clean` calls, at most, without
a fallback or another iteration. Alternate nominal/tighter execution order by
state. Preserve failures and stop on an unexpected failure; do not add calls.

## One-step branches and measurements

Reconstruct each synthetic immutable parent using its original saved truth,
fixed nuisance seed/scales and exact T scaling where applicable. Check it against
the original parent hash. Both branches start from the same preceding saved
total map, applied model and learned-state identity. The experimental recurrence
relearns from the immutable parent on every pass, so previous learned coefficients
are provenance, not reused inputs. Substitute only the saved nominal/tighter
feedback model; subtract it, relearn rank-5 PTC per existing network/chunk,
clean and restore that same model through the normal path, then map normally.
Do not infer another feedback model. No injected location, flux or width enters
the unchanged evaluator. Truth scores the synthetic outputs separately.

Primary outputs are next-total-map fitted peak and centroid changes, their
separate repaired usability judgments, and truth-scored peak/centroid/width
errors for the injected states. Retain signed residual structure, aperture
brightness, support, exterior RMS, rank guards, failed readouts and elapsed time.
Raw differences remain visible if a readout is withheld; no substitute fit is
chosen to make it available.

Compare the rank-5 removed subspaces using singular values of A_nominal^T A_tight:
principal angles and normalized projector distance sqrt(1-mean(cos(angle)^2)).
This is invariant to eigenvector signs and rotations within the subspace. Keep
per-group and per-array results, not only a whole-observation average. Do not
perform a frozen-basis control or extra cleaning calls.

Keep 0.5% peak and 0.1 arcsec allocations visible alongside the 5% and 1-arcsec
operational scales. They describe this diagnostic's effects; they do not replace
the failed image-stability gate or establish trajectory stability. Report truth
scores for both solutions rather than assuming the tighter solution is better.

## Bounds and disposition

Limit the whole experiment to one hour, 8 GiB peak RSS, 4 GiB retained output,
and six cleaning calls. Use four compute threads. A hard campaign alarm and
between-stage/group checks enforce bounds. Preserve existing products and both
opaque review archives (presence/status only). Repair routine implementation
defects only within the registered method, bounds, inputs and call count.

Recommend a formulation change only if consequential reconstruction freedom is
supported by projected, subspace and next-map evidence with little change in
the constrained coefficients. If the objective improves substantially, distinguish
remaining numerical progress from weakly constrained freedom. If next-map effects
are small, a prospectively revised feedback-relevant criterion may be justified.
Mixed evidence should remain mixed; no automatic cure or follow-on campaign.
The candidate still needs demonstrated scientific benefit and independently
authorized replication before a policy recommendation.
