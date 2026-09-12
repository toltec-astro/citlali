# POINT operational purposes and FRUIT responsibilities

2026-09-12. Scientific owner: Grant Wilson.
Status: **owner-established purposes; successor test-gate design requested**.
Decision identities: `SCI-FRUIT-POINT-USES-2026-09-12` and
`SCI-FRUIT-POINT-PEAK-GAIN-2026-09-12`.
Repository recovery: `codex/sci-fruit-historical-control-method-scope-r0.1`,
HEAD `e8acee65e120267a2ec3bb3b2bb50565abe93ee6`, tracked tree clean.

## Program adherence and prior-work recovery

Follow the [program charter](r0.4/inputs/program/README.md),
[downstream roadmap](r0.4/inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md)
and [reviewed prior work](r0.4/PRIOR_WORK.md). Adopt the frozen generic core,
[ordinary-MAP definition](r0.4/METHOD_DEFINITION.md) and
[mode-aware owner direction](r0.4/inputs/owner/LATER_MODE_AWARE_DIRECTION_2026-09-09.txt).
Retain the [accepted POINT brief](POINT_REFERENCE_BRIEF_ACCEPTANCE_2026-09-10.md)
and its exact accepted bytes. This later owner direction broadens its account
of POINT's purposes; it does not retrospectively alter numerical test results.

Cite the [innovation direction](../../../../../../../../validation/fruit_point_coherent_feedback_2026-09-11/OWNER_DIRECTION.md)
and [central-domain result](../../../../../../../../validation/fruit_point_starlet_central_domain_2026-09-12/SCIENTIFIC_REPORT.md)
as implementation-informed development context. The new work is a
[use-specific gate proposal](POINT_USE_CASE_GATE_DESIGN_R0.1.md), outside the
independent scientific-author channel. No frozen package or numerical method is
amended, no fresh author is commissioned, and no additional experiment is
authorized by these notes.

## Owner's operational account

Source: the owner's direct message in the current Codex task on 2026-09-12;
not an inference from the prior ChatGPT discussion. The five uses were:

> 1) we POINT at the start of the night as an initial assessment of two things: are we seeing *any* astronomical flux (a sanity check of the system) and is the telescope wildly out of focus. Even when things are pretty bad after this map we switch to OOF to focus and correct the deformations.
>
> 2) we use POINT regularly before and after OOF to assess the change in gain due to OOF and to recenter the source in the field. The telescope can be deformed or slightly out of focus here but it is not expected to be wildly wrong.
>
> 3) we use POINT throughout the night to make small offset corrections (or just measure them) to the pointing.
>
> 4) we use POINT throughout the night to check to see if the surface or focus has degraded enough to warrant an OOF
>
> 5) we use POINT throughout the night to check the health of the detector tunes. For example, if the total loading changes dramatically due to clouds coming in, this shows up in our POINT maps first.

The owner then directed: **“Let's capture these use cases in our notes and then
design our FRUIT testing gates around them.”** This authorizes the gate-design
work, not selection or execution of another candidate.

For use 2 the assistant asked whether the primary gain measure should be the
change in peak response to the same source, with compatible calibration and
observing-condition corrections, while integrated brightness and shape remain
diagnostics. The owner's explicit answer was **“Yes—peak response is primary.”**
This resolves the primary observable. It does not select a peak estimator,
numerical accuracy, condition correction, or operational OOF trigger threshold.

## What each use needs from a POINT product

| Use | Operational question | Required information | What is not required for this purpose alone |
| --- | --- | --- | --- |
| U1 — startup assessment | Is astronomical signal present, and is the telescope grossly out of focus or deformed? | Credible source evidence, broad location when measurable, visible gross shape/degradation, and an honest insufficient-data state. Retain each array's result. | Accurate recovery of every faint wing, precise photometry or a valid small pointing correction from a badly distorted source. The next focusing/deformation operation is OOF. |
| U2 — before/after OOF | Did peak response improve, and how should the source be recentered? | Same-source relative peak response, freely measured centroid and their uncertainties; width, distortion and integrated brightness as diagnostics. Allow mild defocus/deformation. | A complete OOF reconstruction or an assumption that the pre-OOF source has an ideal beam. |
| U3 — routine pointing | What small pointing offset is present? | Accurate, repeatable centroid offsets with valid uncertainties and acceptable latency. | Universal 5% recovery of all wings or absolute flux solely to obtain an offset. |
| U4 — degradation monitoring | Has focus/surface performance deteriorated enough to consider OOF? | Changes in peak response and shape relative to a suitable reference, their uncertainties and confounding conditions. Preserve evidence of deformation. | A FRUIT solution for focus/surface corrections, or unique causal attribution from a POINT map alone. |
| U5 — tune/condition monitoring | Does POINT reveal a detector-response or observing-condition problem? | Array-specific response changes, map/residual abnormalities, coverage and available upstream detector/condition quality facts. An unreliable measurement must remain visible as such. | A FRUIT retuning algorithm, a weather diagnosis, or proof from a single map that a problem is uniquely caused by detector tunes. |

A single observation may serve several uses. Declare the requested uses before
evaluation and return a separate status for each. A useful startup warning is
not automatically a valid pointing correction, and a valid centroid does not
establish healthy photometric response. Detecting a source in one credible
array can answer “any flux?” while leaving other arrays explicitly unassessed
or unhealthy; it must not manufacture their detections or amplitudes.

## Scientific and ownership consequences

The proposed relaxation is **purpose-specific accuracy**, not permission to
make a damaged beam look healthy. U1 can succeed by exposing a badly distorted
source and supporting an OOF handoff. U2–U4 require the relevant centroid,
response and shape changes to survive processing. A visually compact feedback
model cannot substitute for those measurements.

Keep the bootstrap ordinary total, each reconstructed total, residuals,
support/quality information and the internal feedback model distinct. The
POINT evaluator measures the appropriate total product; it does not take the
feedback model's imposed shape as evidence of telescope health. A fitted
background remains nuisance information, outside astronomical feedback.

FRUIT continues to infer replacement feedback from reconstructed total sky,
subtract the current model from the immutable pre-PTC parent, and request PTC
relearning on every pass. PTC owns correlated-signal processing. POINT owns its
measurement/use decision; OOF owns its solution and corrections; the relevant
upstream producers own detector and condition facts. This adds neither five
FRUIT implementations nor new telescope-control responsibilities.

No known source flux, nominal position or nominal width may be imposed to
manufacture stability. Relative array alignment does not fix the absolute
pointing offset. Any numerical cross-array prior still needs its separately
declared evidence, scope and test.

## Effect on the current development result

The completed central-domain test supports limiting where POINT source
evidence is sought in its tested cases. It does not make 60 arcsec an operational
search limit or require astronomical feedback to end at that radius. Source
evidence, model extent and the surrounding computational/background map are
different domains.

All earlier candidates retain their recorded dispositions. In particular,
the starlet source solves remain unavailable under their frozen completion
rules; no unfinished iterate becomes accepted feedback through a change of
use. Full-image errors from the comatic stress case remain useful evidence,
but the new proposal does not make faithful recovery of that entire image a
universal prerequisite for answering every POINT question.

The next review is the [prospective gate design](POINT_USE_CASE_GATE_DESIGN_R0.1.md).
It distinguishes already accepted purposes and peak-gain meaning from proposed
gate applicability, numerical targets and a bounded future test set. Existing
products and the two protected opaque review archives remain preserved.
