# Independent POINT152430 a2000 failure review

Date2026-09-26. Read-only review of the saved outcome under preflight-approved
source54adaeebd2068cf827fe39f0aae8c3fbddcb03e0 and prospective
2c9a38f43027b4cd3735aa61914fa5121bb23ef21baef9b297f94d363655fd41.
No fit, covariance learning, reduction, injection, feedback, threshold tuning,
or observation replacement was performed. Boolean/arithmetic authentication
used saved maps, masks, parameters, scales, bins, and candidate products.

Product root:
/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152430-independent-a2000-20260926

## Verdict

PASS: the bounded stop is the preserved candidate-support rule applied
correctly. No implementation defect was found in this outcome. The result is
insufficient evidence for the shoulder-benefit/compact-response comparison;
it is not a negative recovery result or a failed expected-source detection.

The unperturbed M0 source guard accepted one finite unconstrained fit:
peak564.3026593026557mJy/nominal beam, conditional frozen-C0 amplitude
sigma6.695788446603807mJy/nominal beam, SNR84.2772533515269,
centroid[4.208131069867596,7.36774489147099]arcsec, sigmas
[3.6317702680363695,4.073014650738158]arcsec. The guard and admission
validity checks serve different purposes; source acceptance does not waive
noise-amplitude interpolation validity for feedback.

## Direct evidence

Authenticated saved candidateSHA
478dde79217afe9f1835bb70e5945d21f1702763f62eb914d3d26b2f63cc6d39
and M0 mapSHA
68d41737e9a939d1a272153ea914ebfb74f9808a2b302607276a1aa13d13fec0
against SUMMARY; receipt and all realized-manifest entries also match.

Saved BASE source mask equals the three-M0-major-FWHM circle about the saved
M0 centroid. Saved background equals finite supported positive-coverage map
pixels outside that mask. Its9861pixels span NORMALIZATION114through456.
The last exclusive edge is nextafter(456,+infinity)=456.00000000000006.
Saved full BASE validity equals good map support intersect that coverage range.
The original point_additive amplitude rule deliberately continues the endpoint
scale outside this range computationally but marks those pixels invalid.

The41x41source stamp has53selected BASE pixels, three invalid pixels inside the
40arcsec disk, two selected-support contacts, and zero outer-disk boundary
contacts. All three invalid pixels have valid finite MAP signal/support:

| Zero-based row,column | East,north arcsec | Coverage | BASE noise validity | Touches selected candidate |
| --- | --- | --- | --- | --- |
|21,18|4,2|475|false|yes; seven selected neighbors|
|28,30|-20,16|493|false|no|
|29,17|6,18|458|false|yes; selected neighbor28,18|

Their finite endpoint-continued scale is15.438983964993769mJy/nominal beam.
Finite scale does not make extrapolated noise transfer admissible. This is
an unsupported amplitude-transfer domain under the frozen recipe, not missing
raw/MAP data or numerical non-finiteness. It does not prove physical noise is
poor at these pixels; it means the accepted method has no admitted training
support for that coverage.

Verified candidate VALID equals disk AND MAP support AND finite brightness
AND BASE noise-valid AND positive finite scale. Verified selected mask equals
the8-connected ratio>=3component at saved anchor(row24,column18), K0 equals
that selected mask, and selected candidate brightness equals saved measured
map minus its own fitted constant. No candidate was re-fitted or re-learned.

The byte-identical inherited point_fruit_trajectory.candidate_guard rejects
`binary_dilation(selected,3x3) & domain & ~valid`. Both contacting pixels
satisfy it, so `candidate unsupported contact` is the exact expected stop.
point_admission.py and point_fruit_trajectory.py are byte-identical to the
pre-campaign43844bfe07b81c3f70bd01f370f46820982f5bba versions.

## Dependency and matrix disposition

BASE-POINT REF completed M0 evaluation and stopped before T1; it is not a
completed ten-application trajectory. PHASE-base and SUMMARY agree:
actual_feedback_applications0, completed_cycle0, readyfalse. No cycle
subdirectories or SHAPES.json exist. No injected physical inputs, ideal-map
controls, or endpoint covariance analyses were warranted.

Without BASE-POINT REF T10 there is no authorized common test-shape fit.
Therefore all12response/seed trajectories (four branches per policy) are
unavailable. CORR/DIAG REF are not run under the mandated BASE-first/shapes-
freeze ordering. The supplementation code cannot repair this particular
common admission problem: it preserves every selected BASE pixel and changes
neither BASE validity nor its unsupported neighbors. This is a code-property
observation, not a new CORR/DIAG candidate evaluation.

Report one BASE-POINT REF stopped-at-M0 cell and fourteen not-executed /
dependency-unavailable cells. Preserve the distinction between accepted source,
unavailable feedback initialization, and unmeasured policy effects. No
CORR-versus-DIAG comparison or recurrence claim is supported. Do not relax
coverage validity, change masks, remove candidate pixels, substitute C0 scale,
change observation, or perform later stages to obtain success.

## Closeout obligations

Retain the M0 map, accepted fit, original three-/four-FWHM masks/scales/model,
K0, frozen geometry, candidate/validity/ratio/admission products, exact input
and prospective identities, logs/receipts, full15-cell status matrix, review,
and reproduction/stop description in the compact record. Failed continuation
does not require emptying this diagnostic candidate: expected-source fit did
not reject, and the saved candidate documents the triggering state. It must
remain unapplied.

The run-owned temporary upstream detector-time products are inactive now;
cleanup remains conditional on exact enumeration, compact-output verification,
and preservation of input/reproduction records. Raw/tune/APT inputs and
unrelated campaign state remain protected. This review confirms the scientific
stop and does not independently approve an unspecified deletion list.

Three-category disposition: scientific/behavioral PASS for correct bounded
failure; architecture/ownership PASS; evidence hygiene PASS for inspected
saved identities, with final compact publication and cleanup receipts pending.
