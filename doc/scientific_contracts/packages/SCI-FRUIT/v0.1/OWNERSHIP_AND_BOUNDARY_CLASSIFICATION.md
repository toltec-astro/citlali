# SCI-FRUIT v0.1 — Ownership And Boundary Classification

Status: Stage A owner-review candidate

This table prevents iterative execution from absorbing adjacent scientific
authority. “May consume” always means only through an exact, compatible,
numerically available product admitted by an owner-approved FRUIT route.

| Process/package | Retained authority | SCI-FRUIT may own | SCI-FRUIT must not infer or redefine |
| --- | --- | --- | --- |
| SCI-RTC | RTC-requested calibration/conditioning, sample-domain identities, units, validity, and RTC-owned learned state | Exact binding of an admitted RTC realization into a FRUIT generation | RTC coefficients, timing, units, validity, or whether an unavailable RTC route is numerically usable |
| SCI-PTC | Cleaning estimator, coefficient families, cleaning state and validity that PTC authority assigns to itself | Whether and when an admitted PTC operation participates in FRUIT recurrence; FRUIT-level carry/reset/relearn binding | PTC estimator, coefficient values/families, normalization, unavailable numerical routes, or PTC validation meaning |
| SCI-MAP | Ordinary observation/coadd map estimands, normalization, grouping, response/covariance roles, support, validity, and route availability | Whether a specific MAP product may seed or inform a feedback model and how FRUIT updates from it | That a normalized MAP signal is automatically a feedback sky model; missing response/covariance; numerical `coverage_cut` or PTC coefficients |
| SCI-JINC | JINC observation-map estimator, signed normalization, geometry, response/covariance roles, support, validity, and route availability | Whether a compatible JINC product may seed or inform a feedback model | A JINC map as an automatically projectable sky model; a base-v0.1 JINC coadd; missing coefficient/array/adequacy authority |
| SCI-FLT-FIXED | Fixed deterministic transform, exact operator/state/product/response/support/validity meanings, and typed unavailable routes | Admission of an exact transformed product to a separately defined FRUIT model-construction route | Inversion/deconvolution, transformed signal as an unfiltered model, profile registration, or unavailable MAP/JINC numerical parents |
| provisional SCI-FLT-MATCHED | No current authority for FRUIT; exact holding study remains provisional | Only a future exact binding after that package is independently approved and FRUIT owner admits it | Package approval, final name, numerical route, JINC route, covariance, learned-state policy, source catalog, or sky reconstruction |
| SCI-NOI | Generation/randomization/uncertainty method, ensemble identity, fixed-state versus replay method, and truthful uncertainty scope | Selection of the FRUIT state/procedure to which an admitted NOI method applies; FRUIT successor-generation graph | Physical-noise equivalence, independent validation, covariance completeness, significance, or pooling fixed/replayed members |
| SCI-VAL | Registry and evaluation identities, admission mechanics, evidence/result lineage | FRUIT method/profile requirements and claims that a later VAL record may evaluate | A registered profile, successful evaluation, validation result, or numerical availability |
| SCI-FRUIT | Feedback-model identity, model construction/selection, forward projection, subtraction/add-back meaning, recurrence/update, state/generation, stopping, restart, response, support, validity, failure, and terminal product identity | Its complete bounded scientific contract | Authority already retained by another package |
| source fitting/catalog | Source detection, candidate/peak selection, fitting, deblending, source parameters, and catalog inference | At most an explicitly bounded internal model-selection operation if the owner assigns it to FRUIT without creating a catalog claim | A source catalog or fitted-source product from a map, matched-filtered field, or FRUIT diagnostic |
| Pointing/OOF | Pointing and OOF estimands, calibration, transfer construction, terminal consumer criteria | Exact terminal/iteration identity and response disclosures those consumers require | Pointing/OOF calibration, transfer, or fitness-for-use |

## Boundary Tests

An interface is not closed merely because shapes or filenames match. Every
admission must answer all of the following:

1. Is the exact upstream method/version/generation numerically available?
2. Is the product role a scientific signal that can legitimately inform the
   chosen FRUIT feedback-model estimand?
3. Are units, calibration, grouping, WCS/grid/frame, response, null space,
   support, validity, missing/non-finite policy, covariance status, and
   lifecycle compatible?
4. Does FRUIT define an explicit model-construction operator rather than
   treating the parent signal as the model by coincidence?
5. Is the forward projector defined for that model's identity and response?
6. Does the route preserve immutable parent and generation identity through
   every iteration and terminal product?

A negative or unresolved answer leaves the route typed unavailable.
