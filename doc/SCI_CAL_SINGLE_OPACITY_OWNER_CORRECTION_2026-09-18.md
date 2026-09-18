# CAL opacity: owner correction r1 — 2026-09-18

Status: scientific-owner approved; implementation and review evidence remain
separate. Scientific owner: Grant Wilson. Policy identity:
`SCI-CAL-WVR-owner-2026-09-18-r1`. Successor scientific epoch:
`cal-opacity-single-or-interpolated-2026-09-18`.

The owner independently confirmed that the 152390 telescope file contains
only one opacity reading and is investigating recording behavior with the LMT
engineer. The owner explicitly corrects the scientific contract:

> when we have multiple values, we should interpolate through them but when
> we have a single value we use that as a constant value throughout the observation.

This is a narrow amendment of SCI-CAL Q06 and the
[approved WP7 addendum](scientific_contracts/audits/WP7_TIMESTREAM_CLEAN_ROOM_F01E22F5F/REPAIR_AND_CLOSURE/WP7_APPROVED_SCIENTIFIC_AUTHORITY_ADDENDUM_2026-08-25.md)
sections 2.1, 2.2 and their dependent section 3 coverage rules. Original frozen
documents, numerical atmosphere nodes, operator, passbands and earlier evidence
remain unchanged. The new policy supersedes the blanket prohibition on using
an observation-associated scalar reading; it does not infer policy from legacy
code or require a new contract derivation. Other scientific decisions remain
binding. This amendment is discoverable from the canonical authority router,
intentional-science-change ledger, living status and continuing handoff.

## Scientific rule and engineering interpretation

1. For exactly one distinct, admitted opacity reading associated with the
   observation, set `tau225(t) = tau225_single` throughout that observation's
   full original time interval. Apply to short POINT/OOF and longer BEAM/science
   observations alike. Do not impose a newly invented age or duration cutoff.
   A header reading can precede the observation; its association is established
   by its owning telescope artifact and verified observation identity.
2. Use `cal_wvr_tau225_single_observation_constant_v1`. Retain the original
   source/record identity, value, supplied update metadata and exact observation
   bounds. Do not manufacture two endpoint measurements or claim time-series
   interpolation. A source timestamp is not needed to evaluate a constant;
   unconverted update text remains text, not an invented mapped measurement time.
3. The observation bounds are the full original ALIGN-assigned native interval,
   before filtering, masking or decimation, including its last native occurrence.
   CAL checks this exact binding. The constant does not authorize data outside
   the observation or restore RTC-invalid/replaced measurements. Supplied
   producer-invalid status remains invalid; record support fields describing
   measurement-time validity do not truncate the owner-authorized constant.
4. For two or more distinct source-time records, retain the existing
   `cal_wvr_tau225_linear_detector_time_v1` interpolation and source validity.
   Identical duplicates at the same time may collapse as already approved;
   distinct times with equal values remain multiple readings. Do not discard
   invalid records to manufacture a singleton. Conflicting duplicates stay
   unavailable. Multi-reading extrapolation and endpoint holding remain
   unselected and are not introduced by this correction.
5. Absent readings remain unavailable; negative/nonfinite values remain invalid
   atmosphere; the numerical operator's opacity/elevation domain still applies.
   Zero is a legitimate measured opacity, never an absence substitute. No
   invented uncertainty, climatological value or unity correction is authorized.
6. Compute observation opacity summaries from the selected representation.
   For a constant, mean/minimum/maximum equal the single reading and excursions
   follow that constant over the observation. Label these as consequences of
   the constant assumption, not measured atmospheric stability or resolved
   extrema. Existing classification thresholds are unchanged. WVR uncertainty
   and unmeasured variability remain unavailable.
7. Constant **opacity** does not mean a constant attenuation correction: use
   each output occurrence's actual AST elevation in the unchanged atmosphere
   operator. Apply the selected APT factor and atmospheric correction once.

## Runtime boundary and verification

CAL Learn retains the input inventory, selected representation and observation
support. Consider freezes the resulting per-occurrence multiplier and causes;
Apply consumes the actual RTC output. VAL retains the realized CAL facts.
These runtime boundaries are distinct from the engineering learn/consider/apply
workflow. No RTC filtering, masks, paired validity, donor policy, PTC/MAP or
FRUIT changes accompany this amendment.

The known telescope ingress admits `Header.Radiometer.Tau`; it preserves
`Header.Radiometer.UpdateDate` without assuming an unverified date conversion.
The known unused Tau2=0/empty-update placeholder is not a second measurement.
A populated second header or unknown time-series layout must receive an
explicit adapter rather than being silently ignored. General new instrument
formats are outside this correction.

Conformance controls cover singleton bounds, missing source time, duplicates,
invalid values, absent input, unchanged multi-record interpolation, constant
summary semantics, exact CAL/observation binding, varying-elevation correction,
and byte-identical RTC support/numerics in the same ordinary 152390 run.
Exact committed source, result, local environment, independent review and
integration disposition are bound externally under
`/private/tmp/citlali-successor-cal-single-opacity-001-20260918`.
