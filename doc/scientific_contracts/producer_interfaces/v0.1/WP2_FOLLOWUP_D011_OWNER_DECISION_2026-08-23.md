# WP2-FOLLOWUP-D011 Scientific-Owner Decision

Date: `2026-08-23`

Scientific owner: Grant Wilson

Status: scientific disposition and exact v0.1/r0.1 artifact approved

## Question Presented

> Should we add a bounded, package-neutral Tune/readout producer-interface
> record upstream of ALIGN, without reopening ALIGN or RTC?

## Recommendation Presented

> Yes. Bind the producer/interface version, observation and Tune/mapping
> revision, detector/tone/network/channel and native-occurrence identity,
> exact paired `x/r` identity and parent readout occurrence, transform identity
> or resolvable record, producer-owned units/sign/reference/normalization,
> applicability and validity domain, epoch, uncertainty availability, runtime
> association, and fail-closed behavior. Do not invent a new sign convention,
> make CAL interpret Tune data, duplicate frozen ALIGN/RTC mathematics, or
> place every observation payload in the frozen repository.

## Owner Response

> approved

Disposition: **approved**.

## Consequences

1. Native acquisition and \(x/r\) mapping semantics remain upstream of RTC.
2. Frozen SCI-ALIGN v0.1/r0.3 and SCI-RTC v0.1/r0.12 remain unchanged.
3. The missing source-binding artifact is tracked as the bounded
   `WP-2A_NATIVE_READOUT_INTERFACE` facet of `F-017/XOD-015`, not as a CAL
   acquisition responsibility.
4. Static interface authority and observation-instance realization remain
   distinct.
5. No new sign, reference, normalization, calibrated meaning, Stokes meaning,
   implementation-conformity claim, or MAP route is authorized.
6. The exact v0.1/r0.1 artifact was approved on `2026-08-24` and is bound by
   `SCIENTIFIC_OWNER_APPROVAL_2026-08-24.md` and `SOURCE_MANIFEST.md`.
