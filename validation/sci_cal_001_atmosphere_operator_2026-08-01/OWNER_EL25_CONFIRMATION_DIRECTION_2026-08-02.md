# SCI-CAL-001 owner EL25 confirmation direction — 2026-08-02

## Record identity

- Package: `SCI-CAL-001`
- Date: 2026-08-02
- Authority: project owner acting as calibration and atmosphere scientist
- Predecessor evidence commit: `742a3c263faf68c2de7d5b8db0d3423127f60480`
- Governing repair base: `9aae0e669384c5c0c0dda93debc194d6b8dac787`
- Decision item resolved here: `LOW-001`
- Selected choice: `confirm_el25_minimum_with_preregistered_evidence`
- Adoption status: not adopted
- Operator authorization: none
- Operational-domain authorization: none

This additive record binds the owner's response, “Let's raise that floor. This
is looking great.”, to the explicit EL25 proposal immediately presented in the
SCI-CAL-001 decision brief. It authorizes a separately preregistered numerical
confirmation with a proposed minimum elevation of 25 degrees. It does not turn
the prior post-hoc EL25 subset into confirmatory evidence.

## Scientific direction

The proposed follow-up support is

```text
zenith tau225: 0 <= tau225 <= 0.158313198574890929
elevation:     25 deg <= EL <= 80 deg
```

The opacity ceiling remains the exact repair-base q75 selector coordinate.
q95 remains excluded and historical only. The one-percent threshold remains a
provisional maximum fractional extinction-correction representation-fidelity
gate, not a physical-photometry claim and not a replacement for the later
approximately 5--10% absolute-flux and approximately 5% repeatability gates.

The confirmation must use opacity/elevation tuples that were not used to
discover the post-hoc EL25 result. It must retain the same frozen AM 12.2
source/input family, anchor constructions, candidate operators, full-sample
modified-secant airmass, top-of-atmosphere pivot, passband inputs, source
spectral indices, and SHA-256 provenance. It may use the completed v2 study as
immutable predecessor evidence but may not rewrite or relabel its artifacts.

## Choices not made by this response

This record resolves only the low-elevation response:

- `BAND-001` remains open. TolTECA v1 ECSV remains the already-frozen primary
  numerical passband family and FTS remains a challenger for this evidence
  study, but neither is selected here as the production passband authority.
- `DOMAIN-001` is constrained to the proposed 25-degree floor but remains an
  evidence-study proposal, not operational-domain authorization.
- `WARN-001` remains open for numerical selection. The bounded predecessor
  study warning contract may be retained for evidence generation only.
- `OBS-001` remains open and mandatory before production authorization.
- `SCI-CAL-001-XAUD-001` remains an open dependency. Only eligible aligned
  sample elevation with explicit sample identity, timing/interpolation origin,
  duration, and original-versus-synthesized status may eventually be used.

## Next-decision and stop boundary

This direction authorizes the EL25 follow-up path but is not a complete
confirmation-study authorization. Before a confirmation protocol is frozen or
AM runs are launched, the owner must still choose the `BAND-001` production
passband authority, complete the proposed `DOMAIN-001` support and
outside-domain policy, and choose the `WARN-001` status-1 evidence policy.
`OBS-001` remains required later, before production authorization.

Stop after binding this partial owner decision. Do not launch the confirmation,
modify Citlali application code, contact Unity, launch repair implementation,
launch the CAL re-audit, or edit the coordination registry.
