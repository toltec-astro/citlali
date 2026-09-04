# SCI-FRUIT EL-F12 — Two-stage cap decision

Date: `2026-09-04`

Decision: `SCI-FRUIT-EL-F12-CAP-001-R0.1`

Status: **proposed amendment; not approved or implemented**

## Decision requested

Approve one explicit rule: **Half may attenuate a selected detector only if
at least one applicable ordinary exclusion stage permits that detector's
hard action in the current iteration.** If neither stage permits it, Half
keeps its ordinary map coefficient. Independent exclusions and flags still
take precedence.

The owner already approved Choice A for the
[EL-F12 screen](EL_F12_RESPONSE_AWARE_INTERVENTION_DESIGN_R0.1.md), as recorded
in the [authorization](SCIENTIFIC_OWNER_EL_F12_AUTHORIZATION_2026-09-04.md).
That approval remains in force. This decision fills one missing rule in its
action eligibility; it does not seek approval again for the whole screen.

## Why a decision is needed

The approved design says to evaluate the unchanged cap at the ordinary
application point and to withhold Half if that gate rejects the hard action.
The actual pointing path has two such points: before raw time-chunk processing
(RTC) and before processed time-chunk cleaning (PTC). Detector removal and
other flags can change the state between them. The proposal did not say how
to combine different outcomes from these two checks.

The existing handler was tested with synthetic boundary states: three proposed
detectors among 100 exceed the 2% cap before RTC. If two of those detectors
are absent at the later boundary, the remaining proposal is 1/98 of the PTC
population and is accepted. With no removals, both checks reject. The exact
2% boundary is accepted. All eight focused handler tests passed; these tests
did not process observation data or run an experimental action. See the
[evidence record](../../../../../../validation/fruit_loop_el_f12_preimplementation_2026-09-04/CAP_BOUNDARY_TEST_RESULT_R0.1.md).

Treating either rejection as a permanent veto would withhold Half in the
first example even though the later historical stage would apply the hard
action. The recommended rule follows whether there is an eligible hard
action to replace. This is a scientific choice about which occurrences are
attenuated, not a routine implementation repair or a new empirical result.

## Exact proposed amendment

This text supplements only the application-cap paragraph in the approved
design's “Three fixed action arms” section. The original review bytes remain
unchanged. If approved, this text controls composition of the two stage gates:

1. At each ordinary stage, evaluate its existing cap on the complete proposed
   set before any experimental suppression. Use the current arm's actual
   pre-action flags and detector population. Preserve the existing threshold,
   strict over-cap comparison, source protections and other-reason precedence.
   Do not use another arm's state, a resulting map, or a reduced proposed set.
2. For each selected key, record whether its map-dominance hard record is
   present, matches a detector and is permitted by that stage's cap. A missing
   key or inapplicable stage supplies no permission. A rejected stage applies
   neither that hard action nor a replacement at that stage.
3. In Hold and Half, suppress only a permitted selected map-dominance hard
   action at either stage. Other reasons and flags retain their ordinary
   behavior. Neither arm restores independently excluded samples or detectors.
4. At final map accumulation, Half uses coefficient 0.5 for an otherwise
   admitted occurrence of that selected key if either applicable stage
   permitted its hard action during this iteration. Otherwise it uses 1.
   The two permissions never stack; Hold always retains ordinary coefficients.
   A later rejection does not revoke an earlier permission, and an earlier
   rejection does not veto a later permission.
5. Save the separate stage receipts and the combined eligibility with
   observation, array, UID, scan and iteration identity. Recompute eligibility
   at the actual stages each iteration; do not carry a previous iteration's
   permission forward. The selected assignment itself retains its approved
   fixed horizon through iteration 6. Restart must reproduce these decisions.

| Before RTC permits this key | Before PTC permits this key | Half coefficient for an otherwise admitted occurrence |
| --- | --- | --- |
| No | No | 1 |
| No | Yes | 0.5 |
| Yes | No | 0.5 |
| Yes | Yes | 0.5 |

Missing or inapplicable stages count as “No” only for this permission table;
receipts must preserve their distinct reasons. Invalid or inconsistent state
remains an unavailable trajectory, not a default permission.

## Scope and resumption

H0 and H retain the historical action and recurrence. The causal selector,
cutoffs, inputs, action duration, source and morphology protections, leakage,
support, convergence, runtime and memory reporting, eight primary trajectories,
conditional restart checks, replacement allowance and all resource ceilings
remain as approved. No independent-pointing replication, policy recommendation,
Gate-D launch, qualification, Stage B, production or Unity work is added.

Approval of `SCI-FRUIT-EL-F12-CAP-001-R0.1` against the
[amendment manifest](EL_F12_TWO_STAGE_CAP_MANIFEST_R0.1.md) allows implementation
to resume with this rule and the existing Choice A authorization. Before any
trajectory, the complete prototype must still pass the original local gates
and have its executable, analyzer, inputs and configuration registered.

The narrow decision is pending. Only input verification, source inspection,
existing-handler tests and review preparation have occurred; no intervention
has been implemented and no EL-F12 replay has begun.
