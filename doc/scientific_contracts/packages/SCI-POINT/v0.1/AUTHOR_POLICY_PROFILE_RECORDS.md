# SCI-POINT Draft Named-Use Policy/Profile Records

Identity: `SCI-POINT_POLICY_PROFILE_DRAFTS v0.1/r0.3`

Status: collision-free draft mechanics for final owner review; not approved

| Draft profile identity | Policy owner | Use | Current evaluability |
| --- | --- | --- | --- |
| `VAL-PROFILE/SCI-POINT/FIT-COMPLETENESS/PER-ARRAY@draft-r0.3` | POINT | decide whether one per-array fit bundle is a complete publication candidate | eligibility decision unavailable while numerical method is unavailable |
| `VAL-PROFILE/POINTING-SUPPORT/POINT-DISPLACEMENT-ADMISSION@draft-r0.3` | pointing-support producer | admit one measured displacement to correction construction | eligibility decision unavailable without displacement and owner policy |
| `VAL-PROFILE/TELESCOPE-QC/POINT-PARAMETER-ADMISSION@draft-r0.3` | named telescope-QC process | admit exact POINT metrics to QC comparison/action | eligibility decision unavailable without metrics and owner policy |
| `VAL-PROFILE/CAL-TOLPROJ/POINT-AMPLITUDE-TRANSFER@draft-r0.3` | CAL/TolProj | admit exact fitted amplitude for photometric transfer | eligibility decision unavailable without amplitude and owner policy |

Each record binds profile version/digest, policy owner, named use, exact source
facts and source bindings, separate request/applicability/eligibility/
realization fields, required unavailable-state behavior, result reasons,
prescribed consumer action/use mode, evaluation identity, and provenance.
Eligibility is exactly `eligible`, `ineligible`, or `decision_unavailable`.
`diagnostic_display_only` may be a prescribed action for an eligible diagnostic
use; it is not an eligibility value and cannot rescue another use.

VAL owns registry identity, source binding, evaluation, and result provenance.
VAL does not author any policy, combine uses into a universal flag, or let one
profile or diagnostic action rescue a datum excluded by another policy for the
same declared use.
No aggregate or universal eligibility profile enters base v0.1.

Where relevant, applicability consumes the separately typed `known`,
`isolated`, `bright`, and `approximately_centered` facts and their causes; a
profile may not recreate them from implementation defaults or diagnostic
ratios.
