# SCI-RTC v0.1 — Scientific Owner Decision Ledger

Status: Stage A draft; scope and packet approval pending

State vocabulary: `open`, `decided`, `deferred`, `superseded`.

| Decision ID | Owning scientific authority | State | Evidence or decision required | Exact blocked claim or output | Resolution authority | Resolution date | Affected documents |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SCI-RTC-OD-001` | Primary `xs` signal producer with SCI-CAL/SCI-BEAM role owners | open | Physical observable, sign, reference/baseline, raw unit, calibrated-role unit, product-role admission matrix beyond the frozen Beammap boundary, and any raw-domain donor-to-target transfer authority | Universal interpretation of `xs`; use of a CAL operator in every non-BEAM role; scientifically interpretable raw-domain cross-detector replacement | Grant Wilson or named signal/CAL authority | — | Rationale; formal contract; adjacent CAL/BEAM interfaces |
| `SCI-RTC-OD-002` | RTC scientific owner | open | Admitted despike/donor, source-protection, FIR/IIR/notch, state, edge, non-finite, fallback, and product-role policy families plus their change authority | Any claim that one universal conditioning chain or current default is scientifically required | Grant Wilson | — | Rationale; formal contract; validation plan |
| `SCI-RTC-OD-003` | RTC learned-sampling scientific owner | open | Compact-source loss/broadening limit; alias/stopband limit; minimum sampling criterion; factor set; maximum filter cost/order; role-specific native fallback/failure; common-cadence array-filter policy | Resolution or application of a learned plan; numerical sampling adequacy claim | Grant Wilson | — | Rationale; formal contract; learned-plan successor; validation plan |
| `SCI-RTC-OD-004` | RTC scientific owner with affected uncertainty consumers | open | Whether and where data-derived despike, donor, dynamic-notch, mask, or learned selection uncertainty must be calculated rather than explicitly unavailable | Unconditional unbiasedness, total covariance, or calibrated significance after data-derived selection | Grant Wilson or named uncertainty owner | — | Rationale; formal contract; PTC/NOI interfaces |
| `SCI-RTC-OD-005` | RTC response authority with consumer owners | open | Accepted factorized/local response representations, approximation/error bounds, domain, and consumer-specific minimum response class | Any complete-response claim based on a reduced scalar or partial kernel; downstream deconvolution/beam qualification | Grant Wilson or named RTC response owner | — | Rationale; formal contract; PTC/MAP/BEAM/MAP-003 interfaces |
| `SCI-RTC-OD-006` | RTC uncertainty authority with PTC/NOI/VAL | open | Required conditional variance/covariance products by role, permitted factorization/subsets, nuisance correlation scope, and explicit unavailable cases | Full precision, total uncertainty, independent-noise weight, or absolute significance | Grant Wilson or named uncertainty owner | — | Rationale; formal contract; downstream contracts |
| `SCI-RTC-OD-007` | ALIGN timing authority | open | Accepted physical integration-event definition, producer epoch/phase, timing uncertainty, and any correction transform | Physical event time, absolute timing accuracy, sub-sample sky placement, or timing correction | Grant Wilson or named ALIGN owner | — | Rationale; ALIGN contract; response/astrometry validation |
| `SCI-RTC-OD-008` | RTC/VAL consumer-policy owners | open | Consumer-specific eligibility and cause precedence beyond the binding rule that synthesized/replaced influence is ineligible | A universal eligibility bit or broader admission of an influenced sample/product | Grant Wilson or named VAL owner | — | Rationale; formal contract; VAL/PTC/MAP/BEAM interfaces |

Open decisions remain visible in the eventual rationale and formal contract.
They do not authorize an implementation default to become scientific policy.
Stage B may proceed only after the owner confirms that these questions may
remain open with the listed fail-closed consequences or supplies decisions
that materially change the author task.
