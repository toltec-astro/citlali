# Exact admitted PTC response crosswalk - r0.3

Complete sole current Stage B review-candidate normative core: SCI-FRUIT-NORMATIVE-CORE v0.1/r0.3. Canonical source-inventory SHA-256: `e6c3a536246e75739a50741de7862d664cd1ffd52ac877d5ef6b9ead45558c17`. Owner approval, freeze, Registry registration and activation are not established. Numerical methods/routes remain unavailable_pending_separate_owner_approval.

Source: [PTC_APPLICATION_REFERENCE.tex](inputs/r0.2/inputs/scientific_core/r0.4-amended/PTC_APPLICATION_REFERENCE.tex), SCI-PTC v0.1/r0.5, SHA-256 `75116261a33ec1adbb092d38da43a590624a08428e51cac32ee0c4a34f216a59`. The [r0.3 directive](inputs/OWNER_DIRECTIVE_R0.3.txt), sections 1--4, authorizes only this conditional FRUIT classification crosswalk. PTC owns the estimator and its exact fixed-state application map. FRUIT owns only the response role assigned to a declared use of that upstream operation inside a FRUIT method.

For PTC-local group/time g,t, define G = Ahat^T W Ahat and evaluate the internal coordinate mhat(x) = x W Ahat G^+. Substitute into the admitted z = x - mhat(x) Ahat^T to obtain P = I - W Ahat G^+ Ahat^T and z = x P. Holding complete operator-defining state fixed gives delta z = delta x P. The canonical [equations](src/common/equations.tex), EqPTC and EqPTCAffine, retain exact indexed symbols and detector-right orientation. This local upstream output is not by itself the complete FRUIT output.

| Scientific role | Fixed-map consequence |
| --- | --- |
| Operator-defining state/coefficients | Freeze loading, metric/influence, normal matrix/inverse, centering/reference, detector/group binding, application/output support, rank requirement/tolerance, mask, orientation/family, branch/generation and every other map-defining fact. |
| Internal application coordinate | Evaluate mhat(x) on the perturbed operand; a changed value under the same P is not relearning, re-resolution, method change, operator-generation change or RF-03. |
| Frozen numerical removed component | Y' - Uhat is a separately named affine subtraction family with identity input derivative H; it is not ordinary fixed-subspace projection. |

For uncentered calibrated input x = Y^CAL - lambda, z = (Y^CAL - lambda)P has fixed-state derivative HP. Retain the term -lambda P: fixed affine does not imply strict complete linearity. No transpose/orientation or family change is inferred.

Every required input and G must be finite; numerical rank under frozen tolerance must equal the source's required rank. Exact application/output support remains binding. Leaving this domain yields affected unavailability/cause or an explicitly typed transition, not silent RF-03, partial-rank subtraction, rank reduction, interpolation or borrowed detectors. When exact source/state cannot establish a particular P, retain that case's unavailability.

A declared rerun of loading, metric, centering, support/mask, detector/group binding, rank/tolerance, grouping, selection, threshold, branch or other defining-state resolution belongs to RF-03 or another separately typed procedure query. Evaluation of mhat alone does not. A full FRUIT derivative composes all its actual path maps; this crosswalk admits no numerical FRUIT/PTC route or fidelity claim.
