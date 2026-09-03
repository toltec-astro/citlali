# SCI-NOI v0.1 r0.5 STD Response-Family Amendment

Scientific owner: Grant Wilson. Date: 2026-08-30. Status: proposed final; not
frozen.

Initial STD uses `sigma_cond=sqrt(Vhat_cond)` and
`zeta_cond=m_MAP/sigma_cond`. The numerator and scale descend from the same
immutable observed parent, so the product is nonlinear and data-dependent.

When the realized scale is explicitly held fixed, the partial derivative is
`delta zeta_fixed_scale=diag(1/sigma_cond) delta m_MAP`. Its identity and claim
must say fixed-scale conditional derivative. It is not the response of the
complete STD procedure.

The complete derivative is
`delta zeta_cond=delta m_MAP/sigma_cond - m_MAP delta Vhat_cond /
(2 sigma_cond^3)`. The second term requires a separately authorized response
for the complete conditional-second-moment procedure. Until it exists, full STD
response is unavailable; dividing MAP response by `sigma_cond` cannot create or
rename it. The bundle and proposed profile carry exact numerator/scale
dependence, support intersection, response state, and cause. No significance or
detection claim follows. D-012, EQ-013-015, ASM-010, REQ-047, and PRED-022 carry
this closure.
