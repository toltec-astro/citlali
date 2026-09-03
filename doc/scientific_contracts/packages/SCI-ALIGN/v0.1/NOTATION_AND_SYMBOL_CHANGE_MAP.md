# SCI-ALIGN Stage B Targeted r0.3 Notation and Symbol Change Map

Status: author-draft change record; not scientific approval

Prepared: `2026-08-22`

| Prior Stage B notation or wording | Targeted r0.3 authority | Semantic effect |
| --- | --- | --- |
| `n` stable ALIGN detector-reference slot | `s` stable ALIGN detector-reference slot | Stable ALIGN identity is `(observation,s)` across package and window boundaries. |
| `j` local row, sometimes adjacent to stable identity wording | `j` local storage row only | Never external identity; every view carries an explicit `s`-to-`j` relation; downstream reconstruction of `s` from `j` is prohibited. |
| no reserved RTC sample symbol in ALIGN notation | `n` stable RTC output-sample identity | Prevents ALIGN/RTC identity collision. |
| reference interface `r=D` | `i_ref=D` | Reserves `r` for the physical KID readout coordinate. |
| corrected time `t^(r)_ik` | `t^ref_ik` | Separates detector-reference time from the KID `r` coordinate. |
| offset `delta_(i->r)` and `delta_(r->r)` | `delta_(i->ref)` and `delta_(ref->ref)` | Makes reference direction explicit; positive-add and exactly-once semantics unchanged. |
| product-state tuple `Q` | `mathcal S_ALIGN` | Avoids collision with acquired microwave quadrature `Q` in `(I,Q)^acq`. |
| generic non-KID field operands `x_a,x_b` | `v_a,v_b` | Reserves lowercase `x` exclusively for the physical KID readout coordinate in field interpolation and variance formulas. |
| generic stacked inputs `boldsymbol x_D,x_T,x_H,boldsymbol x` and containers `X_D,X_Tv,X_Hv` | `boldsymbol v_D,v_T,v_H,boldsymbol v` and `V_D,V_Tv,V_Hv` | Gives detector/telescope/HWPR input containers a neutral value family without changing the paired KID invariant. |
| generic covariance/input expectation `C_x`, `boldsymbol x` | `C_v`, `boldsymbol v` | Separates generic response/covariance operands from the KID `x` coordinate. |
| circular difference `(-P/2,P/2]` | `[-P/2,P/2)` | Establishes exact shared convention; exactly antipodal interpolation remains unavailable absent explicit unwrap authority. |
| “Stokes-I use only” / “Stokes-I support” | “ordinary nonpolarimetric detector-signal path” | Does not call raw KID coordinate `x` Stokes I; HWPR facts still authorize no polarization operation or Stokes reconstruction. |
| `p` as detector identity | `d` as detector occurrence/stable identity; `p` reserved for map pixel | Establishes the shared `s/j/n/d/p/x/r` convention and removes detector/pixel aliasing. |
| `e_sp` called acquired exposure but restricted to valid original payload | `e^acq_sd` physical acquired exposure and `e^vo_sd` valid-original exposure | Separates physical integration from scientific usability; original-invalid may retain physical exposure, while synthesized/missing add zero. |
| current sample time listed as an interpolated observing-state field | exact occurrence time `t_s`; observing state evaluated or mapped at `t_s` | Prevents a producer timestamp or interpolated scalar from competing with the detector-reference grid/time identity. |

The symbols `x` and `r` are reserved exclusively for the paired physical KID
readout coordinates throughout definitions, equations, requirements,
predictions, figures, response/covariance prose, provenance, and the
ALIGN-to-AST boundary. The horizontal audit found and removed every generic
non-KID `x` operand from the formal authority and narrative.

Stable requirement, prediction, equation, assumption, and owner-decision IDs
were not renumbered. This map records a targeted Stage B draft revision, not
scientific-owner approval or implementation conformity.
