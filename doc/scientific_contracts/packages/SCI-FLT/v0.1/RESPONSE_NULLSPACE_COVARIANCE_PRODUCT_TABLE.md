# SCI-FLT-FIXED v0.1 Response, Null-Space, And Covariance Product Table

Status: sanitized Stage A author candidate awaiting exact-byte owner approval

| Product/fact | Definition | Availability/claim rule |
| --- | --- | --- |
| Exact local operator | Complete finite `J_full L_Theta` from parent rows to scientific output rows | Required for every realized product |
| Local spatial/Fourier transfer | Transfer of the exact sampled operator on the declared finite-grid domain where scientifically defined | May be unavailable; never substituted by a continuous ideal |
| Transformed parent response | `R_out = J_full L_Theta R_parent` | Available only for an exact compatible parent response/domain/basis; otherwise unavailable |
| Complete transformed covariance | `C_out = J_full L_Theta C_parent L_Theta^T J_full^T` | Only when the complete required parent covariance domain is available |
| Diagonal-input propagated covariance | Exact transformation of an explicitly diagonal independent parent covariance | Output generally has off-diagonal terms; a marginal plane is not full covariance |
| Marginal variances | Diagonal of an exact or partial transformed covariance | Unknown cross terms remain unknown; no independence claim |
| Structured/operator covariance | Exact content-bound representation of a declared covariance operator/structure | Must state domain, rank, null space, omitted terms, and operations supported |
| Partial covariance | Exact available blocks/terms with omissions named | No complete-covariance claim |
| Unavailable covariance | Typed absence | Never zero, independence, precision, or significance |
| Filter null space | Exact input modes mapped to zero on the declared row domain | Required when derivable; distinct from parent null/additive-reference state |
| Attenuated modes | Exact or bounded operator-dependent attenuation state | Not an upstream sky-to-output transfer claim by itself |
| Invariant modes | Exact modes preserved under declared normalization/domain | Constant preservation does not imply point-source or upstream-mode preservation |
| Phase state | Exact sampled phase/subpixel convention | Required for response/transfer interpretation |
| Source imprint and bias state | Exact inherited parent response/limitations composed with the local operator, plus any claim-specific bias statement or typed unavailability | No source-free, unbiased-estimator, preserved-peak, integrated-flux, or target-PSF claim follows from fixed convolution alone |
| Input-to-output influence | Exact support/coefficient relation from parent rows to output rows | Not physical exposure |
| SCI-NOI empirical uncertainty | NOI-owned attachment from exact transformed compatible ensemble | Distinct from deterministic covariance propagation and never obtained by filtering variance/weight/STD products |

Parameter, kernel, cutoff, beam, WCS, selection, and model uncertainty are
separate from conditional propagation through fixed `L_Theta`. The local
kernel is not automatically a complete source response, realized PSF, or
whole-chain transfer.
