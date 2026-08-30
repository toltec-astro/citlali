# SCI-FLT-FIXED v0.1 Observation And Coadd Noncommutation Table

Status: sanitized Stage A author candidate awaiting exact-byte owner approval

| Method/product | Parent role | Output identity | v0.1 statement |
| --- | --- | --- | --- |
| Filtered observation | One exact complete MAP observation bundle | `SCI-FLT-FIXED` successor of that observation and one exact operator generation | Authorized conditionally on exact parent and FLT admission |
| Filtered raw coadd | One exact complete MAP centered-integer coadd bundle | `SCI-FLT-FIXED` successor of that coadd and one exact operator generation | Authorized conditionally on exact parent and FLT admission |
| Filtered JINC observation | One exact complete JINC observation bundle | `SCI-FLT-FIXED` successor of that JINC bundle and one exact operator generation | Scientifically typed; numerical route remains unavailable with its parent gates |
| Coadd of filtered observations | Multiple FLT products | Not defined by SCI-FLT-FIXED v0.1 | Requires a separately owned coadd contract |

SCI-FLT-FIXED v0.1 performs no coaddition. It assumes no identity

\[
  L_c\,\operatorname{Coadd}(m_o)
  =
  \operatorname{Coadd}(L_o m_o).
\]

Support, coefficients, response, covariance, boundaries, parent admission,
registration, normalization, and operator generation can prevent commutation.
A future proof for one bounded compatible case must bind all of those facts
and creates only that compatibility result, not a universal rule.

Observation and coadd roles remain different even when shapes, WCS values,
kernels, or numerical outputs happen to agree.
