# SCI-MAP_TO_SCI-FLT-MATCHED v0.1/r0.1

Status: exact boundary draft; no realized input, implementation, conformity,
validation, or production claim

Producer: `SCI-MAP`

Consumer: `SCI-FLT-MATCHED`

## Admitted parent role

The producer supplies exactly one immutable normalized ordinary-MAP observation
bundle or one immutable normalized ordinary-MAP coadd bundle. The boundary
shall carry parent identity, class, generation, grouping, signal quantity and
units, WCS/frame/grid/shape/index/order/pixel-center convention, valid support,
missing/nonfinite policy, response identity, covariance availability and
population/conditioning identity, beam and CAL lineage, lifecycle, provenance,
and exact coadd membership when applicable.

`SCI-FLT-MATCHED` consumes but does not rewrite these facts. It derives one
base-v0.1 output anchor at each exact parent pixel center. Observation and coadd
parents remain separate; no commutation, state sharing, or FLT coaddition
follows.

Missing ordinary-MAP authority, invalid lifecycle, unavailable support, or an
unresolved required fact makes only the dependent FLT route unavailable. An
AO choice cannot manufacture or repair a MAP parent. JINC and other derived
parents are rejected by this boundary.
