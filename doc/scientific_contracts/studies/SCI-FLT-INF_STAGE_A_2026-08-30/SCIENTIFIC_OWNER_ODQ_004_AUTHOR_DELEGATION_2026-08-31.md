# SCI-FLT-INF-ODQ-004 scientific-owner author delegation

Decision identity: `SCI-FLT-INF-ODQ-004`

Date: `2026-08-31`

Scientific owner: Grant Wilson

Status: author-delegated development; no noise/covariance option selected

## Owner direction

The exact noise/covariance model, spectral weighting, and parent-coefficient
role require scientific development. A future implementation-blind contract
author shall produce bounded, scientifically coherent options in both the
**Scientific Rationale and Contract** and the **Engineering Conformance
Specification**. The two views must use the same stable option identities and
must agree on the scientific meaning, assumptions, consequences, unavailable
states, and decision still required.

This delegation dispositions ODQ-004 for Stage A sequencing but does not close
the scientific choice. The scientific owner must select or otherwise dispose
of an option after reviewing the authored rationales and before scientific
freeze or any numerical route is authorized.

## Historical candidate to recover

Citlali has historically used a **radially symmetrized average map noise PSD**.
That fact shall be supplied to the author only as an owner-provided historical
candidate requiring scientific examination. It is not a selected default,
exact covariance authority, proof of stationarity or isotropy, proof of
optimality, implementation-conformity claim, or permission to reproduce
historical mechanics.

The author must state what `average`, `map noise PSD`, and radial
symmetrization mean scientifically for each candidate that uses those ideas,
including their population/domain, ordering, Fourier/WCS and unit conventions,
normalization, support/window/edge treatment, missing/nonfinite policy,
rank/null space, approximation status, and dependence on learned state.

## Required option development

Without receiving implementation or historical numerical mechanics, the
author shall develop the smallest bounded option set adequate to decide:

- what exact object supplies noise weighting for an ordinary-MAP observation
  parent and for an ordinary-MAP coadd parent;
- whether the two parent roles use the same parameterized model or distinct
  scientifically justified models;
- which assumptions and approximation statements make each option admissible;
- what optimality and matching-amplitude unbiasedness claims remain valid;
- how normalization, units, response, support, validity, uncertainty, and NOI
  realization handling depend on the selected option;
- whether any option supplies covariance, only a spectral weighting model, or
  an explicitly weaker role;
- whether and how a parent coefficient field participates, without inferring
  precision or covariance from its name, sign, unit, or historical use; and
- which inputs, states, failures, diagnostics, and products are required to
  realize and distinguish each option.

The authored options must include a typed unavailable outcome wherever the
required scientific object or assumptions are absent. They must not silently
select an option, invent a production default, or collapse observation-local
and coadd-local parent identities.

## View-specific requirements

The scientific view must explain the physical/statistical model, estimand
compatibility, assumptions, approximation and bias/variance consequences,
limiting cases, uncertainty meaning, and falsifiable predictions for every
option.

The engineering view must translate those same options into traceable
operator/state/product requirements, exact input identities, units and shape,
requested/effective/resolved/realized provenance, support/null/invalid/failure
states, and validation obligations. It may not introduce an option or
scientific consequence absent from the scientific view.

## Consequences

- `SCI-FLT-INF-ODQ-004` is **author-delegated**, not scientifically selected.
- The historical radially symmetrized average map-noise PSD is an admitted
  candidate for analysis only.
- Parent coefficient precision/covariance meaning remains unavailable.
- Observation-map and coadd-map options remain separately accountable under
  ODQ-003.
- ODQ-005 is the next owner gate for template/kernel identity; it may proceed
  while the ODQ-004 author assignment is preserved for the future author
  packet.

## Nonclaims

This delegation does not approve a PSD estimator, averaging population,
radialization rule, stationarity, isotropy, covariance, precision,
normalization, operator, approximation, units, response, uncertainty, NOI
lifecycle, product, author packet, Stage B launch, implementation conformity,
validation, performance, readiness, production, freeze, or Unity action. It
changes no SCI-FLT-FIXED or frozen SCI-NOI byte.
