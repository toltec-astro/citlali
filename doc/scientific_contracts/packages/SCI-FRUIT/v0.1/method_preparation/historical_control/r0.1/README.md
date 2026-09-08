# SCI-FRUIT — historical-control method scope r0.1

Status: prepared for owner review; numerical method unavailable.
Scientific owner: Grant Wilson. Date: 2026-09-08.
Proposal identity: `SCI-FRUIT-HISTORICAL-CONTROL-METHOD-SCOPE-R0.1`.

## Program adherence and prior-work recovery

This preparation follows the [program charter](inputs/program/README.md),
[pilot review](inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md),
[downstream roadmap](inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md),
and [frozen FRUIT core](inputs/authority/fruit/SCI_FRUIT_STAGE_B_R0.4_OWNER_CONDITIONAL_FREEZE_2026-09-08.md).
The completed manager [prior-work recovery](PRIOR_WORK.md) adopts frozen science,
abstracts historical behavior into questions, and defers empirical evidence.
New work is the definition of one method and its exact upstream permissions;
the generic core and PTC cleaning mathematics need no fresh derivation.
The [proposed author-reference control](PROPOSED_AUTHOR_REFERENCES.md) names the
only proposed scientific inputs. Code, configs, the internal dossier, this
manager review archive and historical evidence remain outside future authorship.
Manager review is complete; owner scope/reference approval is pending.

## Proposed first use

Define one candidate for compact-source pointing: one observation, separate
TolTEC array maps, raw normalized JINC output, with no coaddition or post-map
filter in the feedback path. Its purpose is a precisely described feedback
transformation that a later study could assess for source amplitude, centroid
and morphology. It does not define a pointing fit or establish useful recovery.
No observation, scientific target value, mode count, threshold, cap or numerical
parameter is selected by this proposal.

Use historical Citlali as the compatibility candidate and mandatory later
empirical control. Keep its exact source/configuration/executable record
separate from any scientifically approved successor. A future decision to
change historical behavior cannot redefine the historical comparison baseline.

The historical algorithm carries a complete preceding map, reruns the original
observation, removes a selected projection before residual processing, rejoins
the model, and makes a new complete map. It also carries learned masks and
weight-related state. It is more than repeated PCA cleaning, but this recovery
establishes no benefit over another procedure.

## What the owner reviews

1. [Sanitized eleven-section Scope Brief](author/SCOPE_BRIEF.md).
2. [All twenty-one method-record classes](METHOD_RECORD_PROPOSAL.md), separating
   recovered facts, proposed choices and missing authority.
3. [Upstream permissions and missing inputs](UPSTREAM_PERMISSIONS.md).
4. [Five grouped owner decisions](SCIENTIFIC_OWNER_DECISION_LEDGER.md).
5. [Internal source recovery](INTERNAL_DOSSIER.md), for manager/owner use only.

My recommendation is to approve the bounded scope and authorize a paper-only
closure of the named method and upstream questions. Do not approve a numerical
method from this packet: the actual operators, support rules, state schedule,
resources and executable/input realization are not yet fully bound. If a required
handoff cannot be made scientifically coherent within this scope, return an
unavailable-route disposition instead of adding a cleaner or a new estimator.

The next decision is scope and dependency-definition authority. Complete exact
method approval, author dispatch, implementation assessment, experiments and
qualification remain separate later decisions.

## Review artifacts and limits

[Manifest](PACKET_MANIFEST.md), [verification](VERIFICATION.md), and
[source identities](SOURCE_IDENTITIES.json) bind this version.
`SCI-FRUIT-historical-control-method-scope-r0.1-owner-review.tar.gz` contains the
manager proposal and its copied sources. It is explicitly **not an author packet**.
Links inside copied sources retain their original repository context and are
provenance only; they do not admit unlisted references. The external frozen
Stage B archive remains at its original path and is identified, not repacked.

All earlier reductions, empirical packets and the two untracked review archives
remain preserved. The UID 4460 diagnosis is closed; EL-F14 remains unapproved
and unrun. No numerical work, code/configuration change, Unity activity or push
was performed to prepare this proposal.
