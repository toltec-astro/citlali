# Q02: A and B approved; weighting evidence requested

Date: 2026-09-09. Scientific owner: Grant Wilson. Review r0.2.
Status: A and B approved for controlled amendment preparation; C remains open.
The weighting test design below is proposed, with no numerical work authorized.

## Program adherence and prior-work recovery

This bounded continuation follows the [charter](../../r0.4/inputs/program/README.md),
[pilot workflow](../../r0.4/inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md),
[roadmap](../../r0.4/inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md) and
[r0.4 recovery](../../r0.4/PRIOR_WORK.md). It adopts the
[Q01 freeze](../../r0.4/SCIENTIFIC_OWNER_FREEZE_R0.4.md) and records the owner's
disposition of the exact [Q02 r0.1 decision sheet](../r0.1/README.md).
It cites frozen PTC coefficient restrictions and MAP arithmetic for the
[weighting test proposal](WEIGHTING_TEST_PLAN.md). Empirical outcomes remain
deferred. Historical implementation remains manager context only; the
independent-author reference inventory is unchanged. No new contract package
or author dispatch is commissioned. Exact sources and delivery hashes are in
[the review manifest](REVIEW_MANIFEST.json).

## Owner decision and scope

The owner said:

> I approve A and B. I am concerned about C because of variation in the detector quality and noise. We should construct a set of tests to show that uniform weighting is acceptable or not.

This disposition continues `SCI-FRUIT-OM-BOUNDARY-002` and its existing Q02-A,
Q02-B and Q02-C identities. The approved object is r0.1's scientific substance
for controlled amendment preparation, including A's S1 clarification. It is
not adoption of unspecified successor sources or permission to run a method.

| Decision | Current disposition | Exact scope |
| --- | --- | --- |
| Q02-A | **Approved scientific substance** | B01–B07 with S1: one configured-rank PTC Learn–resolve–Apply pass on the exact current FRUIT residual, preserving ordinary CAL permissions and remaining exclusions. New immutable residual-route profile versions are required where domain/source changes. |
| Q02-B | **Approved scientific substance** | B08 and M04: matching containing-pixel model paths and scale-one/no-added-offset correction/rejoin, conditional on exact quantity, reference, coordinate, occurrence and support compatibility. No unit sky-response or absolute-sky claim. |
| Q02-C | **Open, pending weighting evidence** | B09–B11's uniform occurrence family, fresh QC/retained use and explicitly identified rejoined MAP input remain unadopted. The concern does not prove uniform weighting unsuitable. No separate approval of C's MAP handoff is inferred. |

S1 preserves the ordinary-route @1 records. Residual basis fitting, loading
fitting where used, application, output retention and requested conditional
companions require applicable versioned records; downstream QC/MAP must bind
the exact current residual-route decisions. A's approval settles this rule;
its application to C's eventual records does not settle C's weighting choice.
The [existing registry](../../../../../../../SCI-VAL/v0.1/PROFILE_REGISTRY.md)
is unchanged. B12's actual mode/WCS registration remains open.

## Proposed evidence for C

Uniform weighting needs a measured acceptability case within the intended
detector/noise regime. Ordinary MAP permits nonuniform approved coefficients;
equal weighting is not inherent in the mapmaker. The proposed screen compares
uniform occurrence weights with one fixed, causally estimated noise-aware
family. It separates three questions:

1. Do elementary noise cases expose a loss, and does the estimator recover the
   expected mapping behavior?
2. With identical PTC outputs, how much do weights change noise, recovery,
   morphology, detector influence and usable support?
3. If further testing is warranted, do those differences change feedback
   recovery, leakage or stability when each arm relearns PCA on its residual?

The [test plan](WEIGHTING_TEST_PLAN.md) gives the candidate coefficient law,
small case matrix, comparison metrics, stopping points and required execution
bindings. It proposes no selector improvement or additional PCA method. A
known-variance benchmark is synthetic diagnosis only. Truth and held-out
outcomes never choose deployed coefficients. A recommendation requires
predeclared acceptable losses and independent-pointing replication for a
pointing policy. Inconclusive evidence keeps C open.

## Adoption and next owner decision

A and B do not need another scientific-substance vote. Their exact successor
clauses, immutable profile versions and source bindings can be prepared under
this disposition and must receive controlled review/adoption before use.
The complete amendment set must also resolve C and actual mode applicability.
A bounded experimental permission for both weight families, if approved,
would permit their comparison without selecting either as a policy.

The next significant decision is approval of the concrete first weighting
screen and its missing numerical bindings: exact admitted data/test populations,
coefficient/QC and evidence records, cases, support/settings, acceptance
margins, resources and execution scope. These stay under the existing Q02–Q06
identities; the proposed weighting law is not adopted here. The later feedback
screen is a separate continuation decision after the mapping result.

`FRUIT-FEEDBACK-METHOD = unavailable_pending_separate_owner_approval`.
The route remains `unavailable_under_current_frozen_parent_permissions`.
Q01 r0.4, the B01–B12 annex, generic core, upstream sources, earlier review
packets/archives, historical control and all reduction products are unchanged.
No implementation, numerical test, replay, injection, qualification, production
change, Unity action or push occurred. No source or registry amendment is
adopted by this record.
