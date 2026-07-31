# SCI-MAP-001 cross-audit handoff proposal — 2026-07-31

## Scope and immutable source

This is a handoff-only post-completion proposal for coordinator review. It
does not amend or supersede the completed SCI-MAP-001 audit, choose a repair
base, perform a repair, dispatch a target audit, or record a recipient
disposition.

- Source package: `SCI-MAP-001`.
- Exact audit decision commit and `source.audit_commit`:
  `b9e1e9a9b2fe492c402d8c7b0cf7e5a36c136a53`.
- Approved-contract and completed-audit artifact:
  `doc/audits/packages/SCI-MAP-001_SCIENTIFIC_CONTRACT_AUDIT.tex`.
- Completed-audit artifact SHA-256:
  `6c8decef93f5607bc9e8dfc84e31aee67f45fa5c695fc80563c7e7064f78d556`.
- Governing implementation assessed by the completed audit:
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Frozen cross-audit protocol reviewed at:
  `fba342020e5c241fb06320e3c929d4c4bb050a2f`.
- Related canonical guard-band record, used only by the VAL post-core
  evidence proposal:
  `doc/audits/handoffs/SCI-MAP-001/SCI-MAP-001-XAUD-001.yaml` at the frozen
  framework commit, SHA-256
  `bc4d3c7b4ecf2ac21051d00e7bebd2ba910b6920ab8a5af63638b3cec3fbb206`.

Any commit containing these five proposal files is a distinct coordination
artifact. It is not the audit decision commit, governing implementation SHA,
repair SHA, or final audit artifact identity.

## Proposed outbound records

| ID | Target | Review phase | Requested actions | Record SHA-256 |
| --- | --- | --- | --- | --- |
| `SCI-PTC-001-XAUD-001` | `SCI-PTC-001` | `pre_core_authority` | `add_or_update_dependency`; `add_interface_test` | `ba53e2c2e6af610112696f3edc87c92e43bc2530ac33b700fd5c64e76a8986fa` |
| `SCI-PTC-001-XAUD-002` | `SCI-PTC-001` | `post_core_evidence` | `add_or_update_dependency`; `add_or_update_finding`; `add_interface_test`; `restrict_consumer` | `0aed9f3a625193d882058e7602a6a785093ec2f077d2bdffd9008ac345f23f10` |
| `SCI-VAL-001-XAUD-001` | `SCI-VAL-001` | `pre_core_authority` | `add_or_update_dependency`; `add_interface_test` | `f1542e62eef09b8fa9b122178c10ec629005222c5cd93f231deae85147e309b3` |
| `SCI-VAL-001-XAUD-002` | `SCI-VAL-001` | `post_core_evidence` | `add_or_update_dependency`; `add_or_update_finding`; `add_interface_test`; `restrict_consumer` | `6680e13467dfb083e61bb0594aa3d5cc388bc7c107fb3e7de8ba3d1a457679e0` |

All four submissions have `arrival: before_dispatch` and `status: submitted`.
Every recipient disposition remains `pending`, with no action, rationale,
target branch, target commit, affected finding/dependency, or report location.

## Review-phase boundary

The two `pre_core_authority` records contain only the approved F009/F010
contract facts and exact owner-decision authority. The PTC record states the
nonprecision default and the conditions PTC must prove before precision is
allowed. The VAL record states the approved separation of eligibility,
contribution, exposure, normalization support, science-policy support, and
authoritative science validity.

The two `post_core_evidence` records remain quarantined until the recipient's
independent core is frozen. They carry MAP implementation/source findings,
missing evidence, limitations, and fail-closed restrictions without selecting
a PTC estimator, a correlation model, a detector-weight construction, an
upstream validity policy, or a target implementation conclusion.

The VAL post-core record cites `SCI-MAP-001-XAUD-001` only as evidence of a
downstream consequence. That record does not establish a raw numerical
regression, prove that the raw Boolean caused the guard-band pixel, validate a
filter or noise estimator, close a MAP finding, or authorize production.

## Machine-readable SCI-MAP-001 ledger addendum

This merge addendum contains only `outgoing_handoffs`. The coordinator must
not apply it as a replacement SCI-MAP-001 record.

```yaml
outgoing_handoffs:
  - id: SCI-PTC-001-XAUD-001
    path: >-
      doc/audits/handoffs/SCI-PTC-001/SCI-PTC-001-XAUD-001.yaml
    target_package: SCI-PTC-001
    review_phase: pre_core_authority
    arrival: before_dispatch
    status: submitted
  - id: SCI-PTC-001-XAUD-002
    path: >-
      doc/audits/handoffs/SCI-PTC-001/SCI-PTC-001-XAUD-002.yaml
    target_package: SCI-PTC-001
    review_phase: post_core_evidence
    arrival: before_dispatch
    status: submitted
  - id: SCI-VAL-001-XAUD-001
    path: >-
      doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-001.yaml
    target_package: SCI-VAL-001
    review_phase: pre_core_authority
    arrival: before_dispatch
    status: submitted
  - id: SCI-VAL-001-XAUD-002
    path: >-
      doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-002.yaml
    target_package: SCI-VAL-001
    review_phase: post_core_evidence
    arrival: before_dispatch
    status: submitted
```

## Coordinator action still required

These records and the addendum are proposals only. The coordinator must
review their content, integrate accepted records into the canonical registry,
freeze the recipient inbox manifests, and dispatch the pre-core and post-core
partitions under the protocol. No recipient action or disposition has been
simulated here.
