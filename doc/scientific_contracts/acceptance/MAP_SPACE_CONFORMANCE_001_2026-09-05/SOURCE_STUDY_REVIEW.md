# Independent exact-commit review — MAP-space conformance successor

Date: 2026-09-04
Reviewer: fresh-context independent subagent `/root/map_space_successor_exact_sha_review`.
Preserved by the manager after the reviewer completed; the reviewed candidate was not edited.

The reviewer’s complete final report follows.

---

**ACCEPTABLE FOR OWNER REVIEW.** The two major findings and minor citation omission are adequately repaired at the exact successor below. I found no new actionable findings. This verdict assesses the documentation gap study; scientific-owner acceptance and integration remain pending.

| Reviewed identity | Exact value |
| --- | --- |
| Candidate | `f36cb788dc1ce99dad9c30bcd4442dc8bfbf681e` |
| Candidate tree | `c12a3cb915cd900331661fc1f9e3fdf2899a197c` |
| Sole parent | `402b82bc7c38d8a3739d7803f46ccf3f1bbd90f8` |
| Parent tree | `163b8136066cf56d320cfb24488350118540510f` |
| Inspected application source | `9f42d348298d76c5d5145aaf0c3eace1f3e154c1` |
| Application-source tree | `e51f22760c64454ce7233c45dd740aa710777bae` |
| Preserved original study | `93c2b4591bb5d0cf8efe4491975c31e5f8fb5903` |
| Original-study tree | `e0b51383cdeb4ad318d3548b05ad803dd9ef1cf4` |
| Observed canonical | `ae953ed4d87d1f693d2bbf42aebbc25ef730c771` |
| Canonical tree | `37ee17cf001ceb2c193fbea5e2b5ae3d147ba4a1` |

The reviewed branch is `codex/map-space-conformance-001-review-repair-2026-09-04`, in [/private/tmp/citlali-map-space-conformance-001-review-repair-2026-09-04](/private/tmp/citlali-map-space-conformance-001-review-repair-2026-09-04). Initial and final worktree checks were clean. Canonical and the original study are siblings; the reviewed successor and canonical share the exact inspected source base. Neither lane was moved.

I read repository AGENTS.md, engineering governance, review/conformance governance, the Scientific Contract Library Program, relevant architecture and sequencing records, all thirteen packet artifacts, the complete six-file successor diff, the predecessor review, and relevant frozen authority and admitted implementation. This is a Tier 2 read-only exact-commit review, with no Stage B authorship or new scientific derivation. Timestream Successor governance is not triggered by this documentation-only map-space successor.

Governance is effective through the canonical integration ledger’s acceptance record for `06a3ade51c1b3f38887295433d913811bf25cd14`. I independently confirmed that commit is on canonical ancestry and that the accepted, canonical, and reviewed copies match:

| Governing document | Verified SHA-256 |
| --- | --- |
| ENGINEERING_GOVERNANCE.md | `70769787ce2ef4b7323cd2a38e221ade4af3310e0ad6b7b682e08cb4e4d61e76` |
| REVIEW_AND_CONFORMANCE.md | `691e6d6250102ef2f4a504397581ee67c5707d898ab20fb8dd9e874c47f99bb1` |

The stale “candidate” labels therefore do not negate the recorded effectiveness. Frozen MAP/JINC authority was likewise resolved through controlling freeze records and the accepted shared-conventions repair, preserving the acknowledged frozen editorial residue.

The prior findings have these dispositions:

1. **Coordinate conformance overstatement — adequately repaired.**
   [Product traceability, line 16](/private/tmp/citlali-map-space-conformance-001-review-repair-2026-09-04/doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/PRODUCT_AND_BOUNDARY_IMPLEMENTATION_TRACEABILITY.md:16) and [route classifications, lines 14 and 18](/private/tmp/citlali-map-space-conformance-001-review-repair-2026-09-04/doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/ROUTE_AVAILABILITY_CLASSIFICATION.md:14) now classify MSP-P002, MSP-E002, and MSP-E006 as `IMPLEMENTED_LEGACY_SEMANTICS`.

   Independent source inspection confirms the distinction. The builder checks the committed operation at [timestream_native_science_projection.h:314](/private/tmp/citlali-map-space-conformance-001-review-repair-2026-09-04/include/citlali/core/pipeline/timestream_native_science_projection.h:314) and joins values and pointing through native identities at line 439. The later [consumer guard at line 211](/private/tmp/citlali-map-space-conformance-001-review-repair-2026-09-04/include/citlali/core/pipeline/timestream_native_science_projection.h:211) compares axes/grouping, dimensions, detector inventory, flags, and sample values. It does not consult stored operation/scope or compare incoming occurrence/application generation and exact target-WCS identity. MAP and JINC subsequently consume stored pointing while constructing pixel positions from their map buffers.

   That does not establish the exact ancestry required by [SCI-MAP-REQ-003/005](/private/tmp/citlali-map-space-conformance-001-review-repair-2026-09-04/doc/scientific_contracts/packages/SCI-MAP/v0.1/src/common/requirements.tex:5), the [PTC-to-MAP coordinate join](/private/tmp/citlali-map-space-conformance-001-review-repair-2026-09-04/doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-PTC_TO_SCI-MAP_BOUNDARY.md:41), and [SCI-JINC-REQ-002](/private/tmp/citlali-map-space-conformance-001-review-repair-2026-09-04/doc/scientific_contracts/packages/SCI-JINC/v0.1/src/common/requirements.tex:23).

   The successor preserves useful construction-local evidence and correctly labels the equal-payload/foreign-occurrence counterexample as source-derived, unexecuted, and not an observation about an actual reduction. Its dependent classifications and summaries are consistent.

2. **Unsupported exclusive OOF parent policy — adequately repaired.**
   [OOF_ATTACHMENT_ENVELOPE.md:12](/private/tmp/citlali-map-space-conformance-001-review-repair-2026-09-04/doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/OOF_ATTACHMENT_ENVELOPE.md:12) now makes the examples explicitly non-exhaustive, admits none, and excludes no other proposal, including JINC. Eventual selection remains with separately reviewed OOF scientific authority and exact upstream bindings. This agrees with the recovery-first OOF sequencing record. No graph ID, parent admission, numerical route, or OOF launch was added.

3. **Unmanifested optional MSP-P009 test citation — adequately repaired.**
   [Product traceability, line 23](/private/tmp/citlali-map-space-conformance-001-review-repair-2026-09-04/doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/PRODUCT_AND_BOUNDARY_IMPLEMENTATION_TRACEABILITY.md:23) removes exactly the optional test citation and changes grade `A/C` to `A`; its classification and remaining finding are unchanged. The admitted Wiener/filter-policy source independently supports the retained finding: mutable signal/kernel/weight handling, legacy mode/template selection, and missing frozen parent/template/method bindings. The frozen MATCHED requirements explicitly require those immutable identities and distinguish its amplitude estimator from legacy Wiener/convolution behavior. No executed validation is inferred.

I also corroborated the principal preserved gaps: processed-sample exposure accumulation, pixel-coefficient-weighted ordinary coaddition, JINC’s additional numerical roles and zero substitution, mutable filtering and reciprocal-weight variance treatment, expressly unestablished signal/NOI operator parity, empirical NOI coefficient mutation, and missing POINT named-use/profile bindings. These support the bounded zero-complete-route conclusion. They do not establish that operational reductions are scientifically unusable.

Verification reproduced successfully using the local TolTECA Python environment:

```sh
GIT_OPTIONAL_LOCKS=0 $HOME/tolteca/bin/python -B doc/scientific_contracts/audits/MAP_SPACE_CONTRACT_TO_IMPLEMENTATION_CONFORMANCE_001/verify_packet.py
```

The verifier passed in post-commit mode on the exact candidate/tree. Independent checks additionally established:

- All thirteen artifact digests match the exact candidate and supplied subject manifest.
- Exactly six paths differ from the predecessor; the other seven packet files are byte-identical.
- Every non-packet path is unchanged from the inspected application base.
- All 48 admitted-source bindings match both exact-base objects and current files.
- All 67 frozen horizontal bindings, SRC-002–SRC-068, match; repaired shared conventions match CTI-S001 separately.
- All 17 products, 32 edges, and 16 trace IDs remain present.
- Product-row differences are exactly MSP-P002 and MSP-P009; edge-row differences are exactly MSP-E002 and MSP-E006.
- Both predecessor-to-candidate and source-base-to-candidate `git diff --check` gates pass.
- Canonical, original-study, and predecessor branch tips remain exact.

Independent counts agree with both summary views:

| State | Products | Edges |
| --- | ---: | ---: |
| Source-level conformant | 0 | 0 |
| Legacy semantics | 8 | 6 |
| Missing authority | 1 | 10 |
| Missing implementation | 2 | 1 |
| Contradictory | 5 | 9 |
| Unavailable by design | 0 | 5 |
| Not applicable | 1 | 1 |
| Total | 17 | 32 |

I inspected the recorded eleven manager corruption probes and independently ran five additional in-memory checks. The verifier rejected restored conformance for each of the three coordinate rows, restoration of the removed MSP-P009 citation, and an unrelated MSP-E027 evidence change. These checks changed no files and establish mechanical safeguards only.

The six changed artifact digests are:

| Artifact | SHA-256 |
| --- | --- |
| FINAL_REPORT.md | `06230500bc13308f9a18dfc12668ad74f579df7aaa8d882a93616481d9e907a6` |
| OOF_ATTACHMENT_ENVELOPE.md | `9c9bc6df83d25c5c2390efc9d1ea0868fad75bad0282d0b4ddefb9e8e3956343` |
| PRODUCT_AND_BOUNDARY_IMPLEMENTATION_TRACEABILITY.md | `cccf47eb89b967b39e3b511646ebed20a402d0730c19388c32bb6a90783029df` |
| REVIEW_REPAIR_RECORD_2026-09-04.md | `748d6914de0ef2eb798253bb05e12832c46f7f56049e61f3b56558de00d77a3b` |
| ROUTE_AVAILABILITY_CLASSIFICATION.md | `5270e5eeb69a14cf60400abe35e79158a4bd159ea9aa8d386b39741de3a8a4ab` |
| verify_packet.py | `91492953da3acd3f068ed39511139a5c30ef1ef354ffc2d7c279a1bbb70c2483` |

| Review axis | Disposition |
| --- | --- |
| Scientific/behavioral | **PASS WITH RECORDED LIMITATIONS:** repaired documentation claims are supported; frozen science and numerical behavior are unchanged. Application conformance remains unestablished. |
| Architecture/ownership | **PASS:** no application architecture, lifecycle, interface, dependency, or Engine growth; OOF admission ownership and the implementation/authoring firewall are preserved. |
| Repository/evidence | **PASS:** exact identity, ancestry, scope, digests, preservation, and clean-state gates reproduced. No remaining citation-completeness finding identified. |

No files, commits, branches, worktrees, or refs were changed by this reviewer. No concurrent FRUIT/ALIGN uncommitted content was inspected. No build, CTest campaign, numerical trace, reduction, performance measurement, Unity access, network operation, installation, merge, or push was performed.

CTI-OD-001–006 remain open; CTI-OD-007 remains inherited/closed. Owner acceptance of this exact successor, integration, implementation mapping, FRUIT/OOF attachment, activation, and production remain separate decisions.
