# Input integrity and constrained-read record

Authoring run: **citlali-ptc-uniform-author-run-2026-09-05**  
Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.4**  
Recorded: **2026-09-06**

## Original released eleven-file packet

Before reading content, the manifest at /private/tmp/citlali-ptc-uniform-author-run-2026-09-05/ptc-uniform-author-input/MANIFEST.json was hashed. Its SHA-256 was:

**3140f1769ce1d4ed970c21f8cb56ff50e697880d5676983314d7b0584afd8f00**

It matched the owner-supplied digest. Recursive inventory found exactly eleven regular files: the manifest plus ten declared content files. After reading only the manifest, all declared byte counts and SHA-256 values were checked before content was opened; every check passed.

| Content file | Bytes | Verified SHA-256 |
| --- | ---: | --- |
| PRIOR_WORK.md | 3437 | 2fc533d236ab04b98783c6acd932a23f1e6caa2de0548d273436a2e44e6f9d8d |
| README.md | 2895 | a9402af0d78f93be1bd1ecf693eac582e10b16ca78b9ba5baca21c0601245303 |
| SCOPE_BRIEF.md | 6943 | d42df7ae3e6b2272eb42e420fd2b4c364db7df8cd1db96cea5eee88fd47a7481 |
| references/JINC_BOUNDARY.md | 8747 | 246f9c87d04bdbf09c878155810d089da9f7509f744847a61a83846824c9728c |
| references/MAP_BOUNDARY.md | 8917 | a02a98af1d381336b47bba10d908bd84999613fe5c60a1cc92fc6286b40606a3 |
| references/NOI_BINDING.md | 8903 | b8bf96cf6e32e82b244d1f2f0eb45683435e3725f51ea4135a14703b10c9dde4 |
| references/PROCESS_AND_SCOPE.md | 9256 | 3665e6008399a7af49b7050709fa00c66e2f3a92070da1623c9bd0dfbc41501c |
| references/PTC_COEFFICIENTS.md | 21015 | ffe62f6a9110ac7d442944afaedde90d69e0f28cabc2dc2d2159a53bb011e014 |
| references/REGISTRY_FRAMEWORK.md | 6348 | 9b9a8426b26afdce584bb30189d6dc25dc03f090b6bbebfbf07f15c0e56fef46 |
| references/VAL_RULES.md | 20276 | 579d966ad64a378bfab05b5c8f4ffa3a1400dfb319e3bd8f43a9bd29a0b03d75 |

Content was then read only in this released order: README.md; SCOPE_BRIEF.md; PRIOR_WORK.md; references/PROCESS_AND_SCOPE.md; references/PTC_COEFFICIENTS.md; references/REGISTRY_FRAMEWORK.md; references/MAP_BOUNDARY.md; references/JINC_BOUNDARY.md; references/NOI_BINDING.md; references/VAL_RULES.md.

## Scientific-owner review-1 supplement

Before reading content, the manifest at /private/tmp/citlali-ptc-uniform-owner-review1-2026-09-06/MANIFEST.json was hashed. Its SHA-256 was:

**3732fe857290645e52b6b1a37ec9e161eb46de5c89094380425e0983c29400ea**

It matched the owner-supplied digest. Inventory found exactly four regular files: the manifest plus three declared content files. After reading only the manifest, all three declared byte counts and SHA-256 values were checked; every check passed.

| Content file | Bytes | Verified SHA-256 |
| --- | ---: | --- |
| README.md | 2798 | b54e80d86b270ae4c558e80a01bf68679e08b595091f44972f39cd0cb62e62a9 |
| OWNER_REVIEW_1_VERBATIM.md | 18992 | 1e9074bfa86566a790f7d1a0284817aced321c66222c6899ef0e13a229a223e9 |
| PTC_RANK_AND_PUBLICATION_CLAUSES.md | 3586 | 9e16b036203e0fbb6d4899220864476be4a6cc3f412df6e2157f693fc066abea |

Content was then read only in this order: README.md; OWNER_REVIEW_1_VERBATIM.md; PTC_RANK_AND_PUBLICATION_CLAUSES.md. The clause file admits only exact frozen SCI-PTC v0.1/r0.5 requirements 072, 090, 094, and 095; omitted parent sources were not retrieved.

## Firewall record

No Git checkout, repository instruction, implementation, test, audit, repair, manager or independent-review report, validation result, other task, external scientific source, omitted parent source, or surrounding provenance source was inspected for scientific content. No input was modified. Source names in the packets were treated as provenance, not retrieval permission. Local Tectonic and PDF/checking utilities supplied no scientific content. No network or dependency installation was used.

The original manifest's historical release-pending labels were superseded only by the explicit owner release instruction. Owner-review-1 was treated as permitted substantive owner feedback. Support in principle was not treated as complete adoption.

## Revision provenance and preservation

The sealed r0.1 manifest remains SHA-256 c15bf5e238dd3776a4cf83a506c6d1c7e3b8172761b1f621ca28cf42e9574124 and its 42 listed artifacts passed on 2026-09-06. The sealed r0.2 manifest remains SHA-256 9c05f8d5d14c338ee4f58d81101f7e23d2bc361401012076b8c47a366e063064; its 42 listed artifacts plus the manifest (43 final/source files total) passed verification before revision.

Exactly those 43 r0.2 final/source artifact files were copied into sibling output-r0.3; r0.2 build scratch was excluded. The four owner-review supplement files were copied byte-exact under owner-review-1-input, and OWNER_REVIEW_1_RESPONSE.md was added as the new response artifact. All r0.3 edits and build scratch were confined to output-r0.3.

Before this correction, the sealed r0.3 manifest was verified at SHA-256 87eafdbaed29dc8d810a5bba9d4ba818d57a837c04337a4c71c31b06215f0ee9 and all 47 listed artifacts passed. Its complete 48-file final/source inventory, including the manifest and excluding build scratch, was copied byte-exact into sibling output-r0.4. Revision r0.4 uses no additional scientific input and changes only the structural-binding versus bound-byte-integrity wording plus revision-bearing records and fresh render/sealing artifacts. The sealed output, output-r0.2, and output-r0.3 trees and both isolated input directories remain unchanged.
