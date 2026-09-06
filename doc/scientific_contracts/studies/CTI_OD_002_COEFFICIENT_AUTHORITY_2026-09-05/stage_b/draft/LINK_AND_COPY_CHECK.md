# Link and immutable-copy check

Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.3**  
Check date: **2026-09-06**  
Result: **PASS**

- All 30 relative Markdown links in README.md, PRIOR_WORK.md, SCOPE_BRIEF.md, and OWNER_REVIEW_1_RESPONSE.md resolve locally.
- README.md's first substantive section links the admitted program/roadmap cover, Scope Brief, prior-work record, every predecessor reference, owner-review supplement, and response.
- Root PRIOR_WORK.md, SCOPE_BRIEF.md, and all seven files under references byte-match the original released input.
- All eleven files under inputs byte-match the original isolated packet. inputs/MANIFEST.json retains SHA-256 3140f1769ce1d4ed970c21f8cb56ff50e697880d5676983314d7b0584afd8f00.
- All four files under owner-review-1-input byte-match the isolated scientific-owner review-1 supplement. Its MANIFEST.json retains SHA-256 3732fe857290645e52b6b1a37ec9e161eb46de5c89094380425e0983c29400ea.
- The sealed predecessor manifests retain SHA-256 c15bf5e238dd3776a4cf83a506c6d1c7e3b8172761b1f621ca28cf42e9574124 for r0.1 and 9c05f8d5d14c338ee4f58d81101f7e23d2bc361401012076b8c47a366e063064 for r0.2; both manifest checks pass.

The checker uses /Users/gwilson/tolteca/bin/python -B and is retained only as build/check_links_and_copies.py scratch. The final package checker and PASS output are likewise retained under build and omitted from the artifact manifest.
