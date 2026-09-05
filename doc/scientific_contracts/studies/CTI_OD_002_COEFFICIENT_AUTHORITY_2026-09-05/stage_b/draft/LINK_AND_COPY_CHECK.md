# Link and immutable-copy check

Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.2**  
Draft check date: **2026-09-05**  
Result: **PASS**

- All 24 relative Markdown links in `README.md`, `PRIOR_WORK.md`, and
  `SCOPE_BRIEF.md` resolve locally.
- Root `PRIOR_WORK.md`, `SCOPE_BRIEF.md`, and all seven files under
  `references/` byte-match the released inputs.
- All eleven files under `inputs/` byte-match the released packet.
- `inputs/MANIFEST.json` retains SHA-256
  `3140f1769ce1d4ed970c21f8cb56ff50e697880d5676983314d7b0584afd8f00`.

The r0.2 package checker used `/Users/gwilson/tolteca/bin/python -B` and is
retained as non-authoritative scratch at `build/check_package.py`, with its
PASS record at `build/package_check.txt`. It also confirmed the 25/25
requirement rows, 16/16 prediction rows, both PDF page/header checks, and the
sealed r0.1 manifest hash.
