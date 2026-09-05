# Link and immutable-copy check

Draft check date: **2026-09-05**  
Result: **PASS**

- All 24 relative Markdown links in `README.md`, `PRIOR_WORK.md`, and
  `SCOPE_BRIEF.md` resolve locally.
- Root `PRIOR_WORK.md`, `SCOPE_BRIEF.md`, and all seven files under
  `references/` byte-match the released inputs.
- All eleven files under `inputs/` byte-match the released packet.
- `inputs/MANIFEST.json` retains SHA-256
  `3140f1769ce1d4ed970c21f8cb56ff50e697880d5676983314d7b0584afd8f00`.

The checker used `/Users/gwilson/tolteca/bin/python -B` and is retained as
non-authoritative scratch at `build/check_markdown_links.py`.

