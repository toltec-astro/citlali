# SCI-FRUIT v0.1 / Stage B r0.1 — exact artifact manifest

Status: owner-review draft; not frozen. Numerical methods/routes: unavailable_pending_separate_owner_approval. Scientific read-set identity is bound in SOURCE_IDENTITIES.md. This manifest adds no scientific authority.

The rows bind every regular delivery payload. This manifest does not hash itself; ARTIFACT_MANIFEST.sha256 binds its exact bytes. The archive contains these payloads and this manifest/sidecar exactly once, without symlinks or extra paths. Archive/digest are external delivery controls, not recursively embedded. Caches, build intermediates and rendered PNGs are excluded.

| File | Role | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `.gitignore` | Document navigation, decision, identity or verification record | 48 | `e05caa7d1145c97b5a78dfe50e1bf5b37321064d7574f416ea1e97186da6f4da` |
| `DECISION_LOG.md` | Document navigation, decision, identity or verification record | 3000 | `4ff006430ccc333f9ae3ad19c730f7d01b532629c3bc94731a2d2f9c9f3517b2` |
| `DOCUMENT_VERIFICATION.md` | Document navigation, decision, identity or verification record | 4621 | `1d3feb16847564026e02243e24128cf0acb8fc748efefbaeb4757d8fcb6c57b7` |
| `OWNER_LEDGER.md` | Document navigation, decision, identity or verification record | 6099 | `fa88ee2ef8bc981284b27d85f7685707127e25f9f2bf86e6b8678370f01c3b60` |
| `README.md` | Document navigation, decision, identity or verification record | 3729 | `cf8cb784963d522526afd5ab52dd4d1238c54bf3e481d8e9273d88ee1cf7a1a8` |
| `REQUIREMENT_CROSSWALK.md` | Document navigation, decision, identity or verification record | 7086 | `544a4997aeb87037665f2402f53ba802dea6865d6f891d5deb64cbd8fcc2d98c` |
| `SOURCE_IDENTITIES.md` | Document navigation, decision, identity or verification record | 3555 | `62234297084d11e5c7b328cafd495a90fc41f464c9f376833a155a6c23117213` |
| `pdf/engineering.pdf` | Compiled owner-review view | 160386 | `d66e3f96ebcd8aaaf1c8958e828fd6d8c1489b3fbe1669f21a4702f0fe01b671` |
| `pdf/scientist.pdf` | Compiled owner-review view | 160449 | `5a511ce26952bf1c81453f282406f5a1aa99a7208ce95ad52f509d1c857c6708` |
| `qa/DOCUMENT_CHECKS.json` | Document verification tool or evidence only | 742 | `92b9a438cb94f3faf488335083c2ceb85f681f973ef47da94e51e25ed228ea79` |
| `qa/PAGE_INSPECTION.json` | Document verification tool or evidence only | 1391 | `01abc34fe8bd4996ff7aef1a61226c77e89a2e1c73e5bd8bbe19afed71452b5e` |
| `qa/build_documents.py` | Document verification tool or evidence only | 2447 | `8c3bc63e83854db5d58a3874959e11d71b7142c95fc141711c2122d5bb538f3b` |
| `qa/package_documents.py` | Document verification tool or evidence only | 3399 | `e65e6a2d60ea596a9a4525a7b996de259dae9dda39d0f0073c2b2b09a39077e9` |
| `qa/verify_documents.py` | Document verification tool or evidence only | 6918 | `8a796de8b53e46ebe7eac67538ab29849197d5a047b75334916517bf5d26adcc` |
| `review/REVIEW_REPORT.md` | Bounded independent consistency review | 1980 | `b0d714f071bf00485b5e1427d81607106ab907bd4e47e1abf4c01e878c87e017` |
| `review/ROUND_1.md` | Bounded independent consistency review | 15328 | `2ceee86dcd965bebf1c394fe0de81db1022baa9d19560c196fccf325774b8d27` |
| `review/ROUND_2.md` | Bounded independent consistency review | 8811 | `7a4bf3da39155e082f667b3671f1af37857f0a07c3486aab5ec5eaf3b2ff1663` |
| `src/PTC_APPLICATION_REFERENCE.tex` | Exact permitted frozen quotation under cover | 1965 | `75116261a33ec1adbb092d38da43a590624a08428e51cac32ee0c4a34f216a59` |
| `src/common/assumptions.tex` | Canonical conditional scientific authority | 6207 | `fd6a5b2b1e4600572a817296bdcc2b4e78d041521b6071a0bacb5818c0489e48` |
| `src/common/definitions.tex` | Canonical conditional scientific authority | 20919 | `855125eed78291260fd8fa8d404ff0c80f8e0c46a389ee44c4097ec1d5545db9` |
| `src/common/edge_cases.tex` | Canonical conditional scientific authority | 11025 | `096ce88b977787484774ab7238746dcbd93ab2e710bf78410cb5347f8f96d2b0` |
| `src/common/equations.tex` | Canonical conditional scientific authority | 13617 | `493f9112011efc1350766817ab1266deefc65a2ff703e480c1e13c1fb89082f8` |
| `src/common/notation.tex` | Canonical conditional scientific authority | 6002 | `f92171e25d98e94250dfeafbabe62dcefc4fbbe0d22210f7488c78cecc42a636` |
| `src/common/requirements.tex` | Canonical conditional scientific authority | 19300 | `6e1b207987045f5668080f40f1641e6396e60f13d62e7a87a7b3764599b28fd1` |
| `src/engineering.tex` | View wrapper or shared typesetting | 211 | `2c7eaaaf742a7021424993fe3692d4869d9dce65b87173a37d96a6642dbb63f5` |
| `src/preamble.tex` | View wrapper or shared typesetting | 1488 | `8aeed88f89932684454be12ac064ec1c86fca4021c0ef356d0408e787c9cab86` |
| `src/scientist.tex` | View wrapper or shared typesetting | 208 | `3d4b8f777936fdc77bdcbf9f41b8043788f4f30327d636e7dda84ea48f06ccef` |
