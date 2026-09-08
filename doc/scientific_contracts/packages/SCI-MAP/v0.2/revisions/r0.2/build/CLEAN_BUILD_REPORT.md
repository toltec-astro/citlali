# Clean document build report — SCI-MAP v0.2/r0.2

Build identity: `SCI-MAP-R02-DOCUMENT-BUILD/candidate-build-05`. Status: **PASS for document compilation and artifact checks**. Scientific-owner acceptance, freeze, numerical-route activation and implementation conformance are separate and are not granted by this report.

All three views compiled in new isolated output directories from one exact shared core and their own exact entry sources. Tectonic ran with cached resources only, untrusted input mode, retained logs/intermediates, recorded dependency rules, `SOURCE_DATE_EPOCH=1788739200`, and `TZ=UTC`. The exact invocations, environment and executable hashes are in [the build-attempt record](BUILD_ATTEMPT.json); [the recipe](BUILD_RECIPE.md) describes reproduction.

| View | Pages | PDF SHA256 |
| --- | ---: | --- |
| Formal | 34 | `ec3113ddfc2aba024b14e726d9e227928ab436d2d8a1cb7fc8632ff0c5ca5764` |
| Rationale | 16 | `2fe369dc1acf7958d3c9004fc7e56736cea98fc3b8debd48c49cc70be0798ae0` |
| Engineering | 25 | `7b1205cc8593e87a496d6d0c48d47bc4305e1f36b9fc971c33710a39c201533f` |

The three compiler processes returned exit code 0. Final TeX logs contain zero overfull boxes, missing characters, unresolved references, or undefined control sequences. Underfull spacing notices remain in the retained logs; visual review found no corresponding collision, clipping or unreadable content. Each PDF reopened through pypdf and Poppler and every page rendered independently. Compiler-reported local dependencies are explicitly normalized to existing files in the isolated source tree and their hashes are recorded; this is not a claim that the cached TeX resource bundle is portable or that another host will reproduce identical bytes.

The sandboxed Tectonic startup attempt failed in macOS system configuration before document compilation. Scoped cached/untrusted document invocations outside the sandbox succeeded. Earlier provisional TeX and layout failures are preserved separately in [the attempt history](DOCUMENT_BUILD_ATTEMPT_HISTORY.json); they are not counted as passing attempts. No application source, configuration, implementation test, data product, Spack realization or Unity material was used.

Observed document tools:

- tectonic: `/opt/homebrew/Cellar/tectonic/0.16.9/bin/tectonic`, SHA256 `38eff9059ed622672c9a2590415a8f01c043df4232baa459628a2cd86e512d95`; Tectonic 0.16.9.
- pdftoppm: `/Users/gwilson/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm`, SHA256 `de772e88ab9977ccde25def9b403bf42675d75f5dd82b19fbd7d8123ad183159`; pdftoppm version 26.05.0.
- pdfinfo: `/Users/gwilson/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdfinfo`, SHA256 `fee70ade670fb025343aca2b5c3a2aacacb8ed9edce1b716b233b5de924b6bf5`; pdfinfo version 26.05.0.

The local Python executable and its hash are in the build record; pypdf and Pillow identities are in [the mechanical PDF inspection](PDF_BUILD_INSPECTION.json). All completed input, source and output files are bound by the package source manifest. No application test or prospective ECS evidence was executed.

Candidate-build-02 passed compilation and visual QA but was superseded by the final scientific-source scan: REQ-036/043/046 and PRED-012 now consistently require only reached lifecycle stages, with all five required for completed realization. Candidate-build-03 recompiles that exact corrected core and rechecks the delivered PDFs. The earlier PDFs, sources, and QA reports remain preserved.

Candidate-build-03 passed document compilation and visual QA, but its exact committed candidate received three pre-owner review findings. The bounded repair clarifies role-qualified admission and unavailable coadd-response publication in the rationale, corrects one inventory filename, and restores the charter-linked recovery opening in README and Scope Brief. The shared normative core, requirements, predictions and profiles remain byte-identical. Candidate-build-04 was superseded before sealing by the final explicit coadd-response-unavailable sentence. Candidate-build-05 builds that final rationale and updated cover identities; no prior review verdict transfers to it.
