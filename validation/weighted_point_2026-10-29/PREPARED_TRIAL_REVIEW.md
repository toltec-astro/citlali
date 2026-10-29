# Prepared owner POINT trials: independent audit

**PASS. No blockers.** The two prepared trials are ready for Grant's explicit shell launch. This verdict covers preparation and provenance, not numerical execution or scientific acceptance.

Project: `/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027`

Trial directories:

- `pointings-weighted-152389-20261029`
- `pointings-weighted-152418-20261029`

The authoritative frozen application source is `331ff29ea1720b1454e671cbbab93814d8df8321` (tree `c512f8e68b7dba3255467a9b06d06903aad43185`), regardless of later documentation-only closure commits. Its runtime is `software/configured-point-weighted-331ff29ea`.

## Independent verification

- Recomputed all 241 runtime-member SHA-256 digests and byte counts. Every bundled source member was additionally compared with the exact source commit through `git show`. The actual frozen launcher `--version` reports `331ff29ea`.
- Native SHA-256: `197649397221c6b780605a732c4436ffde334c87faeaaec59e584984448f0852`.
- Manifest SHA-256: `7a39242d98e999f3a4c36ed79511b97d35af7b90b745749f9f0ccb069329be01`.
- Launcher SHA-256: `a013cac086724b7e778fd7e51ad4b7a91a2ec17ad763f52ab0f3950f237b24d5`.
- Both default `bash 02_redu.sh` invocations exited zero and explicitly reported CHECK ONLY. Neither was given `--run`. The checks exercise the actual preparation environment's TolTECA merged configuration and TolProj resource/runtime binding.
- Both trials select only their named observation, all three arrays, ranks 3/4/6, 2 arcsec pixels, coverage cut 0.15, three workers, optical-response threshold 0.10, ten feedback applications, quiet mode, required FITS/fit diagnostics and released generated transport after successful publication.
- All nine numbered configuration files per trial match READY, with the exact file set unchanged. READY-bound support files, external preparation entry/identity/IERS data, runtime snapshot and shared APT manifests also match.
- Verified all ten manifest component transport digests/byte counts for each selected matched APT. All eleven associated KMP tune files per observation uniquely match the canonical sources table and its content digest/byte count. The available networks are 0,1,2,3,4,5,7,8,9,11,12; networks 6 and 10 remain absent as inventoried. Both inputs still cover all three arrays.
- Original source numbered configurations match every saved `source_inventory` hash. The 152389 reference retains `redu00`; the 152418 reference retains `redu00` and `redu01`. All nine existing reference maps match their original RUN product digests.
- Both new `reduced` directories remain absent after the check-only calls.

## Launch and preservation guard

The driver defaults to check-only and reaches `tolteca ... reduce` only through explicit `--run`. Frozen configuration, support and runtime checks run first. `set -euo pipefail` is contained in the child Bash script. The supplied owner commands do not alter persistent interactive shell options.

Preparation creates fresh trial/evidence directories, rejects existing destinations, copies source configuration and does not remove historical outputs. The owner run sets the exact TolProj snapshot/digest and a trial-owned local runtime directory. Its only launcher `rm` is the guarded temporary executable copy. The configured application reserves a new `reduNN`; its transport cleanup is limited to generated binary files under that new run's `.point-work`, after consumers and required product publication complete, with symlink/containment checks. Existing references, shared inputs and APTs are outside those cleanup paths.

## Evidence and limitations

Audit details: `/tmp/weighted-point-prepared-audit.json`.

READY SHA-256 values:

- 152389: `28fb64e8a874f16ae773effe504ca73adce7d80c6d2ec7fd6a56b958e4a4439b`.
- 152418: `c2ad3e74065b8d0c525765690bf7a45b1bf1ef5bb3480a61b39c84e01ac892ee`.

The check-only imports emitted sandbox-specific Matplotlib warnings because the prepared evidence cache directory is read-only to this review agent; they used writable temporary cache directories and completed successfully. No unexpected error-level output occurred. Runtime retains the recorded local Mac/external-Python/shared-library and `kids 04088da-dirty` qualifications. Raw detector data were not processed or rehashed by this audit; actual low-level raw/header association validation occurs at owner launch. No reduction, Unity access, source edit, push or cleanup was performed.

Disposition by review axis: scientific/behavioral preparation **pass with execution still untested**; architectural/lifecycle preparation **pass**; repository/artifact/reference-preservation hygiene **pass**.
