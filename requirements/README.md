# Configured POINT Python deployment

The installable application contains an explicit 28-module runtime plus its packaging command. It preserves the existing configured POINT options, TolTECA merged-YAML boundary, versioned schemas, native `--version` delegation, and copyable TolProj launcher. The native engine and its shared libraries remain separate dependencies. `0.0.0` identifies an unqualified packaging candidate, not a collaborator release.

The source compatibility target is CPython 3.11–3.13. CPython 3.13.2 on macOS arm64 has been exercised locally. Linux/Unity and the other Python versions remain qualification targets; compatibility declarations are not validation evidence. Use the accepted Spack native realization for supported Unity qualification. Neither these Python files nor local tests establish SCIENCE, coadd, calibration, or production acceptance.

## Exact local recreation

The three `macos-arm64-cp313-*.lock` files bind full cached wheel artifacts by SHA-256. Runtime, optional plotting diagnostics, and development/build tools are separate layers. `macos-arm64-cp313-artifacts.json` records the corresponding filenames, versions, wheel tags and byte counts. These hashes were obtained from complete existing wheel artifacts, not installed-file fingerprints. They are specific to CPython 3.13 on macOS arm64 and must not be used as a Linux lock.

Create a separate environment; do not install into Grant's shared `$HOME/tolteca` environment. From the source checkout, with a chosen new environment path:

```bash
$HOME/tolteca/bin/python -m venv /absolute/path/to/new-point-env
/absolute/path/to/new-point-env/bin/python -m pip install --require-hashes -r requirements/macos-arm64-cp313-dev.lock
/absolute/path/to/new-point-env/bin/python -m build --wheel --no-isolation
/absolute/path/to/new-point-env/bin/python -m pip install --no-deps dist/citlali_configured_point-0.0.0-py3-none-any.whl
/absolute/path/to/new-point-env/bin/python -m pip check
```

For offline recreation, supply `--no-index --find-links /absolute/path/to/verified-wheelhouse` to the locked dependency installation. A local fresh environment was recreated this way using all 31 cached distributions. Preserve the lock files, artifact manifest, application wheel hash, source commit, Python interpreter identity, and native environment identity with the deployment record. The application wheel changes with source; a version string alone does not identify its bytes.

Install only `runtime.lock` when plotting is not required, or `diagnostics.lock` when it is. Development gates use `dev.lock`. The `diagnostics` project extra likewise supplies Matplotlib. Ordinary operation does not import plotting backends unless a diagnostic path requests them.

## Linux target lock candidate

The `linux-x86_64-cp313-*.lock` layers contain the same 31 exact package versions, with complete Linux wheel hashes downloaded from official PyPI. The artifact manifest records wheel tags, active target dependency metadata and the target marker environment. Every selected wheel is compatible with CPython 3.13/x86_64; the largest required glibc floor is **2.28**. Target-specific dependency markers and pinned versions form a complete closure. No versions were silently substituted.

This is a prepared deployment candidate: the Linux wheels were downloaded and inspected on the Mac, never installed or executed. Unity was not accessed. Grant must create an isolated target environment, install the hash-locked wheels, and run the synthetic/configuration/native gates before the target deployment is qualified. On Unity, use the cluster-provided `python` and the accepted Spack native realization; the Python lock does not replace or validate native dependency policy. A glibc version below 2.28 is a concrete installation blocker for this wheel set.

From the owner-selected source checkout on Linux, use `python -m venv /absolute/path/to/new-point-env`, then the analogous locked installation with `requirements/linux-x86_64-cp313-dev.lock`. Check that the chosen interpreter is CPython 3.13 before creating the environment. `--no-index --find-links` permits installation from a separately transferred verified Linux wheelhouse. No reduction launch is part of installation or qualification here.

For CPython 3.11/3.12, `pyproject.toml` declares compatibility targets, but no exact wheel lock or runtime verification is supplied. Do not substitute either 3.13 wheel set for those interpreters. Additional target qualification must preserve exact artifact hashes and pass gates.

## Installed commands and immutable launchers

`citlali-configured-point --native /absolute/path/to/citlali-native --dump_config` prints the existing profile without launching a reduction. Installed resources come from the environment's `share/citlali-point/data`; source checkouts and frozen bundles retain their existing `data/` layout.

For TolProj, retain the immutable, copyable launcher workflow using `citlali-package-point --repository /absolute/path/to/committed/source --native /absolute/path/to/matching/native --python /absolute/path/to/new-point-env/bin/python --destination /absolute/path/to/fresh/runtime`. It checks exact native/source revision agreement and copies only the declared runtime members. Packaging does not launch reductions. Grant alone launches reductions.

The local frozen launcher still binds its absolute bundle and interpreter paths. Its manifest reports external packages as unpinned because launch-time verification does not hash every installed package or native shared library. A separately verified dependency lock records environment installation provenance; it does not turn that mutable external environment into a content-verified runtime snapshot. Keep the environment unchanged or reverify/recreate it before preparing another launcher. Existing frozen launchers and trials are not rewritten.

## Maintained-code gates

Ruff lint/format is intentionally limited in `pyproject.toml` to the configured application modules and packaging tool. The selected `E4,E7,E9,F` baseline is checked without blanket suppressions. Historical tools are not reformatted. Pyright checks the typed request/preparation/resource boundaries; it does not claim complete typing of historical numerical dictionaries. Pyright needs Node.js already provisioned on PATH; the local check used the existing Codex-bundled Node runtime and installed Pyright 1.1.405, with no download.

```bash
$HOME/tolteca/bin/python -m ruff check --no-cache .
$HOME/tolteca/bin/python -m ruff format --check --no-cache .
$HOME/tolteca/bin/python -m pyright
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/citlali-point-mpl XDG_CACHE_HOME=/private/tmp/citlali-point-cache PYTHONDONTWRITEBYTECODE=1 \
  $HOME/tolteca/bin/python -m pytest -p no:cacheprovider -q tools/timestream_successor/test_configured_point_*.py tools/timestream_successor/test_package_configured_point.py
```

The runtime-membership tests check transitive local imports and wheel/freezer parity. Packaging tests exercise a copied launcher, exact inventory, omitted historical tools, payload tampering, interpreter identity, and failure propagation. Fixture imports in existing tests remain intentional.

Historical experiment modules still enter the conservative runtime closure because the current measurement helpers import them. Their names are explicit in `configured_point_runtime.RUNTIME_MODULES`; there is no arbitrary glob copying all successor tools. They can be removed only after their reusable services are extracted and both callers are tested, without changing scientific behavior.
