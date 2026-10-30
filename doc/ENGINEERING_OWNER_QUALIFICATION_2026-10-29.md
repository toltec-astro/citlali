# Owner-run engineering qualification — 2026-10-29

Grant performs these checks on Unity after selecting an exact delivered
engineering commit. This document was prepared on the Mac; Unity was not
accessed. These commands build software and exercise synthetic/configuration
contracts. They do not launch reductions or replace trial executables.

Use an already prepared, clean, isolated checkout under
`~/work_toltec/wilson/citlali_testing/citlali-engineering-20261029`, populated
from the owner-pushed engineering branch or an explicitly verified source
transfer. Any owner-run GitHub clone/fetch on Unity uses
`https://github.com/toltec-astro/citlali.git`. Transfers from the Mac use host
alias `unity_toltec`, with this same testing destination. Keep the existing
accepted Spack environment and installed dependencies unchanged. Do not use an
install/overwrite acceptance script for this developer build.

The accepted `unity-gcc13` profile requires the configured GCC 13.3 realization
and a concrete `spack.yaml`/`spack.lock`. Select an appropriate compute allocation
for eight build workers under local policy. The Python wheel candidate requires
CPython 3.13 on Linux x86_64 and glibc >= 2.28. If those prerequisites are absent,
resolve the target environment separately; do not substitute a Mac lock or
claim successful qualification from skipped gates.

Replace the three `REPLACE_...` values below with the final delivered full source
commit and the existing accepted Spack executable/environment paths. Start in an
already logged-in Unity session. Strict options, activation and exports are
contained in the subshell. The Python environment and native build directories
must be new; the accepted Spack environment is reused unchanged.

```bash
(
  set -euo pipefail
  source_dir="$HOME/work_toltec/wilson/citlali_testing/citlali-engineering-20261029"
  expected_revision='REPLACE_WITH_DELIVERED_FULL_SOURCE_SHA'
  spack_executable='REPLACE_WITH_EXISTING_SPACK_EXECUTABLE'
  spack_environment='REPLACE_WITH_EXISTING_ACCEPTED_SPACK_ENVIRONMENT'
  cluster_python="$(command -v python)"
  test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_revision"
  test -z "$(git -C "$source_dir" status --porcelain)"
  test -x "$spack_executable"
  test -f "$spack_environment/spack.yaml"
  test -f "$spack_environment/spack.lock"
  python -c 'import platform, sys; assert sys.version_info[:2] == (3, 13); assert platform.machine() == "x86_64"; name, version = platform.libc_ver(); assert name == "glibc" and tuple(map(int, version.split(".")[:2])) >= (2, 28)'
  qualification_dir="$source_dir/build/engineering-qualification"
  test ! -e "$qualification_dir"
  mkdir -p "$qualification_dir"
  python -m venv "$qualification_dir/python"
  source "$qualification_dir/python/bin/activate"
  cd "$source_dir"
  export MPLBACKEND=Agg
  export MPLCONFIGDIR="$qualification_dir/mpl"
  export XDG_CACHE_HOME="$qualification_dir/cache"
  export PYTHONDONTWRITEBYTECODE=1
  python -m pip install --require-hashes -r requirements/linux-x86_64-cp313-dev.lock
  python -m build --wheel --no-isolation --outdir "$qualification_dir/dist"
  python -m pip install --no-deps "$qualification_dir/dist/citlali_configured_point-0.0.0-py3-none-any.whl"
  python -m pip check
  python -I -c 'import importlib; import configured_point_runtime as r; [importlib.import_module(name) for name in r.RUNTIME_MODULES]; print(r.data_root())'
  python -I -m configured_point_cli --native /nonexistent/citlali --dump_config
  node --version
  python -m ruff check --no-cache .
  python -m ruff format --check --no-cache .
  python -m pyright
  python -m pytest -q -p no:cacheprovider tools/timestream_successor/test_configured_point_*.py tools/timestream_successor/test_package_configured_point.py
  python tools/build/prepare_spack_sources.py
  python tools/build/run_spack_citlali_dev.py all \
    --profile unity-gcc13 --source "$source_dir" \
    --spack "$spack_executable" --spack-python "$cluster_python" \
    --environment "$spack_environment" \
    --build-dir "$qualification_dir/native" --fresh -j 8
  export CITLALI_TEST_CLI="$(cat "$qualification_dir/native/citlali_cli_path.txt")"
  SPACK_PYTHON="$cluster_python" "$spack_executable" -e "$spack_environment" build-env citlali -- \
    bash -c 'source "$1" && python -m unittest discover -s tools/baseline -p "test_*.py"' \
      citlali-baseline "$qualification_dir/python/bin/activate"
  python -m unittest discover -s tools/build -p 'test_*.py'
  python -m unittest discover -s tools/config -p 'test_*.py'
  python tools/config/run_config_preflight.py --require-all --work-dir "$qualification_dir/config-preflight"
  test "$(git rev-parse HEAD)" = "$expected_revision"
  test -z "$(git status --porcelain)"
  git rev-parse HEAD
  git status --porcelain
  sha256sum "$spack_environment/spack.lock" "$CITLALI_TEST_CLI" "$qualification_dir/dist/citlali_configured_point-0.0.0-py3-none-any.whl"
)
```

The baseline subprocess reactivates the fresh environment after Spack sets its
PATH, preserving the locked Python interpreter. Full configuration preflight
requires the four existing compatibility YAML inputs named in
`tools/config/compact_compatibility_cases.yaml`; provision their exact reviewed
contents read-only before qualification. Missing required cases remain a failed
gate, not a skipped success.

Pyright requires an existing Node runtime on PATH. The Python lock does not
provision Node. Offline installations may add `--no-index --find-links` with an
owner-transferred verified Linux wheelhouse. Pinned first-party source
preparation uses only the new checkout's ignored `build/spack-sources` and
refuses to change a dirty or unexpected existing checkout.

Retain a compact execution transcript, exact source/tree, compiler and Python
identities, Spack DAG and lock digest, installed wheel/CLI hashes, CTest summary
and failed-gate diagnostics. The developer runner checks source/executable
revision agreement and runs the complete ordinary CTest aggregate. The inherited
`MapFitterLifecycle.ExactProductSequence` disabled test must remain disclosed;
newly missing executables, skipped required inputs or unexpected error-level
messages are failures. Passing these gates qualifies this environment within
engineering scope; POINT calibration, SCIENCE/coadds, downstream consumers and
production authorization still require their separate owner decisions.
