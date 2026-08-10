# Native Spack Build Lane

This directory owns Citlali's successor native build entry. It does not
replace the existing CMake/FetchContent build until the acceptance gates in
`doc/TOLTECA_SPACK_BUILD_INTEGRATION_REVIEW_2026-07-31.md` pass.

## Supported Development Sequence

1. Materialize the exact accepted `tula_cmake`, `tula`, and `kidscpp` revisions
   in Citlali's ignored `build/spack-sources/` area:

   ```console
   $HOME/tolteca/bin/python tools/build/prepare_spack_sources.py
   $HOME/tolteca/bin/python tools/build/verify_spack_source_revisions.py
   ```
2. Use exact Homebrew LLVM 20 on Apple Silicon. AppleClang and unversioned
   Homebrew `llvm` are not accepted substitutes.
3. Use Spack 1.2.2 as the dependency and environment authority.
4. Build and run fast gates natively on macOS.
5. Push the accepted commit, then build that exact commit in user-owned space
   on Unity and run the required reduction validation.

Containers are optional CI or troubleshooting tools. They are not part of the
required local workflow.

This lane does not invoke, modify, or vendor `tolteca_deploy`. That repository
may later consume an accepted Citlali release, but Citlali owns this adaptation,
its pinned source manifest, and its Mac/Unity acceptance profiles.

## Prerequisite Check

Run:

```console
$HOME/tolteca/bin/python tools/build/check_macos_spack_prerequisites.py \
  --spack "$SPACK_ROOT/bin/spack"
```

The check rejects AppleClang, the wrong Spack release, missing pinned package
sources, and shell flags that force the independently versioned Homebrew
`libomp` into an LLVM 20 build. The Spack environment must supply a compatible
OpenMP runtime instead.

## Foundation Environment

The first environment deliberately stops below Kidscpp and Citlali:

```console
export SPACK_ROOT="$HOME/GitHub/spack"
export SPACK_PYTHON="$HOME/tolteca/bin/python"
. "$SPACK_ROOT/share/spack/setup-env.sh"

spack -e spack/environments/foundation-macos-llvm20 concretize --force
spack -e spack/environments/foundation-macos-llvm20 \
  install --show-log-on-error
spack -e spack/environments/foundation-macos-llvm20 clean --stage tula
spack -e spack/environments/foundation-macos-llvm20 \
  install -y --test=root --overwrite --show-log-on-error tula
$HOME/tolteca/bin/python tools/build/test_spack_foundation.py
```

It builds the Tula component closure needed by Citlali without OpenMP. This
separates native compiler, package-repository, and installed-consumer failures
from the Kidscpp API and full Citlali source port. It does not claim production
equivalence.

The environment constrains NetCDF-C to a shared build with explicit Spack
Szip and Zstandard dependencies and a matching shared HDF5 1.14 ABI. The HDF5
core has its optional Szip filter disabled; NetCDF owns the explicit Szip
plugin edge used by this stack. NetCDF's
CMake otherwise auto-detects unrelated Homebrew compression libraries on this
host, placing the generic Homebrew include directory ahead of the declared
HDF5 dependency. That can compile against Homebrew HDF5 2 headers while
linking Spack HDF5 1.14. The explicit graph prevents that undeclared mixed-ABI
build.

The accepted Tula CMake source now owns the normalized
`tula_deps::netcdf_cxx4` target. It locates the C++ and C libraries from the
concrete Spack graph without relying on metadata that NetCDF C++ 4.3.1 does not
install. Citlali's temporary local adapter has therefore been removed.
The upstream target does not yet propagate NetCDF-C's include directory to
installed consumers, so Citlali explicitly links `netCDF::netcdf` as a bounded
workaround. Remove that direct edge when the upstream adapter exports the
complete transitive interface.

The two installation steps are intentional. Dependencies are installed
normally first; then the first-party Tula root is rebuilt with package tests.
The stale root build stage is removed before that rebuild so Spack re-runs
configuration with `TULA_BUILD_TESTS=ON`; otherwise its incremental-stage
optimization can reuse the initial no-test configuration.
Asking Spack to test the root during the initial graph install also enables
build-time test configuration in some dependencies. The resulting fmt 9.1 and
Eigen 3.4 private test failures against macOS 26 libc++ are not failures of the
libraries or Citlali contracts. Third-party private test compatibility is not
a Citlali gate.

OpenMP remains a separate required gate because Homebrew LLVM 20 does not
bundle `libomp`. The accepted upstream `tula-perflibs` package now declares
exact `llvm-openmp@20.1.8` on macOS and exports the resolved runtime to
installed consumers.

## Kidscpp Environment

The next environment extends the foundation with Kidscpp and OpenMP:

```console
spack -e spack/environments/kidscpp-macos-llvm20 concretize --force
spack -e spack/environments/kidscpp-macos-llvm20 \
  install --show-log-on-error
$HOME/tolteca/bin/python tools/build/test_spack_kidscpp.py \
  --require-real-data --fixture /path/to/raw-timestream.nc
```

Kidscpp's `+openmp` variant now propagates through Tula's `+perflibs+openmp`
contract. The graph uses the upstream Tula CMake recipe and target; Citlali no
longer owns an OpenMP compatibility recipe.

The acceptance tool first builds the Kidscpp repository's independent
installed-package consumer. With `--require-real-data`, it also builds a
separate reader consumer, records the fixture SHA-256, opens the supplied raw
TolTEC NetCDF file, and reads a two-sample I/Q slice. Omitting the fixture is
useful for a fast API check but is not a complete Kidscpp gate.

The upstream native test suite currently assumes a historical file under
`TOLTECA_TEST_DATA_ROOT`. Its CMake configuration exports an empty environment
value when that root is unavailable, so the test does not skip and fails on
the missing path; the invalid-stride companion can then pass for the wrong
reason. The local real-reader consumer is the current macOS data-path evidence
until that historical fixture has an accessible immutable manifest. Solver,
Welch, and synthetic metadata tests compile and run under the same LLVM 20 and
OpenMP graph.

## Full Citlali Environment

The full environment carries the refactored library, production CLI, complete
compiled test surface, direct HDF5/Zlib ownership, normalized Tula CCfits
ownership, and distinct pipeline-OpenMP and OpenMP-Wiener build identities:

```console
spack -e spack/environments/citlali-macos-llvm20 concretize --force
spack -e spack/environments/citlali-macos-llvm20 clean --stage citlali || true
spack -e spack/environments/citlali-macos-llvm20 \
  install -y --overwrite --show-log-on-error citlali
```

This is the packaging and release-candidate gate. Do not use repeated
`spack install` calls as the ordinary edit/build loop: a development package
is restaged and its header-heavy CLI translation unit can dominate the
rebuild.
Removing Citlali's stage before the packaging install is required when the
source commit changes without changing the concrete dependency graph. Spack's
incremental development stage can otherwise reuse an executable carrying the
previous source revision; the installed-artifact gate rejects that mismatch.

Use the persistent native build tree instead:

```console
$HOME/tolteca/bin/python tools/build/run_spack_citlali_dev.py all --fresh
```

After the first configure, the normal cycle is:

```console
$HOME/tolteca/bin/python tools/build/run_spack_citlali_dev.py build
$HOME/tolteca/bin/python tools/build/run_spack_citlali_dev.py test
```

The script validates the concrete graph before every action, runs CMake/Ninja
inside the exact Citlali dependency environment, and embeds the concrete root
DAG hash in the CLI. It does not reinstall dependencies or the Citlali
package. The current native checkpoint passes all 539 enabled CTests; the sole
disabled lifecycle test remains explicitly reported by CTest. A measured
no-op invocation completed in 0.82 seconds.

After installing a candidate, run the installed-artifact gate:

```console
$HOME/tolteca/bin/python tools/build/test_spack_citlali.py
```

The gate rejects graph drift, checks the installed CLI version and full help
surface, requires a clean exact-HEAD source plus
source/build/compiler/variant/DAG identity in `--version`,
builds and tests an independent `find_package(citlali)` consumer, and reruns
the complete compiled suite from the persistent developer tree. The
`--skip-developer-ctest` option is only for diagnosing packaging in isolation;
it is not a complete acceptance result.

The current macOS graph deliberately uses Homebrew FFTW and a Homebrew GCC 15
Fortran compiler as declared externals. All Citlali, Kidscpp, Tula, and other
C/C++ compilation remains exact Homebrew LLVM 20. The prerequisite checker
verifies both host externals rather than allowing them to be selected
silently.

## Managed Deployment Identity

Ordinary developer launches are labeled `unmanaged`. A release or acceptance
activation must export all three deployment values together:

```console
export TOLTECA_CPP_ENV=/absolute/path/to/concrete/spack/environment
export TOLTECA_SPACK_PROFILE=unity-gcc13
export TOLTECA_SPACK_LOCK_SHA256="$(sha256sum "$TOLTECA_CPP_ENV/spack.lock" | awk '{print $1}')"
```

The lock must have exactly one root and its root DAG hash must equal the DAG
compiled into the Citlali executable. A partial identity, malformed digest,
missing lock, or DAG mismatch fails before reduction setup. `--version`
reports the profile, lock digest, and `binding=dag-match`; FITS, NetCDF, and
product-index metadata retain the same deployment identity. The activation
layer is responsible for computing the lock digest from the real file;
Citlali independently checks the scientifically relevant executable-to-DAG
binding rather than trusting a profile label alone.

## Unity GCC 13 Acceptance Profile

Unity's application modules are generated by a system Spack installation, but
the Spack command itself is not exposed to users. The first acceptance lane
therefore uses an isolated Spack 1.2.2 checkout and install tree in user-owned
space. It does not modify the cluster module tree or the existing Citlali
build.

The measured host compiler is Ubuntu GCC 13.3.0, including matching GCC, G++,
and GFortran frontends. GFortran is registered because OpenBLAS requires a
Fortran compiler internally even when its public Fortran interface is
disabled. This profile treats GCC 13 as a new Unity acceptance lane; it does
not relabel it as part of the upstream GCC 14/LLVM 20 matrix. All compiled
tests and real reductions remain required evidence.

Bootstrap Spack once in the acceptance workspace:

```console
export WORKSPACE="$HOME/work_toltec/citlali_spack_acceptance"
export SPACK_ROOT="$HOME/work_toltec/spack-1.2.2"
export SPACK_PYTHON="$(command -v python3)"

git clone --branch v1.2.2 --depth 1 \
  https://github.com/spack/spack.git "$SPACK_ROOT"
. "$SPACK_ROOT/share/spack/setup-env.sh"
spack --version
```

Prepare the exact first-party sources declared by the Citlali manifest:

```console
cd "$WORKSPACE/citlali"
"$SPACK_PYTHON" tools/build/prepare_spack_sources.py
"$SPACK_PYTHON" tools/build/verify_spack_source_revisions.py
```

Run the fail-fast host gate before concretizing:

```console
cd "$WORKSPACE/citlali"
"$SPACK_PYTHON" tools/build/check_unity_spack_prerequisites.py \
  --workspace-root "$WORKSPACE/citlali/build/spack-sources" \
  --spack "$SPACK_ROOT/bin/spack" \
  --spack-python "$SPACK_PYTHON"
```

Concretize and inspect before installing anything:

```console
ENV="$WORKSPACE/citlali/spack/environments/citlali-unity-gcc13"
spack -e "$ENV" concretize --force
spack -e "$ENV" find -cvl
```

The concrete root must report `citlali@4.0.0+openmp+tests+wiener_openmp` with
`%cxx=gcc@13.3.0`. Stop if it selects a second C/C++ compiler, an external
first-party package, or a package outside the pinned build-source area.

After graph review, create the log directory and submit the checked acceptance
script from the Citlali checkout. The submission captures the exact source SHA
and refuses to run if the checkout or any pinned first-party revision drifts:

```console
cd "$WORKSPACE/citlali"
mkdir -p logs
EXPECTED_CITLALI_SHA="$(git rev-parse HEAD)"
sbatch --export=ALL,WORKSPACE="$WORKSPACE",EXPECTED_CITLALI_SHA="$EXPECTED_CITLALI_SHA" \
  tools/build/run_unity_spack_acceptance.sh
```

The job source-builds and installs the graph, configures the persistent
developer tree, runs all enabled CTests, tests the installed CLI and independent
consumer, and writes a provenance manifest under `logs/`. Only after that job
passes should the installed executable be snapshotted through TolProj and used
for a point smoke reduction.
