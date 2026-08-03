# Citlali 4.0

Citlali is the TolTEC data-reduction pipeline engine. The `v3.x_spack` branch
changes the build and distribution infrastructure while retaining the v4
library and CLI behavior.

## Implemented scope

The package builds the same five v4 library sources:

- calibration;
- telescope data handling;
- mapmaking;
- PTC sensitivity; and
- Gaussian models.

It also builds and installs the `citlali` CLI. Citlali consumes
`kids::kids`; raw TolTEC NetCDF metadata reading and slicing remain Kidscpp
responsibilities. Citlali calls the Kidscpp reader and timestream solver rather
than carrying a second adapter. The removed Kidscpp multipurpose CLI and sweep
fitter are not rebuilt inside Citlali.

Citlali directly consumes Ceres, Boost, Spectra, FFTW (including the threaded
library), Kidscpp, and the Tula capabilities used by its public headers. FITS
I/O uses the `tula-ccfits` dependency adapter, which supplies the CCfits C++ API
and its CFITSIO C dependency through one `tula_deps::ccfits` target. Spack may satisfy
the pair from platform externals or build both from source.

OpenMP is the explicit Spack variant `citlali+openmp` (the default). It
propagates consistently to Kidscpp, Tula, and `tula-perflibs`. Citlali links
`tula::perflibs` and does not discover an OpenMP runtime itself. On native
macOS, the stack pairs Homebrew Clang 20.1.8 with Spack-built
`llvm-openmp@20.1.8`; Linux uses the matching compiler runtime. The
`citlali~openmp` graph remains available and consistently removes the
capability across the transitive package chain.

## Fresh development machine

The maintained orchestration lives in `tolteca_deploy`. Clone
`tolteca_deploy`, `tula_cmake`, `tula`, `kidscpp`, and `citlali` as siblings,
select one of its native Spack profiles, and let it generate a location-local
environment. The repositories own their package recipes and development source
mappings; the deployment location owns compiler policy, externals, the lock,
build stage, environment, and view. Its install store and download cache follow
the configured user-shared or location-scoped storage policy.

Once generated, the underlying workflow remains ordinary Spack:

```console
cd ../toltec_astro_dev
source dotbashrc
just cpp-install
spack -e "$TOLTECA_CPP_ENV" location -i citlali
```

Choose the generated profile in `location.yaml`. Linux development profiles
use GCC 14 or LLVM 20; the native macOS profile uses Homebrew LLVM 20. All
lanes require C++23.

## Use the installed executable

The environment view contains the root executable:

```console
"$TOLTECA_CPP_VIEW/bin/citlali" --help
```

Alternatively:

```console
spack -e "$TOLTECA_CPP_ENV" load citlali
citlali --version
```

## Consume the library

```cmake
find_package(citlali 4 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE citlali::citlali)
```

`tests/installed_consumer` verifies this contract from the installed prefix,
including transitive NetCDF, FFTW threads, the `TulaCcfits` adapter, Ceres,
Kidscpp, and Tula metadata.

`spack_repo/develop.yaml` declares that the `citlali` development spec maps to
this source root. The deployment repository selects revisions and profiles;
Citlali retains ownership of its Spack package-to-source mapping.

## Validation

Measured in Ubuntu 24.04 arm64:

| Lane | Citlali tests | Installed consumer |
| --- | --- | --- |
| GCC 14.2 / C++23 | 6/6 | pass |
| Clang 20.1.2 / C++23 | 6/6 | pass |

The six tests include CLI help/version/config checks and a real-data RTC
comparison against the direct Kidscpp reader/solver path.

The native macOS arm64 Homebrew LLVM 20.1.8 profile also concretizes and
installs the complete graph. Its installed Citlali 4.0 CLI processed all 123
scans in observation 149101 and wrote raw and filtered FITS products for all
three arrays. Runtime memory reporting uses portable `getrusage` semantics, so
the diagnostic no longer assumes Linux `/proc` exists.

The complete observation-level gate uses the installed CLI, eleven TolTEC
NetCDF streams, the recomputed telescope stream, and the APT ECSV table:

```console
cd ../tula_cmake
just citlali-real-workdir
```

Run it with the sibling `tolteca_test_data/tolteca_workdir` fixture available.
The gate writes generated FITS products under `/tmp/citlali-o149101-output`
and retains the full CLI log under
`tula_cmake/build/citlali-real-workdir/`.

The prior Conan implementation is preserved on its baseline branch and in the
workspace archive. `refs/citlali` remains read-only evidence.
