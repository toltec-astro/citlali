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
library), CCfits, CFITSIO, Kidscpp, and the Tula capabilities used by its public
headers.

## Fresh development machine

The supported reproducible setup is the workspace dev container:

1. Install Docker and a Dev Container-capable client.
2. Clone `tula_cmake`, `tula`, `kidscpp`, `citlali`, and
   `tolteca_test_data` as siblings.
3. Rebuild the container; `.devcontainer/postCreate.sh` installs Spack 1.2.2,
   GCC 14, LLVM 20, and required system development packages.
4. Run:

   ```console
   cd /workspaces/cpp
   just production
   ```

The native commands for one lane are:

```console
spack -e tula_cmake/environments/production/gcc14 concretize --force
spack -e tula_cmake/environments/production/gcc14 \
  install --test=all --overwrite --show-log-on-error
spack -e tula_cmake/environments/production/gcc14 \
  location -i citlali
```

Replace `gcc14` with `llvm20` for Clang 20. Both lanes require C++23.

## Use the installed executable

The environment view contains the root executable:

```console
tula_cmake/environments/production/gcc14/.spack-view/bin/citlali --help
```

Alternatively:

```console
spack -e tula_cmake/environments/production/gcc14 load citlali
citlali --version
```

## Consume the library

```cmake
find_package(citlali 4 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE citlali::citlali)
```

`tests/installed_consumer` verifies this contract from the installed prefix,
including transitive NetCDF, FFTW threads, CCfits/CFITSIO, Ceres, Kidscpp, and
Tula metadata.

## Validation

Measured in Ubuntu 24.04 arm64:

| Lane | Citlali tests | Installed consumer |
| --- | --- | --- |
| GCC 14.2 / C++23 | 6/6 | pass |
| Clang 20.1.2 / C++23 | 6/6 | pass |

The six tests include CLI help/version/config checks and a real-data RTC
comparison against the direct Kidscpp reader/solver path.

The complete observation-level gate uses the installed CLI, eleven TolTEC
NetCDF streams, the recomputed telescope stream, and the APT ECSV table:

```console
cd /workspaces/cpp
just citlali-real
```

Run it from the workspace root after rebuilding the dev container so the
external `toltec_astro/run` source tree is mounted read-only. The fixture
writes generated FITS products under `/tmp/citlali-o149101-output` and retains
the full CLI log under `tula_cmake/build/citlali-real-workdir/`.

The prior Conan implementation is preserved on its baseline branch and in the
workspace archive. `refs/citlali` remains read-only evidence.
