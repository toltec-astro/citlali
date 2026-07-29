Citlali
=======

Citlali is the TolTEC data-reduction pipeline engine. This branch,
``v4.x_conan2``, is the Conan 2 port of the v4 library and is based directly
on ``v4.x``.

Current scope
-------------

The verified GCC 13 slice builds the same five library sources selected by the
v4 CMake project:

* calibration;
* telescope data handling;
* mapmaking;
* PTC sensitivity;
* Gaussian models.

The package consumes ``kidscpp/3.1.0``, which consumes ``tula/3.1.0``. Public
headers and libraries propagate through Conan package metadata; sibling source
directories are not build inputs.

The ``citlali`` reduction CLI is built and installed with the library. It
retains the v4 configuration and observation workflow, but calls the explicit
Kidscpp raw-reader and timestream-solver APIs. Citlali does not build or expose
the former Kidscpp multipurpose CLI.

Raw TolTEC NetCDF ingestion remains a Kidscpp responsibility. Citlali chooses
observation files and sample slices, then calls
``kids::toltec::get_raw_timestream_meta`` and
``kids::toltec::read_raw_timestream_slice`` before invoking
``kids::TimeStreamSolver``. Citlali does not carry a duplicate file parser,
and the removed Kidscpp sweep fitter and multipurpose CLI are not restored.

Citlali adds Conan-backed Spectra, Boost, FFTW, CCfits, and Ceres features.
Their versions and normalized CMake targets are owned by the ``tula_cmake``
registry. Ceres uses dense QR for the current fitting path, so generated Schur
specializations are disabled to reduce compile-time memory.

Quick start on a fresh machine
------------------------------

The currently verified platform is 64-bit Linux with GCC 13. Creating
``citlali/4.0.0`` installs both ``libcitlali.a`` and ``bin/citlali`` into the
package. A Conan ``VirtualRunEnv`` places the packaged executable on ``PATH``
for a consuming environment.

Prerequisites
^^^^^^^^^^^^^

Install GCC 13, Git, CMake 3.25 or newer, Ninja, Python 3.11 or newer, the
NetCDF C and C++ development packages, and `uv <https://docs.astral.sh/uv/>`_.
For example, on a Debian-derived GCC 13 system:

.. code-block:: console

   sudo apt-get update
   sudo apt-get install \
       cmake g++-13 git libnetcdf-c++4-dev libnetcdf-dev ninja-build \
       python3 python3-venv
   curl -LsSf https://astral.sh/uv/install.sh | sh

OpenMP and Threads are supplied by the GCC toolchain. NetCDF is deliberately
resolved from the operating system in the current feature selection.

On macOS, install ``llvm``, ``libomp``, ``netcdf``, and ``netcdf-cxx`` with
Homebrew. Use the generated ``macos-brew-llvm-debug`` profile; the supported
macOS gate intentionally uses Homebrew ``clang++`` with libc++, not native
AppleClang.

Released workflow
^^^^^^^^^^^^^^^^^

Once the TolTEC packages and shared Conan configuration are published, the
complete source-build interface is:

.. code-block:: console

   git clone --branch v4.x_conan2 https://github.com/toltec-astro/citlali.git
   cd citlali
   ./build

The launcher obtains the pinned ``tula_cmake`` CLI from its GitHub tag. The
CLI installs the shared Conan configuration, resolves the package graph,
generates CMake presets, configures, and builds. Tula and kidscpp are not Git
submodules or FetchContent projects.

The final organization configuration source may be supplied without changing
the repository:

.. code-block:: console

   TULA_CONAN_CONFIG_SOURCE=https://example.org/toltec-conan-config.zip \
       ./build

The URL is intentionally a placeholder until the TolTEC Conan service is
deployed.

Local pre-publication workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before that remote exists, developers need the four sibling repositories at
their local release branches:

.. code-block:: text

   tula_cmake  v3.x_conan2
   tula        v3.x
   kidscpp     v3.x
   citlali     v4.x_conan2

The workspace ``just citlali`` gate exports ``tula-cmake/3.1.0`` and creates
Tula, kidscpp, then Citlali in one isolated Conan home. To exercise the
release-facing launcher against the local CLI, set:

.. code-block:: console

   TULA_CMAKE_DEV_PROJECT=../tula_cmake ./build

Where dependencies come from
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Dependency
     - Retrieval
     - Current behavior
   * - ``tula_cmake/3.1.0``
     - TolTEC Conan remote
     - Recipe infrastructure; exported locally only before publication.
   * - ``tula/3.1.0``
     - TolTEC Conan remote
     - Header package; created locally only before publication.
   * - ``kidscpp/3.1.0``
     - TolTEC Conan remote
     - Static library; created locally only before publication.
   * - ``citlali/4.0.0``
     - Local source or TolTEC remote
     - Current static library and the target of this quick start.
   * - Conan dependencies
     - ConanCenter
     - Includes fmt, spdlog, yaml-cpp, Eigen, Spectra, Boost, FFTW,
       CCfits, Ceres, and their transitive dependencies.
   * - CPM dependencies
     - Upstream source archives
     - Downloaded by CMake using versions and checksums from the
       ``tula_cmake`` registry.
   * - System dependencies
     - Operating-system packages
     - NetCDF C/C++, Threads, and the selected compiler's OpenMP runtime
       (GNU on Linux or Homebrew ``libomp`` with Homebrew LLVM on macOS).

There is no hidden sibling-source lookup during a package build. After each
``conan create``, downstream packages resolve Tula and kidscpp from
``$CONAN_HOME`` through ordinary Conan requirements.

A downstream CMake project normally declares
``self.requires("citlali/4.0.0")`` in its own recipe. Conan then retrieves the
complete internal and third-party graph from the configured virtual remote.

Build and test
--------------

The workspace development gate runs the complete package chain in the GCC 13
dev container:

.. code-block:: console

   just citlali

The recipe creates Tula, kidscpp, and Citlali in a fresh Conan home, compiles
``libcitlali.a`` and ``citlali``, then runs the library, CLI, and real-data
adapter CTests. Set
``CITLALI_CONAN_HOME`` to a disposable cache directory only when iterating
locally; the default gate remains isolated.

The package consumer also runs ``citlali --version`` and
``citlali --dump_config`` from the installed Conan package. The active RTC
adapter is tested against a real TolTEC NetCDF slice and compared sample by
sample with the direct Kidscpp reader/solver path, including matching NaNs.

Reference policy
----------------

``refs/citlali`` is read-only evidence for the previous production
implementation. Changes belong in this active repository.

License
-------

3-Clause BSD.
