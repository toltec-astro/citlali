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

Citlali adds Conan-backed Spectra, Boost, FFTW, CCfits, and Ceres features.
Their versions and normalized CMake targets are owned by the ``tula_cmake``
registry. Ceres uses dense QR for the current fitting path, so generated Schur
specializations are disabled to reduce compile-time memory.

Quick start on a fresh machine
------------------------------

The currently verified platform is 64-bit Linux with GCC 13. Citlali is a
static C++ library at this milestone, so "install" means creating
``citlali/4.0.0`` and its dependency graph in the local Conan cache. It does
not yet install a command-line program into ``PATH``.

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
     - NetCDF C/C++, Threads, and the GCC OpenMP runtime.

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
``libcitlali.a``, and runs the Gaussian-model CTests. Set
``CITLALI_CONAN_HOME`` to a disposable cache directory only when iterating
locally; the default gate remains isolated.

The old CLI is not part of this library milestone. It depends on kidscpp sweep
APIs and generated version headers that are outside the trimmed kidscpp v3
contract. Its product boundary will be decided separately instead of being
silently carried into the port.

Reference policy
----------------

``refs/citlali`` is read-only evidence for the previous production
implementation. Changes belong in this active repository.

License
-------

3-Clause BSD.
