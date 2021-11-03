Citlali
=======

Citlali is the official data reduction pipeline engine for
`TolTEC <http://toltec.astro.umass.edu>`_.

It is developed as part of TolTECA, the TolTEC data analysis software suite.

While citlali is developed targeting LMT/TolTEC, it can also be adapted to
with other telescope/detectors that shares similar architectural properties.


System requirements
-------------------

Citlali requires a C++20 compiler and CMake 3.20+ to build.


The software is fully tested for the following
platform/operating system/compilers:

* x86_64 macOS 11 (Big Sur); LLVM 13+

* x86_64 Ubuntu 20.04; GCC 10+

Build on other Linux-like operating systems should also work, given a C++20
capable compiler and the required dependencies.


x86_64 macOS 11 (Big Sur)
^^^^^^^^^^^^^^^^^^^^^^^^^

Homebrew is required to install the compiler, build tools, and some
optional dependencies.

To install the compiler and build tools:

.. code-block::

    $ brew install git cmake llvm libomp python conan

The package :code:`libomp` and :code:`python` are not required, but highly recommended.

:code:`conan` can be installed either through ::code:`brew` or ::code'python'.

By default, the installed LLVM compiler is in :code:`/usr/local/opt/llvm/bin`. If
not sure, consult :code:`brew info llvm`.

Optionally, the following packages can be installed via Homebrew, and be made
available to the build system via the CMake variables :code:`USE_INSTALLED_*`
(see section below for details of the CMake configuration):

.. code-block::

    $ brew install fmt spdlog gflags glog gtest benchmark boost cfitsio ccfits netcdf numpy


x86_64 Linux (Ubuntu 20.04)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following package are required to build citlali:

.. code-block::

    $ sudo apt install build-essential gcc-10 g++-10


By default, citlali requires a CMake version that is newer than what the APT
repo would provide. To install the latest version of CMake, follow the
instruction here: https://apt.kitware.com.

Optionally, the following packages can be installed and be made available to
the build system via the CMake variables :code:`USE_INSTALLED_*`:

.. code-block::

    $ sudo apt install libnetcdf-dev python3-pip python3-dev libboost-all-dev


Intel OneAPI toolkit
^^^^^^^^^^^^^^^^^^^^

For additional performance with Intel CPUs, the `Intel OneAPI toolkit (MKL) <https://software.intel.com/content/www/us/en/develop/tools/oneapi/all-toolkits.html>`_
may be installed. The instruction can be found in the official website.


Build
-----

First, clone the repo or download the source code from the github page:

.. code-block::

    $ git clone https://github.com/toltec-astro/citlali.git
    $ cd citlali

To build, go into the source directory:

.. code-block::

    // in the cloned citlali directory:
    $ mkdir build
    $ cd build
    $ cmake .. -DCMAKE_BUILD_TYPE=Release [more options...]
    $ make citlali_cli

To customize the build, add options like :code:`-D<key>=<value>` to the cmake command
line. Some options to set are:

* :code:`CMAKE_BUILD_TYPE`: The build type, can be :code:`Release` or :code:`Debug`.

* :code:`CMAKE_C_COMPILER` and :code:`CMAKE_CXX_COMPILER`: The compiler to use, if they
  are not in the default location. For example, macOS users would need to
  specify the LLVM compiler paths as::

  -DCMAKE_C_COMPILER=/usr/local/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang++

* :code:`USE_INTEL_ONEAPI`: This can be set to :code:`ON` to use the Intel OneAPI toolkit
  (MKL) for additional performance.

The build dependencies are managed by the CMake super-build scripts provided
in the `tula_cmake` folder. The following dependencies are managed in this way:

  * :code:`Boost`
  * :code:`Ceres`
  * :code:`Clipp`
  * :code:`Csv`
  * :code:`Eigen3`
  * :code:`Enum`
  * :code:`Grppi`
  * :code:`MXX`
  * :code:`NetCDF`
  * :code:`NetCDFCXX4`
  * :code:`Re2`
  * :code:`Spectra`
  * :code:`Yaml`
  * :code:`logging`
  * :code:`testing`
  * :code:`perflibs`

Each dependency comes with three CMake options to configure how it is
integrated:

* :code:`USE_INSTALLED_{dep}`: Use the dependency installed in the system via brew or apt.

* :code:`CONAN_INSTALL_{dep}`: Use `Conan <https://conan.io>`_ to install the dependency
  automatically. To use this option, the Python package :code:`conan` has to be installed::

     $ python3 -m pip install conan

* :code:`FETCH_{dep}`: Use CMake :code:`FetchContent` to download the source code of the
  package and build the dependency inline.

By default, most of the dependencies above are set to use the
:code:`CONAN_INSTALL_*` option whenever they are available in the Conan Index,
otherwise :code:`FETCH_*` is used.


Usage
-----

Once successfully built, the created executables will be available in
:code:`build/bin`.

To check the version of the program:

.. code-block::

    // In the build directory:
    $ ./bin/citlali --version

To show the help screen of the commandline interface:

.. code-block::

    // In the build directory:
    $ ./bin/citlali --help

Please see the `API documentation
<https://toltec-astro.github.io/citlali>`_ for details.


License
-------

3-Clause BSD.
