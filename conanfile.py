import os

from conan import ConanFile
from conan.tools.cmake import CMake


class CitlaliRecipe(ConanFile):
    """Conan 2 recipe for the Citlali v4 data-reduction library."""

    name = "citlali"
    version = "4.0.0"
    description = "TolTEC data-reduction pipeline library"
    license = "BSD-3-Clause"
    url = "https://github.com/toltec-astro/citlali"
    package_type = "static-library"
    required_conan_version = ">=2.31"
    python_requires = "tula-cmake/3.1.0"
    python_requires_extend = "tula-cmake.TulaConan"
    settings = ()
    options = {}
    default_options = {
        "boost/*:header_only": True,
        "ceres-solver/*:use_schur_specializations": False,
        "fftw/*:precision_single": False,
        "fftw/*:precision_longdouble": False,
        "fftw/*:threads": True,
        "kidscpp/*:logging": "conan",
        "kidscpp/*:yaml_cpp": "conan",
        "kidscpp/*:csv_parser": "cpm",
        "kidscpp/*:netcdf_c": "system",
        "kidscpp/*:netcdf_cxx4": "system",
        "kidscpp/*:bitmask": "cpm",
        "kidscpp/*:meta_enum": "cpm",
        "kidscpp/*:perflibs": "system",
        "kidscpp/*:eigen": "conan",
        "kidscpp/*:grppi": "cpm",
        "tula/*:logging": "conan",
        "tula/*:yaml_cpp": "conan",
        "tula/*:csv_parser": "cpm",
        "tula/*:netcdf_c": "system",
        "tula/*:netcdf_cxx4": "system",
        "tula/*:bitmask": "cpm",
        "tula/*:meta_enum": "cpm",
        "tula/*:perflibs": "system",
        "tula/*:eigen": "conan",
        "tula/*:grppi": "cpm",
    }
    tula_default_options = {
        "logging": "conan",
        "yaml_cpp": "conan",
        "csv_parser": "cpm",
        "netcdf_c": "system",
        "netcdf_cxx4": "system",
        "bitmask": "cpm",
        "meta_enum": "cpm",
        "perflibs": "system",
        "eigen": "conan",
        "grppi": "cpm",
        "spectra": "conan",
        "boost": "conan",
        "fftw": "conan",
        "ccfits": "conan",
        "ceres": "conan",
        "clipp": "conan",
    }
    tula_public_features = tuple(tula_default_options)
    exports_sources = (
        "CMakeLists.txt",
        "data/*",
        "include/*",
        "src/*",
        "tests/*",
    )

    def requirements(self) -> None:
        super().requirements()
        self.requires(
            "kidscpp/3.1.0",
            transitive_headers=True,
            transitive_libs=True,
        )

    def build_requirements(self) -> None:
        if not self.conf.get("tools.build:skip_test", default=False, check_type=bool):
            self.test_requires("gtest/1.17.0")

    def build(self) -> None:
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        if not self.conf.get("tools.build:skip_test", default=False, check_type=bool):
            cmake.ctest()

    def package(self) -> None:
        CMake(self).install()

    def package_info(self) -> None:
        self.cpp_info.set_property("cmake_file_name", "citlali")
        self.cpp_info.set_property("cmake_target_name", "citlali::citlali")
        self.cpp_info.libs = ["citlali"]
        self.runenv_info.prepend_path(
            "PATH", os.path.join(self.package_folder, "bin")
        )
        super().package_info()
