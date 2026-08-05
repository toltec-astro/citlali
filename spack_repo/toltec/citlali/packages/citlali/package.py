"""Spack package for the Citlali data-reduction pipeline."""

from spack.package import (
    depends_on,
    on_package_attributes,
    run_after,
    variant,
    version,
    working_dir,
)
from spack.util.executable import Executable
from spack_repo.builtin.build_systems.cmake import CMakePackage


class Citlali(CMakePackage):
    """Build the Citlali library, CLI, and behavior tests."""

    homepage = "https://github.com/toltec-astro/citlali"
    git = "https://github.com/toltec-astro/citlali.git"

    version("4.1.0", commit="91b7febcd55a3bca18d5ef9be725c3acc1ac53f0")

    variant(
        "openmp",
        default=True,
        description="Build the pipeline with OpenMP parallelism",
    )

    depends_on("cmake@3.25:", type="build")
    depends_on("cxx", type="build")
    depends_on("pkgconf", type="build")
    depends_on("tula-cmake@3.2.0", type="build")
    depends_on("kidscpp@3.1.0+openmp", when="+openmp", type=("build", "link"))
    depends_on("kidscpp@3.1.0~openmp", when="~openmp", type=("build", "link"))
    depends_on(
        "tula@3.1.0+ecsv+netcdf+enum+cli+grppi+openmp",
        when="+openmp",
        type=("build", "link"),
    )
    depends_on(
        "tula@3.1.0+ecsv+netcdf+enum+cli+grppi~openmp",
        when="~openmp",
        type=("build", "link"),
    )
    depends_on("ceres-solver@2.2.0", type=("build", "link"))
    depends_on("boost@1.83.0", type=("build", "link"))
    depends_on("spectra@1.0.1", type=("build", "link"))
    depends_on("fftw@3.3.10", type=("build", "link"))
    depends_on("tula-ccfits@1.0.0", type=("build", "link"))
    depends_on("googletest@1.14:~shared", type=("build", "test"))

    def cmake_args(self) -> list[str]:
        """Build the production CLI and enable tests when requested."""
        return [
            self.define("CITLALI_BUILD_CLI", True),
            self.define("CITLALI_BUILD_TESTS", self.run_tests),
            self.define_from_variant("CITLALI_ENABLE_OPENMP", "openmp"),
            self.define("CITLALI_PACKAGE_SPEC", str(self.spec)),
            self.define("CITLALI_DAG_HASH", self.spec.dag_hash()),
        ]

    @run_after("build")
    @on_package_attributes(run_tests=True)
    def check(self) -> None:
        """Run library behavior and CLI smoke tests before installation."""
        with working_dir(self.build_directory):
            ctest = Executable("ctest")
            listing = ctest("-N", output=str)
            if "Total Tests: 0" in listing:
                raise RuntimeError("Citlali configured without tests")
            ctest("--output-on-failure")
