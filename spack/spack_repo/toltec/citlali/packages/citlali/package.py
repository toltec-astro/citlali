"""Spack package for the full refactored Citlali application."""

from spack.package import depends_on, variant, version
from spack_repo.builtin.build_systems.cmake import CMakePackage


class Citlali(CMakePackage):
    """Build the refactored Citlali library and production CLI."""

    homepage = "https://github.com/toltec-astro/citlali"
    root_cmakelists_dir = "cmake/spack"

    version("4.0.0")

    variant(
        "wiener_openmp",
        default=True,
        description="Build the OpenMP Wiener-filter implementation",
    )
    variant(
        "tests",
        default=False,
        description="Build and discover the compiled Citlali tests",
    )

    depends_on("cmake@3.25:", type="build")
    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("googletest@1.17.0", when="+tests", type=("build", "link"))
    depends_on("pkgconf", type="build")
    depends_on("tula-cmake@3.2.0", type="build")
    depends_on("tula-perflibs@0.1.0+openmp", type=("build", "link"))
    depends_on("kidscpp@3.1.0", type=("build", "link"))
    depends_on(
        "tula@3.1.0+ecsv+netcdf+enum+cli+grppi+openmp~fitting",
        type=("build", "link"),
    )
    depends_on("ceres-solver@2.2.0", type=("build", "link"))
    depends_on("boost@1.83.0", type=("build", "link"))
    depends_on("spectra@1.0.1", type=("build", "link"))
    depends_on("fftw@3.3.10~mpi", type=("build", "link"))
    depends_on("ccfits@2.6", type=("build", "link"))
    depends_on("cfitsio@4.3.0~fortran", type=("build", "link"))
    depends_on("hdf5@1.14.6+hl~fortran~mpi", type=("build", "link"))
    depends_on("zlib-api", type=("build", "link"))

    def cmake_args(self) -> list[str]:
        """Build the full CLI through the parallel Spack project."""
        return [
            self.define("CITLALI_BUILD_CLI", True),
            self.define("CITLALI_SPACK_DAG_HASH", self.spec.dag_hash()),
            self.define_from_variant("CITLALI_BUILD_TESTS", "tests"),
            self.define_from_variant(
                "CITLALI_USE_WIENER_FILTER_OMP", "wiener_openmp"
            ),
        ]
