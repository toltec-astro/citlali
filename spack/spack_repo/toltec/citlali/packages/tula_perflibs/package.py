"""macOS OpenMP dependency adaptation for the Tula performance interface."""

from spack.package import depends_on, join_path, variant, version
from spack_repo.builtin.build_systems.cmake import CMakePackage


class TulaPerflibs(CMakePackage):
    """Install Threads/OpenMP with an explicit macOS LLVM runtime edge."""

    homepage = "https://github.com/toltec-astro/tula_cmake"

    version("0.1.0")

    variant(
        "openmp",
        default=True,
        description="Require and propagate compiler-native OpenMP",
    )

    depends_on("cmake@3.25:", type="build")
    depends_on("cxx", type="build")
    depends_on("tula-cmake@3.2.0", type="build")
    depends_on(
        "llvm-openmp@20.1.8",
        when="+openmp platform=darwin",
        type=("build", "link"),
    )

    def cmake_args(self) -> list[str]:
        """Translate the Spack variant into the ordinary CMake option."""
        args = [
            self.define_from_variant(
                "TULA_PERFLIBS_ENABLE_OPENMP",
                "openmp",
            )
        ]
        if self.spec.satisfies("+openmp platform=darwin"):
            prefix = self.spec["llvm-openmp"].prefix
            args.extend(
                [
                    self.define("OpenMP_CXX_FLAGS", "-fopenmp=libomp"),
                    self.define("OpenMP_CXX_LIB_NAMES", "omp"),
                    self.define(
                        "OpenMP_omp_LIBRARY",
                        join_path(prefix.lib, "libomp.dylib"),
                    ),
                    self.define(
                        "OpenMP_CXX_INCLUDE_DIR",
                        str(prefix.include),
                    ),
                ]
            )
        return args
