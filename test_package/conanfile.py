import os

from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout


class CitlaliTestPackage(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    test_type = "explicit"

    def requirements(self) -> None:
        self.requires(self.tested_reference_str)

    def layout(self) -> None:
        cmake_layout(self)

    def generate(self) -> None:
        CMakeDeps(self).generate()
        CMakeToolchain(self).generate()

    def build(self) -> None:
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self) -> None:
        if can_run(self):
            executable = os.path.join(
                self.cpp.build.bindirs[0],
                "citlali_package_test",
            )
            self.run(executable, env="conanrun")
            self.run("citlali --version", env="conanrun")
            self.run("citlali --dump_config", env="conanrun")
