include_guard(GLOBAL)
include(FetchContentHelper)
FetchContentHelper(benchmark GIT https://github.com/google/benchmark.git v1.5.0
    ADD_SUBDIR CONFIG_SUBDIR
        BENCHMARK_ENABLE_GTEST_TESTS=OFF
        BENCHMARK_ENABLE_TESTS=OFF
        BENCHMARK_ENABLE_TESTING=OFF
        BENCHMARK_ENABLE_ASSEMBLY_TESTS=OFF
   PATCH_SUBDIR
        ${FCH_PATCH_DIR}/patch.sh "benchmark_*.patch"
    )
FetchContentHelper(googletest GIT https://github.com/google/googletest.git release-1.8.1 ADD_SUBDIR)
enable_testing()
# https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/EmulateMakeCheck
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
include(GoogleTest)
