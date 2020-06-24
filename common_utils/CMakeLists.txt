cmake_minimum_required(VERSION 3.12)
project(common_utils LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CommonSettings.cmake")

find_package(OpenMP REQUIRED)
find_package(NetCDF REQUIRED COMPONENTS CXX4)
include(FileSystem)
include(SpdlogAndFmt)
include(Grppi)
include(Eigen3)
if (VERBOSE)
    print_target_properties(Eigen3::Eigen)
endif()

add_library(common_utils INTERFACE)
target_include_directories(common_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(common_utils
    INTERFACE
        spdlog::spdlog
        fmt::fmt
        Eigen3::Eigen
        grppi::grppi
    )

include(GBenchAndGTest)
add_subdirectory(tests)

# test programs
option(UTILS_BUILD_TEST_PROGRAMS "Build test programs" ON)
if (UTILS_BUILD_TEST_PROGRAMS)
    add_subdirectory(test_progs)
endif()

# doc
option(UTILS_BUILD_DOCS "Build docs" ON)
if (UTILS_BUILD_DOCS)
    include(DoxygenDocTarget)
endif()
