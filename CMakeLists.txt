cmake_minimum_required(VERSION 3.12)
project(citlali LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/common_utils/cmake/CommonSettings.cmake")

set (CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(PythonInterp   2.7     REQUIRED)
find_package(PythonLibs     2.7     REQUIRED)
#find_package(NetCDF REQUIRED COMPONENTS CXX4)
include(NetCDFCXX4)
include(FileSystem)
include(SpdlogAndFmt)
include(Boost)
include(SharedMemory)
include(Grppi)
include(Eigen3)
include(Ceres)
include(Tclap)
include(Spectra)
include(Yaml)
include(MXX)
if (VERBOSE)
    print_target_properties(NetCDF::NetCDFCXX4)
    print_target_properties(Boost::boost)
    print_target_properties(Eigen3::Eigen)
    print_target_properties(ceres::ceres)
    print_target_properties(tclap::tclap)
    print_target_properties(Spectra::Spectra)
    print_target_properties(yaml-cpp::yaml-cpp)
    print_target_properties(mxx::mxx)
endif()

# lali_core
add_library(citlali_core INTERFACE)
target_include_directories(citlali_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(citlali_core
    INTERFACE
        NetCDF::NetCDFCXX4
        spdlog::spdlog
        fmt::fmt
        Boost::boost
        Eigen3::Eigen
        ceres::ceres
        grppi::grppi
        yaml-cpp::yaml-cpp
        mxx::mxx
    )

include_directories(${PYTHON_INCLUDE_DIRS})
link_libraries(${PYTHON_LIBRARIES})

# executables
add_executable(lali)
target_sources(lali PRIVATE src/lali.cpp)
target_link_libraries(lali PRIVATE citlali_core)

target_link_libraries(lali PRIVATE
    citlali_core
    tclap
    spectra
    Boost::program_options
    Boost::random
    )
