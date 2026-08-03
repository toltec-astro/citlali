cmake_minimum_required(VERSION 3.25)

project(
    citlali
    VERSION 4.1.0
    DESCRIPTION "TolTEC data reduction pipeline engine"
    HOMEPAGE_URL "https://toltec-astro.github.io/citlali"
    LANGUAGES CXX
)

include(CTest)
include(GNUInstallDirs)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(CITLALI_ENABLE_OPENMP "Build with OpenMP parallelism" ON)
set(CITLALI_HAS_OPENMP "${CITLALI_ENABLE_OPENMP}")

find_package(TulaCMake 3 CONFIG REQUIRED)
find_package(kidscpp 3.1 CONFIG REQUIRED)
find_package(
    tula 3.1 CONFIG REQUIRED
    COMPONENTS
        core
        logging
        yaml
        ecsv
        eigen
        netcdf
        enum
        cli
        perflibs
        grppi
)
find_package(Ceres 2.2 CONFIG REQUIRED)
find_package(Boost 1.83 CONFIG REQUIRED)
find_package(spectra 1.0 CONFIG REQUIRED)
find_package(TulaCcfits 1 CONFIG REQUIRED)
if(CITLALI_ENABLE_OPENMP AND NOT tula_OPENMP_ENABLED)
    message(FATAL_ERROR
        "Citlali requires OpenMP, but tula was built without it")
elseif(NOT CITLALI_ENABLE_OPENMP AND tula_OPENMP_ENABLED)
    message(FATAL_ERROR
        "Citlali disables OpenMP, but tula was built with it")
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
find_package(FFTW3Threads REQUIRED)

add_library(citlali_config INTERFACE)
add_library(citlali::config ALIAS citlali_config)
set_target_properties(citlali_config PROPERTIES EXPORT_NAME config)
tula_cmake_generate_config_header(
    TARGET citlali_config
    INPUT "${CMAKE_CURRENT_SOURCE_DIR}/include/citlali/config.h.in"
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/citlali/config.h"
    BUILD_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/data/config.yaml"
     CITLALI_DEFAULT_CONFIG_CONTENT)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/data/default_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/citlali/default_config.h"
    @ONLY
)

add_library(citlali STATIC)
add_library(citlali::citlali ALIAS citlali)
set_target_properties(citlali PROPERTIES EXPORT_NAME citlali)
tula_cmake_target_defaults(TARGET citlali)
set(CITLALI_PACKAGE_SPEC "" CACHE STRING "Concrete Spack package spec")
set(CITLALI_DAG_HASH "" CACHE STRING "Concrete Spack dependency DAG hash")
set(TOLTECA_BUILD_PROFILE "" CACHE STRING "tolteca_deploy profile identity")
set(TOLTECA_LOCK_SHA256 "" CACHE STRING "SHA-256 of the deployment Spack lock")
tula_cmake_generate_version_header(
    TARGET citlali
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/citlali/version.h"
    BUILD_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated"
    NAMESPACE "citlali::build"
    VERSION "${PROJECT_VERSION}"
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    PACKAGE_SPEC "${CITLALI_PACKAGE_SPEC}"
    DAG_HASH "${CITLALI_DAG_HASH}"
    BUILD_PROFILE "${TOLTECA_BUILD_PROFILE}"
    LOCK_SHA256 "${TOLTECA_LOCK_SHA256}"
)
target_sources(
    citlali
    PRIVATE
        src/citlali/core/engine/calib.cpp
        src/citlali/core/engine/telescope.cpp
        src/citlali/core/mapmaking/map.cpp
        src/citlali/core/timestream/ptc/sensitivity.cpp
        src/citlali/core/utils/gauss_models.cpp
)
target_include_directories(
    citlali
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(
    citlali
    PUBLIC
        kids::kids
        Ceres::ceres
        tula::core
        tula::logging
        tula::yaml
        tula::ecsv
        tula::eigen
        tula::netcdf
        tula::enum
        tula::perflibs
        tula::grppi
        Spectra::Spectra
        Boost::headers
        PkgConfig::FFTW3
        FFTW3::Threads
        tula_deps::ccfits
        citlali::config
)

option(CITLALI_BUILD_CLI "Build the Citlali data-reduction CLI" ON)
if(CITLALI_BUILD_CLI)
    add_executable(citlali_cli src/citlali/cli/main.cpp)
    set_target_properties(citlali_cli PROPERTIES OUTPUT_NAME citlali)
    tula_cmake_target_defaults(TARGET citlali_cli)
    target_link_libraries(
        citlali_cli
        PRIVATE
            citlali::citlali
            tula::cli
    )
    install(TARGETS citlali_cli RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    PATTERN "config.h.in" EXCLUDE
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/generated/citlali/config.h"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/citlali/default_config.h"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/citlali/version.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/citlali"
)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindFFTW3Threads.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/citlali"
)
tula_cmake_install_package(
    PACKAGE citlali
    EXPORT citlaliTargets
    NAMESPACE citlali::
    VERSION "${PROJECT_VERSION}"
    TARGETS citlali citlali_config
    CONFIG_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/citlaliConfig.cmake.in"
)

option(
    CITLALI_BUILD_TESTS
    "Build and run Citlali behavior and CLI tests"
    "${PROJECT_IS_TOP_LEVEL}"
)
if(CITLALI_BUILD_TESTS)
    set(
        TOLTECA_TEST_DATA_ROOT
        ""
        CACHE PATH
        "Path to the tolteca_test_data checkout used by real NetCDF tests"
    )
    if(NOT TOLTECA_TEST_DATA_ROOT)
        set(_tolteca_test_data_candidate
            "${CMAKE_CURRENT_SOURCE_DIR}/../tolteca_test_data")
        if(IS_DIRECTORY "${_tolteca_test_data_candidate}")
            cmake_path(
                ABSOLUTE_PATH _tolteca_test_data_candidate
                NORMALIZE
                OUTPUT_VARIABLE TOLTECA_TEST_DATA_ROOT
            )
        endif()
    endif()
    string(
        CONCAT _citlali_real_fixture
        "${TOLTECA_TEST_DATA_ROOT}/data_lmt/toltec/ics/toltec0/"
        "toltec0_018230_111_0000_2024_05_20_13_40_08_timestream.nc"
    )
    if(EXISTS "${_citlali_real_fixture}")
        set(CITLALI_HAS_REAL_TEST_DATA 1)
    else()
        set(CITLALI_HAS_REAL_TEST_DATA 0)
        message(
            STATUS
            "(citlali) real reader/solver test disabled: fixture not found"
        )
    endif()
    add_subdirectory(tests)
endif()
