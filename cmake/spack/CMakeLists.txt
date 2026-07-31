cmake_minimum_required(VERSION 3.25)

project(
    citlali
    VERSION 4.0.0
    DESCRIPTION "TolTEC data reduction pipeline engine"
    HOMEPAGE_URL "https://toltec-astro.github.io/citlali"
    LANGUAGES C CXX
)

include(CTest)
include(GNUInstallDirs)

get_filename_component(
    CITLALI_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." REALPATH
)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

find_package(TulaCMake 3 CONFIG REQUIRED)
find_package(kidscpp 3.1 CONFIG REQUIRED)
find_package(
    tula 3.1 CONFIG REQUIRED
    COMPONENTS core logging yaml ecsv eigen netcdf enum cli grppi
)
find_package(Ceres 2.2 CONFIG REQUIRED)
find_package(Boost 1.83 CONFIG REQUIRED)
find_package(spectra 1.0 CONFIG REQUIRED)
find_package(TulaPerflibs 0.1 CONFIG REQUIRED)
if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()
find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(ZLIB REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
find_package(FFTW3Threads REQUIRED)
pkg_check_modules(CCFITS REQUIRED IMPORTED_TARGET CCfits)
pkg_check_modules(CFITSIO REQUIRED IMPORTED_TARGET cfitsio)

if(TARGET hdf5::hdf5-shared AND TARGET hdf5::hdf5_hl-shared)
    set(CITLALI_HDF5_TARGETS hdf5::hdf5-shared hdf5::hdf5_hl-shared)
elseif(TARGET hdf5-shared AND TARGET hdf5_hl-shared)
    set(CITLALI_HDF5_TARGETS hdf5-shared hdf5_hl-shared)
else()
    set(CITLALI_HDF5_TARGETS ${HDF5_LIBRARIES})
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(
    CITLALI_USE_WIENER_FILTER_OMP
    "Use OpenMP Wiener filter implementation"
    ON
)
set(
    CITLALI_SPACK_DAG_HASH ""
    CACHE STRING "Concrete Spack DAG hash for build provenance"
)
if(NOT CITLALI_SPACK_DAG_HASH)
    message(FATAL_ERROR "CITLALI_SPACK_DAG_HASH is required")
endif()
set(CITLALI_BUILD_SYSTEM "spack")
if(CITLALI_USE_WIENER_FILTER_OMP)
    set(CITLALI_BUILD_VARIANTS "+wiener_openmp")
else()
    set(CITLALI_BUILD_VARIANTS "~wiener_openmp")
endif()

set(CITLALI_GENERATED_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY
    "${CITLALI_GENERATED_INCLUDE}/citlali_config"
    "${CITLALI_GENERATED_INCLUDE}/kidscpp_config"
    "${CITLALI_GENERATED_INCLUDE}/tula_config"
)

execute_process(
    COMMAND git rev-parse --short=9 HEAD
    WORKING_DIRECTORY "${CITLALI_SOURCE_ROOT}"
    OUTPUT_VARIABLE CITLALI_GIT_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
execute_process(
    COMMAND git describe --tags --always --dirty
    WORKING_DIRECTORY "${CITLALI_SOURCE_ROOT}"
    OUTPUT_VARIABLE CITLALI_GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT CITLALI_GIT_REVISION)
    set(CITLALI_GIT_REVISION "unknown")
endif()
if(NOT CITLALI_GIT_VERSION)
    set(CITLALI_GIT_VERSION "${PROJECT_VERSION}")
endif()
string(TIMESTAMP CITLALI_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%S" UTC)

configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/templates/config.h.in"
    "${CITLALI_GENERATED_INCLUDE}/citlali_config/config.h"
    @ONLY
)
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/templates/gitversion.h.in"
    "${CITLALI_GENERATED_INCLUDE}/citlali_config/gitversion.h"
    @ONLY
)
file(READ "${CITLALI_SOURCE_ROOT}/data/config.yaml"
     CITLALI_DEFAULT_CONFIG_CONTENT)
configure_file(
    "${CITLALI_SOURCE_ROOT}/data/default_config.h.in"
    "${CITLALI_GENERATED_INCLUDE}/citlali_config/default_config.h"
    @ONLY
)
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/templates/kidscpp_gitversion.h.in"
    "${CITLALI_GENERATED_INCLUDE}/kidscpp_config/gitversion.h"
    @ONLY
)
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/templates/tula_gitversion.h.in"
    "${CITLALI_GENERATED_INCLUDE}/tula_config/gitversion.h"
    @ONLY
)

add_library(citlali_config INTERFACE)
add_library(citlali::config ALIAS citlali_config)
set_target_properties(citlali_config PROPERTIES EXPORT_NAME config)
target_compile_definitions(
    citlali_config
    INTERFACE CITLALI_KIDSCPP_V3=1 CITLALI_TULA_V3=1
)
target_include_directories(
    citlali_config
    INTERFACE
        "$<BUILD_INTERFACE:${CITLALI_GENERATED_INCLUDE}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

add_library(citlali STATIC)
add_library(citlali::citlali ALIAS citlali)
set_target_properties(citlali PROPERTIES EXPORT_NAME citlali)
tula_cmake_target_defaults(TARGET citlali)
target_sources(
    citlali
    PRIVATE
        "${CITLALI_SOURCE_ROOT}/src/citlali/core/config/timestream_enums.cpp"
        "${CITLALI_SOURCE_ROOT}/src/citlali/core/engine/calib.cpp"
        "${CITLALI_SOURCE_ROOT}/src/citlali/core/engine/telescope.cpp"
        "${CITLALI_SOURCE_ROOT}/src/citlali/core/mapmaking/map.cpp"
        "${CITLALI_SOURCE_ROOT}/src/citlali/core/pipeline/output_root_lease.cpp"
        "${CITLALI_SOURCE_ROOT}/src/citlali/core/pipeline/reduction_restart_checkpoint.cpp"
        "${CITLALI_SOURCE_ROOT}/src/citlali/core/timestream/ptc/sensitivity.cpp"
        "${CITLALI_SOURCE_ROOT}/src/citlali/core/utils/gauss_models.cpp"
)
target_include_directories(
    citlali
    PUBLIC
        "$<BUILD_INTERFACE:${CITLALI_SOURCE_ROOT}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(
    citlali
    PUBLIC
        citlali::config
        kids::kids
        Ceres::ceres
        tula::core
        tula::logging
        tula::yaml
        tula::ecsv
        tula::eigen
        tula::netcdf
        tula::enum
        tula::cli
        tula::grppi
        Spectra::Spectra
        Boost::headers
        PkgConfig::FFTW3
        FFTW3::Threads
        PkgConfig::CCFITS
        PkgConfig::CFITSIO
        OpenMP::OpenMP_CXX
        ZLIB::ZLIB
        ${CITLALI_HDF5_TARGETS}
)

if(CITLALI_USE_WIENER_FILTER_OMP)
    target_compile_definitions(
        citlali PUBLIC CITLALI_USE_WIENER_FILTER_OMP=1
    )
endif()

option(CITLALI_BUILD_CLI "Build the Citlali reduction CLI" ON)
if(CITLALI_BUILD_CLI)
    add_executable(
        citlali_cli "${CITLALI_SOURCE_ROOT}/src/citlali/cli/main.cpp"
    )
    set_target_properties(citlali_cli PROPERTIES OUTPUT_NAME citlali)
    tula_cmake_target_defaults(TARGET citlali_cli)
    target_link_libraries(
        citlali_cli PRIVATE citlali::citlali tula::cli
    )
    install(TARGETS citlali_cli RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

option(CITLALI_BUILD_TESTS "Build the full Citlali test suite" OFF)
if(CITLALI_BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)
    enable_testing()
    add_subdirectory("${CITLALI_SOURCE_ROOT}/tests" tests)
    set_property(
        TARGET citlali_test citlali_safety_test
        PROPERTY EXCLUDE_FROM_ALL FALSE
    )
endif()

install(
    DIRECTORY "${CITLALI_SOURCE_ROOT}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(
    DIRECTORY "${CITLALI_GENERATED_INCLUDE}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(
    FILES "${CMAKE_CURRENT_LIST_DIR}/FindFFTW3Threads.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/citlali"
)
tula_cmake_install_package(
    PACKAGE citlali
    EXPORT citlaliTargets
    NAMESPACE citlali::
    VERSION "${PROJECT_VERSION}"
    TARGETS citlali citlali_config
    CONFIG_TEMPLATE "${CMAKE_CURRENT_LIST_DIR}/citlaliConfig.cmake.in"
)
