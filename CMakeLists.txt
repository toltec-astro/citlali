cmake_minimum_required(VERSION 3.25)

project(
    citlali
    VERSION 4.0.0
    DESCRIPTION "TolTEC data reduction pipeline engine"
    HOMEPAGE_URL "https://toltec-astro.github.io/citlali"
    LANGUAGES CXX
)

include(TulaProject)
tula_resolve_features()
find_package(kidscpp CONFIG REQUIRED)
tula_add_config_header(
    TARGET citlali_config
    ALIAS citlali::config
    PREFIX CITLALI
    OUTPUT citlali/config.h
)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/data/config.yaml" CITLALI_DEFAULT_CONFIG_CONTENT)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/data/default_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/citlali/default_config.h"
    @ONLY
)

add_library(citlali STATIC)
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
        "$<INSTALL_INTERFACE:include>"
)
target_compile_features(citlali PUBLIC cxx_std_23)
target_link_libraries(
    citlali
    PUBLIC
        kids::kids
        tula::logging
        tula::yaml_cpp
        tula::csv_parser
        tula::netcdf_cxx4
        tula::bitmask
        tula::meta_enum
        tula::perflibs
        tula::eigen
        tula::grppi
        tula::spectra
        tula::boost
        tula::fftw
        tula::ccfits
        tula::ceres
        citlali::config
)
add_library(citlali::citlali ALIAS citlali)

option(CITLALI_BUILD_CLI "Build the Citlali data-reduction CLI" ON)
if(CITLALI_BUILD_CLI)
    add_executable(citlali_cli src/citlali/cli/main.cpp)
    set_target_properties(citlali_cli PROPERTIES OUTPUT_NAME citlali)
    target_link_libraries(
        citlali_cli
        PRIVATE
            citlali::citlali
            tula::clipp
    )
    install(TARGETS citlali_cli RUNTIME DESTINATION bin)
endif()

option(CITLALI_BUILD_TESTS "Build Citlali behavior tests" "${PROJECT_IS_TOP_LEVEL}")
if(CITLALI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(
    TARGETS citlali
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/generated/citlali/config.h"
        "${CMAKE_CURRENT_BINARY_DIR}/generated/citlali/default_config.h"
    DESTINATION include/citlali
)
