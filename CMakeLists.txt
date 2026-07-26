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
)
add_library(citlali::citlali ALIAS citlali)

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
