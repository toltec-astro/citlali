cmake_minimum_required(VERSION 3.20)
project(citlali
    DESCRIPTION
        "TolTEC data reduction pipeline engine"
    HOMEPAGE_URL
        "https://toltec-astro.github.io/citlali"
    )

# Fetch kidscpp, which fetches tula
include(FetchContent)
FetchContent_Declare(
  kidscpp
  GIT_REPOSITORY https://github.com/toltec-astro/kidscpp.git
  GIT_TAG        "v1.x"
)
FetchContent_MakeAvailable(kidscpp)

# Add the tula cmake modules
set(CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}/" ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "${tula_SOURCE_DIR}/tula_cmake/cmake" ${CMAKE_MODULE_PATH})
include(common_settings)

# Check if this project is embedded as sub project.
if(NOT DEFINED CITLALI_STANDALONE)
    if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(CITLALI_STANDALONE ON)
    else()
        set(CITLALI_STANDALONE OFF)
    endif()
endif()

# Add dependencies
set(dep_libs kids::kids)

include(Spectra)
set(dep_libs ${dep_libs} tula::Spectra)

include(Boost)
set(dep_libs ${dep_libs} tula::Boost)

include(CCfits)
set(dep_libs ${dep_libs} tula::CCfits)

#include(Matplotlibcpp)
#set(dep_libs ${dep_libs} tula::Matplotlibcpp)

include(FFTW)
set(dep_libs ${dep_libs} tula::FFTW)

find_package(ZLIB REQUIRED)
if(TARGET ZLIB::ZLIB)
    set(dep_libs ${dep_libs} ZLIB::ZLIB)
else()
    set(dep_libs ${dep_libs} ${ZLIB_LIBRARIES})
endif()

find_package(HDF5 REQUIRED COMPONENTS C HL)
if(TARGET hdf5::hdf5-shared AND TARGET hdf5::hdf5_hl-shared)
    set(dep_libs ${dep_libs} hdf5::hdf5-shared hdf5::hdf5_hl-shared)
elseif(TARGET hdf5-shared AND TARGET hdf5_hl-shared)
    set(dep_libs ${dep_libs} hdf5-shared hdf5_hl-shared)
else()
    set(dep_libs ${dep_libs} ${HDF5_LIBRARIES})
endif()

add_library(citlali STATIC)
target_sources(citlali
    PRIVATE
    "src/citlali/core/config/timestream_enums.cpp"
    "src/citlali/core/pipeline/output_root_lease.cpp"
    "src/citlali/core/pipeline/reduction_restart_checkpoint.cpp"
    "src/citlali/core/engine/calib.cpp"
    "src/citlali/core/engine/telescope.cpp"
    "src/citlali/core/utils/gauss_models.cpp"
    "src/citlali/core/timestream/ptc/sensitivity.cpp"
    "src/citlali/core/mapmaking/map.cpp"
    )
target_include_directories(citlali PUBLIC "include")
target_link_libraries(citlali PUBLIC ${dep_libs})

option(CITLALI_USE_WIENER_FILTER_OMP "Use OpenMP Wiener filter implementation" OFF)
if (CITLALI_USE_WIENER_FILTER_OMP)
    target_compile_definitions(citlali PUBLIC CITLALI_USE_WIENER_FILTER_OMP=1)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(citlali PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP not found; Wiener filter OMP path will compile without OpenMP parallelization.")
    endif()
endif()
add_library(citlali::citlali ALIAS citlali)

## optional targets
option(CITLALI_BUILD_CLI "Build CLI" ON)
if (CITLALI_BUILD_CLI)

    set(dep_libs citlali::citlali)

    include(Clipp)
    set(dep_libs ${dep_libs} tula::Clipp)

    add_executable(citlali_cli)
    target_sources(citlali_cli
        PRIVATE
            "src/citlali/cli/canonical_apt_contract_protocol_v1.cpp"
            "src/citlali/cli/canonical_apt_contract_protocol_v2.cpp"
            "src/citlali/cli/main.cpp"
            )
    target_link_libraries(citlali_cli PRIVATE ${dep_libs})
    target_include_directories(citlali_cli PUBLIC "include")
    set_property(TARGET citlali_cli PROPERTY RULE_LAUNCH_COMPILE "\"${CMAKE_COMMAND}\" -E time")
    set_property(TARGET citlali_cli PROPERTY OUTPUT_NAME citlali)
endif()

if (CITLALI_STANDALONE)
    # Tests, examples, and docs
    option(CITLALI_BUILD_TESTS "Build tests" ON)
    if (CITLALI_BUILD_TESTS)
        enable_testing()
        add_subdirectory(tests)
    endif()

    option(CITLALI_BUILD_EXAMPLES "Build example programs" OFF)
    if (CITLALI_BUILD_EXAMPLES)
        add_subdirectory(examples)
    endif()

    option(CITLALI_BUILD_DOC "Build docs" OFF)
    if (CITLALI_BUILD_DOC)
        include(docxgen_target)
    endif()

    # This is an explicit, opt-in representative-data gate.  It is not part
    # of the production CLI and deliberately publishes no persistent TOD.
    add_executable(citlali_wp7_identity_rtc_acceptance EXCLUDE_FROM_ALL
        "tools/wp7/identity_rtc_acceptance.cpp"
    )
    set(wp7_dependency_identity_dir
        "${CMAKE_BINARY_DIR}/generated")
    set(wp7_dependency_identity_header
        "${wp7_dependency_identity_dir}/citlali_wp7/acceptance_build_identity.h")
    add_custom_target(citlali_wp7_identity_rtc_acceptance_build_identity
        COMMAND "${CMAKE_COMMAND}"
            "-DWP7_CITLALI_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DWP7_CITLALI_BINARY_DIR=${CMAKE_BINARY_DIR}"
            "-DWP7_KIDSCPP_SOURCE_DIR=${kidscpp_SOURCE_DIR}"
            "-DWP7_KIDSCPP_REVISION=04088da182622c3e879f04314974a7c0d60ee2d6"
            "-DWP7_KIDSCPP_PATCH=${CMAKE_CURRENT_SOURCE_DIR}/patches/local/kidscpp-local-build.patch"
            "-DWP7_TULA_SOURCE_DIR=${tula_SOURCE_DIR}"
            "-DWP7_TULA_REVISION=f30f81d97c44bd79618273bb842302ef839c6ab1"
            "-DWP7_TULA_PATCH=${CMAKE_CURRENT_SOURCE_DIR}/patches/local/tula-local-build.patch"
            "-DWP7_DEPENDENCY_IDENTITY_HEADER=${wp7_dependency_identity_header}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_wp7_dependency_state.cmake"
        COMMENT "Verify exact WP-7 acceptance dependency state"
        VERBATIM
    )
    add_dependencies(citlali_wp7_identity_rtc_acceptance
        citlali_wp7_identity_rtc_acceptance_build_identity)
    target_link_libraries(citlali_wp7_identity_rtc_acceptance
        PRIVATE citlali::citlali
    )
    target_include_directories(citlali_wp7_identity_rtc_acceptance
        PRIVATE "include" "${wp7_dependency_identity_dir}"
    )
    set_target_properties(citlali_wp7_identity_rtc_acceptance
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# Create config headers
generate_gitversion_header(${PROJECT_NAME})
include("${CMAKE_CURRENT_LIST_DIR}/cmake/citlali_gitversion.cmake")
get_config_header_output_dir(output_dir ${PROJECT_NAME})
citlali_enable_buildtime_gitversion_refresh(
    gitversion_header_${PROJECT_NAME}
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${output_dir}/gitversion.h"
)
generate_config_header(${PROJECT_NAME})

# create default citlali config file header
set(citlali_data_dir ${CMAKE_CURRENT_LIST_DIR}/data)
get_config_header_output_dir(output_dir citlali)
set(output_path ${output_dir}/default_config.h)
verbose_message("Generate default config header ${output_path}")
file(READ ${citlali_data_dir}/config.yaml CITLALI_DEFAULT_CONFIG_CONTENT)
configure_file(${citlali_data_dir}/default_config.h.in ${output_path} @ONLY)
add_custom_target(default_config_header_citlali DEPENDS
    ${output_path}
)
add_dependencies(citlali default_config_header_citlali)
