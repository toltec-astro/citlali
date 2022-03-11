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
  GIT_TAG        "v0.1.x"
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

add_library(citlali STATIC)
target_sources(citlali
    PRIVATE
    "src/citlali/core/gaussfit.cpp"
    "src/citlali/core/sensitivity.cpp"
    )
target_include_directories(citlali PUBLIC "include")
target_link_libraries(citlali PUBLIC ${dep_libs})
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
endif()

# Create concfig headers
generate_gitversion_header(${PROJECT_NAME})
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
