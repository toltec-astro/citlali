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
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE CITLALI_GIT_COMMIT_FULL_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE CITLALI_GIT_COMMIT_FULL_RESULT)
string(LENGTH "${CITLALI_GIT_COMMIT_FULL_VALUE}"
       CITLALI_GIT_COMMIT_FULL_LENGTH)
if(NOT CITLALI_GIT_COMMIT_FULL_RESULT EQUAL 0 OR
   NOT CITLALI_GIT_COMMIT_FULL_LENGTH EQUAL 40 OR
   NOT CITLALI_GIT_COMMIT_FULL_VALUE MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "Citlali requires an exact full 40-hex Git commit identity")
endif()
execute_process(
    COMMAND git status --porcelain --untracked-files=no
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE CITLALI_GIT_DIRTY_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE CITLALI_GIT_DIRTY_RESULT)
if(NOT CITLALI_GIT_DIRTY_RESULT EQUAL 0)
    message(FATAL_ERROR "Citlali could not determine the Git worktree state")
endif()
if(CITLALI_GIT_DIRTY_OUTPUT STREQUAL "")
    set(CITLALI_GIT_WORKTREE_DIRTY_VALUE 0)
else()
    set(CITLALI_GIT_WORKTREE_DIRTY_VALUE 1)
endif()
target_compile_definitions(citlali PUBLIC
    CITLALI_GIT_COMMIT_FULL="${CITLALI_GIT_COMMIT_FULL_VALUE}"
    CITLALI_GIT_WORKTREE_DIRTY=${CITLALI_GIT_WORKTREE_DIRTY_VALUE})

# Re-evaluate the source identity on every ordinary build.  The configured
# constants remain the compile-time identity, while this target makes a stale
# clean identity a hard build failure after any ref/index/worktree transition.
set(CITLALI_SOURCE_IDENTITY_CHECK_SCRIPT
    "${CMAKE_CURRENT_BINARY_DIR}/citlali_source_identity_check.cmake")
file(WRITE "${CITLALI_SOURCE_IDENTITY_CHECK_SCRIPT}" [=[
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${CITLALI_SOURCE_DIR}"
    OUTPUT_VARIABLE current_commit
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE commit_result)
execute_process(
    COMMAND git status --porcelain --untracked-files=no
    WORKING_DIRECTORY "${CITLALI_SOURCE_DIR}"
    OUTPUT_VARIABLE dirty_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE dirty_result)
if(NOT commit_result EQUAL 0 OR NOT dirty_result EQUAL 0)
    message(FATAL_ERROR "Citlali could not refresh the ordinary-build source identity")
endif()
string(LENGTH "${current_commit}" current_commit_length)
if(NOT current_commit_length EQUAL 40 OR
   NOT current_commit MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "Citlali ordinary build resolved a non-40-hex source identity")
endif()
if(dirty_output STREQUAL "")
    set(current_dirty 0)
else()
    set(current_dirty 1)
endif()
if(NOT current_commit STREQUAL CITLALI_CONFIGURED_COMMIT OR
   NOT current_dirty EQUAL CITLALI_CONFIGURED_DIRTY)
    message(FATAL_ERROR
        "Citlali source identity changed after configure; rerun CMake before building")
endif()
]=])
add_custom_target(citlali_source_identity_check ALL
    COMMAND "${CMAKE_COMMAND}"
        -DCITLALI_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DCITLALI_CONFIGURED_COMMIT=${CITLALI_GIT_COMMIT_FULL_VALUE}
        -DCITLALI_CONFIGURED_DIRTY=${CITLALI_GIT_WORKTREE_DIRTY_VALUE}
        -P "${CITLALI_SOURCE_IDENTITY_CHECK_SCRIPT}"
    VERBATIM)
add_dependencies(citlali citlali_source_identity_check)
target_sources(citlali
    PRIVATE
    "src/citlali/core/config/timestream_enums.cpp"
    "src/citlali/core/pipeline/output_root_lease.cpp"
    "src/citlali/core/pipeline/reduction_restart_checkpoint.cpp"
    #"src/citlali/core/engine/todproc.cpp"
    #"src/citlali/core/engine/kidsproc.cpp"
    "src/citlali/core/engine/calib.cpp"
    "src/citlali/core/engine/telescope.cpp"
    #"src/citlali/core/engine/engine.cpp"
    "src/citlali/core/utils/gauss_models.cpp"
    "src/citlali/core/timestream/ptc/sensitivity.cpp"
    "src/citlali/core/mapmaking/map.cpp"
    #"src/citlali/core/mapmaking/wiener_filter.cpp"
    #"src/citlali/core/engine/lali.cpp"
    #"src/citlali/core/engine/pointing.cpp"
    #"src/citlali/core/engine/beammap.cpp"
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
