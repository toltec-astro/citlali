cmake_minimum_required(VERSION 3.20)

foreach(required_variable
        CITLALI_GIT_SOURCE_DIR
        CITLALI_GITVERSION_HEADER)
    if(NOT DEFINED "${required_variable}"
       OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR
            "Missing required variable: ${required_variable}")
    endif()
endforeach()

find_program(git_executable NAMES git REQUIRED)

execute_process(
    COMMAND "${git_executable}" rev-parse --short HEAD
    WORKING_DIRECTORY "${CITLALI_GIT_SOURCE_DIR}"
    RESULT_VARIABLE revision_result
    OUTPUT_VARIABLE revision
    ERROR_VARIABLE revision_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT revision_result EQUAL 0 OR revision STREQUAL "")
    message(FATAL_ERROR
        "Unable to resolve Citlali Git revision: ${revision_error}")
endif()

execute_process(
    COMMAND "${git_executable}" describe --tags --always --broken
    WORKING_DIRECTORY "${CITLALI_GIT_SOURCE_DIR}"
    RESULT_VARIABLE version_result
    OUTPUT_VARIABLE version
    ERROR_VARIABLE version_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT version_result EQUAL 0 OR version STREQUAL "")
    message(FATAL_ERROR
        "Unable to resolve Citlali Git version: ${version_error}")
endif()

set(identity_current FALSE)
if(EXISTS "${CITLALI_GITVERSION_HEADER}")
    file(READ "${CITLALI_GITVERSION_HEADER}" existing_header)
    string(REGEX MATCH
        "#define CITLALI_GIT_REVISION \"([^\"]*)\""
        revision_match "${existing_header}")
    set(existing_revision "${CMAKE_MATCH_1}")
    string(REGEX MATCH
        "#define CITLALI_GIT_VERSION \"([^\"]*)\""
        version_match "${existing_header}")
    set(existing_version "${CMAKE_MATCH_1}")
    if(existing_revision STREQUAL revision
       AND existing_version STREQUAL version)
        set(identity_current TRUE)
    endif()
endif()

if(identity_current)
    return()
endif()

string(TIMESTAMP build_timestamp)
get_filename_component(output_dir "${CITLALI_GITVERSION_HEADER}" DIRECTORY)
file(MAKE_DIRECTORY "${output_dir}")
set(temporary_header "${CITLALI_GITVERSION_HEADER}.tmp")
file(WRITE "${temporary_header}"
    "#pragma once\n\n"
    "#define CITLALI_GIT_REVISION \"${revision}\"\n"
    "#define CITLALI_GIT_VERSION \"${version}\"\n"
    "#define CITLALI_BUILD_TIMESTAMP \"${build_timestamp}\"\n")
file(RENAME "${temporary_header}" "${CITLALI_GITVERSION_HEADER}")
