cmake_minimum_required(VERSION 3.20)

foreach(required_variable TEST_ROOT REFRESH_MODULE)
    if(NOT DEFINED "${required_variable}"
       OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR
            "Missing required variable: ${required_variable}")
    endif()
endforeach()

find_package(Git REQUIRED)

set(source_dir "${TEST_ROOT}/source")
set(build_dir "${TEST_ROOT}/build")
file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${source_dir}")

file(WRITE "${source_dir}/payload.txt" "first\n")
file(WRITE "${source_dir}/main.cpp" [=[
#include <gitversion.h>
#include <iostream>

int main() {
    std::cout << CITLALI_GIT_REVISION << "\n";
}
]=])
file(WRITE "${source_dir}/CMakeLists.txt" "
cmake_minimum_required(VERSION 3.20)
project(gitversion_refresh_probe LANGUAGES CXX)
include(\"${REFRESH_MODULE}\")
set(generated_dir \"\${CMAKE_BINARY_DIR}/generated\")
file(MAKE_DIRECTORY \"\${generated_dir}\")
file(WRITE \"\${generated_dir}/gitversion.h\"
    \"#pragma once\\n\\n\"
    \"#define CITLALI_GIT_REVISION \\\"stale\\\"\\n\"
    \"#define CITLALI_GIT_VERSION \\\"stale\\\"\\n\"
    \"#define CITLALI_BUILD_TIMESTAMP \\\"stale\\\"\\n\")
add_custom_target(gitversion_header_probe)
citlali_enable_buildtime_gitversion_refresh(
    gitversion_header_probe
    \"\${CMAKE_SOURCE_DIR}\"
    \"\${generated_dir}/gitversion.h\")
add_executable(gitversion_probe main.cpp)
target_include_directories(gitversion_probe PRIVATE \"\${generated_dir}\")
add_dependencies(gitversion_probe gitversion_header_probe)
")

function(run_checked)
    execute_process(
        COMMAND ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "Command failed (${result}): ${ARGN}\n${output}\n${error}")
    endif()
endfunction()

run_checked("${GIT_EXECUTABLE}" init "${source_dir}")
run_checked("${GIT_EXECUTABLE}" -C "${source_dir}"
    add CMakeLists.txt main.cpp payload.txt)
run_checked("${GIT_EXECUTABLE}" -C "${source_dir}"
    -c user.name=Citlali-Test
    -c user.email=citlali-test@example.invalid
    commit -m first)
run_checked("${CMAKE_COMMAND}" -S "${source_dir}" -B "${build_dir}")
run_checked("${CMAKE_COMMAND}" --build "${build_dir}"
    --target gitversion_probe)

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse --short HEAD
    OUTPUT_VARIABLE first_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND "${build_dir}/gitversion_probe"
    OUTPUT_VARIABLE first_reported_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
if(NOT first_reported_revision STREQUAL first_revision)
    message(FATAL_ERROR
        "Initial executable revision mismatch: "
        "expected ${first_revision}, got ${first_reported_revision}")
endif()

set(header "${build_dir}/generated/gitversion.h")
file(TIMESTAMP "${header}" unchanged_timestamp_before "%s")
run_checked("${CMAKE_COMMAND}" -E sleep 1)
run_checked("${CMAKE_COMMAND}" --build "${build_dir}"
    --target gitversion_probe)
file(TIMESTAMP "${header}" unchanged_timestamp_after "%s")
if(NOT unchanged_timestamp_after STREQUAL unchanged_timestamp_before)
    message(FATAL_ERROR
        "No-op build rewrote the unchanged Git-version header")
endif()

file(APPEND "${source_dir}/payload.txt" "second\n")
run_checked("${GIT_EXECUTABLE}" -C "${source_dir}" add payload.txt)
run_checked("${GIT_EXECUTABLE}" -C "${source_dir}"
    -c user.name=Citlali-Test
    -c user.email=citlali-test@example.invalid
    commit -m second)
run_checked("${CMAKE_COMMAND}" --build "${build_dir}"
    --target gitversion_probe)

execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse --short HEAD
    OUTPUT_VARIABLE second_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND "${build_dir}/gitversion_probe"
    OUTPUT_VARIABLE second_reported_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
if(first_revision STREQUAL second_revision)
    message(FATAL_ERROR "Test repository revision did not advance")
endif()
if(NOT second_reported_revision STREQUAL second_revision)
    message(FATAL_ERROR
        "Build-only executable revision mismatch after HEAD advance: "
        "expected ${second_revision}, got ${second_reported_revision}")
endif()
