cmake_minimum_required(VERSION 3.20)

foreach(required_variable TEST_ROOT VERIFY_SCRIPT)
    if(NOT DEFINED "${required_variable}"
       OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

find_package(Git REQUIRED)
file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")

function(run_checked)
    execute_process(
        COMMAND ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "Command failed (${result}): ${ARGN}\n${output}\n${error}")
    endif()
endfunction()

function(make_dependency name revision_variable patch_variable)
    set(source_dir "${TEST_ROOT}/${name}")
    set(patch "${TEST_ROOT}/${name}.patch")
    file(MAKE_DIRECTORY "${source_dir}")
    file(WRITE "${source_dir}/payload.txt" "base\n")
    run_checked("${GIT_EXECUTABLE}" init "${source_dir}")
    if(ARGC GREATER 3)
        run_checked("${GIT_EXECUTABLE}"
            -c protocol.file.allow=always
            -C "${source_dir}" submodule add "${ARGV3}" tula_cmake)
    endif()
    run_checked("${GIT_EXECUTABLE}" -C "${source_dir}" add -A)
    run_checked("${GIT_EXECUTABLE}" -C "${source_dir}"
        -c user.name=Citlali-Test
        -c user.email=citlali-test@example.invalid
        commit -m base)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" rev-parse HEAD
        OUTPUT_VARIABLE revision
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    file(APPEND "${source_dir}/payload.txt" "approved\n")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" diff --binary
        OUTPUT_FILE "${patch}"
        COMMAND_ERROR_IS_FATAL ANY)
    set("${revision_variable}" "${revision}" PARENT_SCOPE)
    set("${patch_variable}" "${patch}" PARENT_SCOPE)
endfunction()

set(nested_source "${TEST_ROOT}/nested-source")
file(MAKE_DIRECTORY "${nested_source}")
file(WRITE "${nested_source}/nested.txt" "nested base\n")
run_checked("${GIT_EXECUTABLE}" init "${nested_source}")
run_checked("${GIT_EXECUTABLE}" -C "${nested_source}" add nested.txt)
run_checked("${GIT_EXECUTABLE}" -C "${nested_source}"
    -c user.name=Citlali-Test
    -c user.email=citlali-test@example.invalid
    commit -m nested-base)

make_dependency(kidscpp kidscpp_revision kidscpp_patch)
make_dependency(tula tula_revision tula_patch "${nested_source}")
set(citlali_source "${TEST_ROOT}/citlali")
file(MAKE_DIRECTORY "${citlali_source}")
file(WRITE "${citlali_source}/source.txt" "clean\n")
run_checked("${GIT_EXECUTABLE}" init "${citlali_source}")
run_checked("${GIT_EXECUTABLE}" -C "${citlali_source}" add source.txt)
run_checked("${GIT_EXECUTABLE}" -C "${citlali_source}"
    -c user.name=Citlali-Test
    -c user.email=citlali-test@example.invalid
    commit -m source)
set(identity_header "${TEST_ROOT}/generated/identity.h")
set(verify_command
    "${CMAKE_COMMAND}"
    "-DWP7_CITLALI_SOURCE_DIR=${citlali_source}"
    "-DWP7_KIDSCPP_SOURCE_DIR=${TEST_ROOT}/kidscpp"
    "-DWP7_KIDSCPP_REVISION=${kidscpp_revision}"
    "-DWP7_KIDSCPP_PATCH=${kidscpp_patch}"
    "-DWP7_TULA_SOURCE_DIR=${TEST_ROOT}/tula"
    "-DWP7_TULA_REVISION=${tula_revision}"
    "-DWP7_TULA_PATCH=${tula_patch}"
    "-DWP7_DEPENDENCY_IDENTITY_HEADER=${identity_header}"
    -P "${VERIFY_SCRIPT}")

run_checked(${verify_command})
file(READ "${identity_header}" identity)
if(NOT identity MATCHES "CITLALI_WP7_DEPENDENCY_STATE_VERIFIED 1")
    message(FATAL_ERROR "Verified dependency identity header is incomplete")
endif()

file(APPEND "${citlali_source}/source.txt" "dirty\n")
execute_process(
    COMMAND ${verify_command}
    RESULT_VARIABLE dirty_source_result
    OUTPUT_QUIET ERROR_QUIET)
if(dirty_source_result EQUAL 0)
    message(FATAL_ERROR "Dirty Citlali source content was accepted")
endif()
file(WRITE "${citlali_source}/source.txt" "clean\n")

file(WRITE "${TEST_ROOT}/kidscpp/untracked.txt" "unapproved\n")
execute_process(
    COMMAND ${verify_command}
    RESULT_VARIABLE untracked_result
    OUTPUT_QUIET ERROR_QUIET)
if(untracked_result EQUAL 0)
    message(FATAL_ERROR "Untracked dependency content was accepted")
endif()
file(REMOVE "${TEST_ROOT}/kidscpp/untracked.txt")

file(APPEND "${TEST_ROOT}/kidscpp/.git/info/exclude" "ignored.txt\n")
file(WRITE "${TEST_ROOT}/kidscpp/ignored.txt" "unapproved ignored input\n")
execute_process(
    COMMAND ${verify_command}
    RESULT_VARIABLE ignored_result
    OUTPUT_QUIET ERROR_QUIET)
if(ignored_result EQUAL 0)
    message(FATAL_ERROR "Ignored dependency content was accepted")
endif()
file(REMOVE "${TEST_ROOT}/kidscpp/ignored.txt")

file(WRITE "${TEST_ROOT}/tula/staged-addition.txt" "unapproved staged input\n")
run_checked("${GIT_EXECUTABLE}" -C "${TEST_ROOT}/tula"
    add staged-addition.txt)
execute_process(
    COMMAND ${verify_command}
    RESULT_VARIABLE staged_addition_result
    OUTPUT_QUIET ERROR_QUIET)
if(staged_addition_result EQUAL 0)
    message(FATAL_ERROR "Staged dependency addition was accepted")
endif()
run_checked("${GIT_EXECUTABLE}" -C "${TEST_ROOT}/tula"
    rm --cached -- staged-addition.txt)
file(REMOVE "${TEST_ROOT}/tula/staged-addition.txt")

file(APPEND "${TEST_ROOT}/tula/payload.txt" "extra\n")
execute_process(
    COMMAND ${verify_command}
    RESULT_VARIABLE extra_diff_result
    OUTPUT_QUIET ERROR_QUIET)
if(extra_diff_result EQUAL 0)
    message(FATAL_ERROR "Unapproved tracked dependency content was accepted")
endif()
file(WRITE "${TEST_ROOT}/tula/payload.txt" "base\napproved\n")

file(APPEND "${TEST_ROOT}/tula/tula_cmake/nested.txt" "dirty\n")
execute_process(
    COMMAND ${verify_command}
    RESULT_VARIABLE dirty_submodule_result
    OUTPUT_QUIET ERROR_QUIET)
if(dirty_submodule_result EQUAL 0)
    message(FATAL_ERROR "Dirty dependency submodule content was accepted")
endif()
