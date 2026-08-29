cmake_minimum_required(VERSION 3.20)

foreach(required_variable
        WP7_CITLALI_SOURCE_DIR
        WP7_KIDSCPP_SOURCE_DIR
        WP7_KIDSCPP_REVISION
        WP7_KIDSCPP_PATCH
        WP7_TULA_SOURCE_DIR
        WP7_TULA_REVISION
        WP7_TULA_PATCH
        WP7_DEPENDENCY_IDENTITY_HEADER)
    if(NOT DEFINED "${required_variable}"
       OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "Missing required variable: ${required_variable}")
    endif()
endforeach()

find_program(git_executable NAMES git REQUIRED)

get_filename_component(output_dir
    "${WP7_DEPENDENCY_IDENTITY_HEADER}" DIRECTORY)
file(MAKE_DIRECTORY "${output_dir}")
set(state_dir "${output_dir}/dependency-state")
file(MAKE_DIRECTORY "${state_dir}")

execute_process(
    COMMAND "${git_executable}" -C "${WP7_CITLALI_SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE citlali_revision_result
    OUTPUT_VARIABLE citlali_revision
    ERROR_VARIABLE citlali_revision_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT citlali_revision_result EQUAL 0 OR citlali_revision STREQUAL "")
    message(FATAL_ERROR
        "Unable to resolve the WP-7 Citlali source revision: "
        "${citlali_revision_error}")
endif()
execute_process(
    COMMAND "${git_executable}" -C "${WP7_CITLALI_SOURCE_DIR}"
        status --porcelain=v1 --untracked-files=all
    RESULT_VARIABLE citlali_status_result
    OUTPUT_VARIABLE citlali_status
    ERROR_VARIABLE citlali_status_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT citlali_status_result EQUAL 0 OR NOT citlali_status STREQUAL "")
    message(FATAL_ERROR
        "WP-7 acceptance requires a clean Citlali source worktree: "
        "${citlali_status} ${citlali_status_error}")
endif()

function(verify_dependency label source_dir approved_revision patch_path
         revision_variable patch_sha_variable tree_variable)
    if(NOT IS_DIRECTORY "${source_dir}/.git" AND
       NOT EXISTS "${source_dir}/.git")
        message(FATAL_ERROR "${label} source is not a Git worktree: ${source_dir}")
    endif()
    if(NOT EXISTS "${patch_path}")
        message(FATAL_ERROR "${label} approved patch is absent: ${patch_path}")
    endif()

    execute_process(
        COMMAND "${git_executable}" -C "${source_dir}" rev-parse HEAD
        RESULT_VARIABLE revision_result
        OUTPUT_VARIABLE revision
        ERROR_VARIABLE revision_error
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT revision_result EQUAL 0 OR
       NOT revision STREQUAL approved_revision)
        message(FATAL_ERROR
            "${label} base revision is not approved: ${revision} ${revision_error}")
    endif()

    execute_process(
        COMMAND "${git_executable}" -C "${source_dir}"
            ls-files --others --exclude-standard
        RESULT_VARIABLE untracked_result
        OUTPUT_VARIABLE untracked
        ERROR_VARIABLE untracked_error
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT untracked_result EQUAL 0 OR NOT untracked STREQUAL "")
        message(FATAL_ERROR
            "${label} dependency worktree contains unapproved untracked content: "
            "${untracked} ${untracked_error}")
    endif()
    execute_process(
        COMMAND "${git_executable}" -C "${source_dir}"
            ls-files --others --ignored --exclude-standard
        RESULT_VARIABLE ignored_result
        OUTPUT_VARIABLE ignored
        ERROR_VARIABLE ignored_error
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT ignored_result EQUAL 0 OR NOT ignored STREQUAL "")
        message(FATAL_ERROR
            "${label} dependency worktree contains unapproved ignored content: "
            "${ignored} ${ignored_error}")
    endif()

    string(TOLOWER "${label}" label_lower)
    set(expected_index "${state_dir}/${label_lower}-expected.index")
    set(actual_index "${state_dir}/${label_lower}-actual.index")
    file(REMOVE "${expected_index}" "${actual_index}")

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "GIT_INDEX_FILE=${expected_index}"
            "${git_executable}" -C "${source_dir}" read-tree "${revision}"
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "GIT_INDEX_FILE=${expected_index}"
            "${git_executable}" -C "${source_dir}" apply
                --cached --whitespace=nowarn "${patch_path}"
        RESULT_VARIABLE apply_result
        OUTPUT_VARIABLE apply_output
        ERROR_VARIABLE apply_error)
    if(NOT apply_result EQUAL 0)
        message(FATAL_ERROR
            "${label} approved patch does not apply to its approved base: "
            "${apply_output} ${apply_error}")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "GIT_INDEX_FILE=${expected_index}"
            "${git_executable}" -C "${source_dir}" write-tree
        OUTPUT_VARIABLE expected_tree
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "GIT_INDEX_FILE=${actual_index}"
            "${git_executable}" -C "${source_dir}" read-tree "${revision}"
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "GIT_INDEX_FILE=${actual_index}"
            "${git_executable}" -C "${source_dir}" add -A -- .
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "GIT_INDEX_FILE=${actual_index}"
            "${git_executable}" -C "${source_dir}" write-tree
        OUTPUT_VARIABLE actual_tree
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    if(NOT actual_tree STREQUAL expected_tree)
        message(FATAL_ERROR
            "${label} dependency worktree is not exactly its approved base plus patch: "
            "expected tree ${expected_tree}, actual tree ${actual_tree}")
    endif()

    execute_process(
        COMMAND "${git_executable}" -C "${source_dir}"
            submodule status --recursive
        RESULT_VARIABLE submodule_status_result
        OUTPUT_VARIABLE submodule_status
        ERROR_VARIABLE submodule_status_error
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT submodule_status_result EQUAL 0 OR
       submodule_status MATCHES "(^|\n)[-+U]")
        message(FATAL_ERROR
            "${label} dependency submodule identity is incomplete: "
            "${submodule_status} ${submodule_status_error}")
    endif()
    execute_process(
        COMMAND "${git_executable}" -C "${source_dir}"
            submodule foreach --recursive --quiet "echo \"$displaypath\""
        RESULT_VARIABLE submodule_paths_result
        OUTPUT_VARIABLE submodule_paths_output
        ERROR_VARIABLE submodule_paths_error
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT submodule_paths_result EQUAL 0)
        message(FATAL_ERROR
            "${label} dependency submodule inventory failed: "
            "${submodule_paths_error}")
    endif()
    string(REPLACE "\n" ";" submodule_paths "${submodule_paths_output}")
    foreach(submodule_path IN LISTS submodule_paths)
        if(submodule_path STREQUAL "")
            continue()
        endif()
        execute_process(
            COMMAND "${git_executable}"
                -C "${source_dir}/${submodule_path}"
                status --porcelain=v1 --untracked-files=all --ignored=matching
            RESULT_VARIABLE submodule_dirty_result
            OUTPUT_VARIABLE submodule_dirty
            ERROR_VARIABLE submodule_dirty_error
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT submodule_dirty_result EQUAL 0 OR
           NOT submodule_dirty STREQUAL "")
            message(FATAL_ERROR
                "${label} dependency submodule ${submodule_path} is not clean: "
                "${submodule_dirty} ${submodule_dirty_error}")
        endif()
    endforeach()

    file(SHA256 "${patch_path}" patch_sha)
    set("${revision_variable}" "${revision}" PARENT_SCOPE)
    set("${patch_sha_variable}" "${patch_sha}" PARENT_SCOPE)
    set("${tree_variable}" "${actual_tree}" PARENT_SCOPE)
endfunction()

verify_dependency(
    KIDSCPP "${WP7_KIDSCPP_SOURCE_DIR}" "${WP7_KIDSCPP_REVISION}"
    "${WP7_KIDSCPP_PATCH}" kidscpp_revision kidscpp_patch_sha kidscpp_tree)
verify_dependency(
    TULA "${WP7_TULA_SOURCE_DIR}" "${WP7_TULA_REVISION}"
    "${WP7_TULA_PATCH}" tula_revision tula_patch_sha tula_tree)

set(header_content
    "#pragma once\n\n"
    "#define CITLALI_WP7_SOURCE_STATE_VERIFIED 1\n"
    "#define CITLALI_WP7_SOURCE_REVISION \"${citlali_revision}\"\n"
    "#define CITLALI_WP7_DEPENDENCY_STATE_VERIFIED 1\n"
    "#define CITLALI_WP7_KIDSCPP_REVISION \"${kidscpp_revision}\"\n"
    "#define CITLALI_WP7_KIDSCPP_PATCH_SHA256 \"${kidscpp_patch_sha}\"\n"
    "#define CITLALI_WP7_KIDSCPP_TREE \"${kidscpp_tree}\"\n"
    "#define CITLALI_WP7_TULA_REVISION \"${tula_revision}\"\n"
    "#define CITLALI_WP7_TULA_PATCH_SHA256 \"${tula_patch_sha}\"\n"
    "#define CITLALI_WP7_TULA_TREE \"${tula_tree}\"\n")
string(JOIN "" header_content ${header_content})

set(write_header TRUE)
if(EXISTS "${WP7_DEPENDENCY_IDENTITY_HEADER}")
    file(READ "${WP7_DEPENDENCY_IDENTITY_HEADER}" existing_header)
    if(existing_header STREQUAL header_content)
        set(write_header FALSE)
    endif()
endif()
if(write_header)
    set(temporary_header "${WP7_DEPENDENCY_IDENTITY_HEADER}.tmp")
    file(WRITE "${temporary_header}" "${header_content}")
    file(RENAME "${temporary_header}" "${WP7_DEPENDENCY_IDENTITY_HEADER}")
endif()

file(REMOVE_RECURSE "${state_dir}")
