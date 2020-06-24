include_guard(GLOBAL)
set(gitversion_dir ${CMAKE_CURRENT_LIST_DIR})
add_library(cmake_utils_gitversion INTERFACE)
add_library(cmake_utils::gitversion ALIAS cmake_utils_gitversion)

function(GenVersionHeader output_dir name)
    if (NOT name)
        set(output_path ${output_dir}/gitversion.h)
    else()
        set(output_path ${output_dir}/${name}/gitversion.h)
    endif()
    message(STATUS "Resolving GIT Version")
    set(_build_version "unknown")
    find_package(Git)
    if(GIT_FOUND)
        set(local_dir ${CMAKE_CURRENT_LIST_DIR})
        message( STATUS "GIT exec: ${GIT_EXECUTABLE}")
        message( STATUS "GIT dir: ${local_dir}")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY "${local_dir}"
            OUTPUT_VARIABLE _build_revision
#            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        message( STATUS "GIT hash: ${_build_version}")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --tags --always --broken
            WORKING_DIRECTORY "${local_dir}"
            OUTPUT_VARIABLE _build_version
#            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

    else()
        message(STATUS "GIT not found")
    endif()

    string(TIMESTAMP _time_stamp)

    configure_file(${gitversion_dir}/gitversion.h.in ${output_path} @ONLY)
    add_dependencies(cmake_utils_gitversion ${output_dir}/gitversion.h)
    target_include_directories(
        cmake_utils_gitversion INTERFACE ${output_dir})
endfunction()
