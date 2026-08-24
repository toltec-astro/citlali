include_guard(GLOBAL)

function(citlali_enable_buildtime_gitversion_refresh
         gitversion_target source_dir output_path)
    if(NOT TARGET "${gitversion_target}")
        message(FATAL_ERROR
            "Git-version refresh target does not exist: ${gitversion_target}")
    endif()

    set(refresh_target "${gitversion_target}_buildtime_refresh")
    if(TARGET "${refresh_target}")
        message(FATAL_ERROR
            "Git-version refresh target already exists: ${refresh_target}")
    endif()

    add_custom_target("${refresh_target}"
        COMMAND "${CMAKE_COMMAND}"
            "-DCITLALI_GIT_SOURCE_DIR=${source_dir}"
            "-DCITLALI_GITVERSION_HEADER=${output_path}"
            -P
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/refresh_gitversion_header.cmake"
        COMMENT "Refresh Citlali Git-version provenance"
        VERBATIM
    )
    add_dependencies("${gitversion_target}" "${refresh_target}")
endfunction()
