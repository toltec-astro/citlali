# Internal application service shared by execution and explicit acceptance tools.
function(citlali_add_successor_input_bindings source_root)
    add_library(citlali_successor_input_bindings STATIC
        "${source_root}/src/citlali/cli/successor_input_bindings.cpp")
    target_link_libraries(citlali_successor_input_bindings PUBLIC citlali::citlali)
    target_include_directories(citlali_successor_input_bindings INTERFACE
        "${source_root}/src/citlali/cli")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(citlali_successor_input_bindings PRIVATE
            -fno-fast-math -ffp-contract=off)
    endif()
endfunction()
