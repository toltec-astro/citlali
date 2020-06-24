project (common_utils_test LANGUAGES CXX C)

add_executable(common_utils_test)
set_target_properties(common_utils_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_sources(common_utils_test
    PRIVATE
        main.cpp
        formatter.cpp
        eigeniter.cpp
    )
target_link_libraries(common_utils_test
    PRIVATE
        common_utils
        gtest gmock benchmark
    )
add_custom_command(TARGET common_utils_test
    POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR}/data
    )
add_dependencies(check common_utils_test)
gtest_discover_tests(common_utils_test TEST_PREFIX "common_utils::")
