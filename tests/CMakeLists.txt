project (citlali_test LANGUAGES CXX C)

include(testing)

add_executable(citlali_test EXCLUDE_FROM_ALL)
set_target_properties(citlali_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_sources(citlali_test
    PRIVATE
        main.cpp
        test_config_scaffold.cpp
        test_utils.cpp
    )
target_link_libraries(citlali_test
    PRIVATE
        citlali::citlali
        tula::testing
    )
add_dependencies(check citlali_test)
gtest_discover_tests(
    citlali_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        TEST_PREFIX "citlali::"
        DISCOVERY_MODE PRE_TEST
    )

add_executable(citlali_safety_test EXCLUDE_FROM_ALL)
set_target_properties(citlali_safety_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_sources(citlali_safety_test
    PRIVATE
        test_config_safety.cpp
        test_output_schema.cpp
)
target_link_libraries(citlali_safety_test
    PRIVATE
        citlali::citlali
        GTest::gtest_main
)
target_compile_definitions(citlali_safety_test PRIVATE LOGLEVEL=${LOGLEVEL})
gtest_discover_tests(
    citlali_safety_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        TEST_PREFIX "citlali::safety::"
        DISCOVERY_MODE PRE_TEST
)
