find_package(GTest CONFIG REQUIRED)

include(GoogleTest)
foreach(_name gauss_models kidsproc)
    add_executable("citlali_test_${_name}" "test_${_name}.cpp")
    target_link_libraries(
        "citlali_test_${_name}"
        PRIVATE
            citlali::citlali
            GTest::gtest_main
    )
    gtest_discover_tests(
        "citlali_test_${_name}"
        TEST_PREFIX "citlali::"
        DISCOVERY_MODE PRE_TEST
    )
endforeach()

if(TARGET citlali_cli)
    add_test(NAME citlali::cli.help COMMAND citlali_cli --help)
    add_test(NAME citlali::cli.version COMMAND citlali_cli --version)
    add_test(NAME citlali::cli.dump_config COMMAND citlali_cli --dump_config)
    set_tests_properties(
        citlali::cli.dump_config
        PROPERTIES
            PASS_REGULAR_EXPRESSION "kids:[^\n]*KIDs timestream solving parameters"
            FAIL_REGULAR_EXPRESSION "fitter:"
    )
endif()
