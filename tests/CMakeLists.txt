find_package(GTest CONFIG REQUIRED)

include(GoogleTest)
set(_citlali_test_names gauss_models)
if(CITLALI_HAS_REAL_TEST_DATA)
    list(APPEND _citlali_test_names kidsproc)
endif()
foreach(_name IN LISTS _citlali_test_names)
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
        DISCOVERY_TIMEOUT 30
        PROPERTIES
            ENVIRONMENT
                "TOLTECA_TEST_DATA_ROOT=${TOLTECA_TEST_DATA_ROOT}"
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
