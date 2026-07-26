find_package(GTest CONFIG REQUIRED)

add_executable(citlali_test_gauss_models test_gauss_models.cpp)
target_link_libraries(
    citlali_test_gauss_models
    PRIVATE
        citlali::citlali
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(
    citlali_test_gauss_models
    TEST_PREFIX "citlali::"
    DISCOVERY_MODE PRE_TEST
)
