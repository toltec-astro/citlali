project (citlali_test LANGUAGES CXX C)

add_executable(citlali_test EXCLUDE_FROM_ALL)
set_target_properties(citlali_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_sources(citlali_test
    PRIVATE
        main.cpp
        test_utils.cpp
    )
target_include_directories(citlali_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )
target_link_libraries(citlali_test
    PRIVATE
        gtest gmock benchmark
        spdlog::spdlog
        fmt::fmt
        Eigen3::Eigen
    )
add_dependencies(check citlali_test)
gtest_discover_tests(citlali_test TEST_PREFIX "citlali::")
