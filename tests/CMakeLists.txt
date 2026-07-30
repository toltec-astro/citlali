project (citlali_test LANGUAGES CXX C)

include(testing)

add_executable(citlali_test EXCLUDE_FROM_ALL)
set_target_properties(citlali_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_sources(citlali_test
    PRIVATE
        test_astrometry_config_plan.cpp
        test_beammap_config_plan.cpp
        test_calib_apt_filtering.cpp
        test_coherent_iq_mode_observer.cpp
        test_config_scaffold.cpp
        test_external_config_provenance.cpp
        test_fruit_loop_recurrence.cpp
        reduction_session_header_result.cpp
        reduction_session_header_state.cpp
        fruit_loop_feedback_validation_header.cpp
        fruit_loop_map_input_validation_header.cpp
        output_root_lease_header.cpp
        reduction_restart_checkpoint_header.cpp
        wiener_filter_validation_header.cpp
        test_kids_input_validation.cpp
        test_learning_and_fruit_contracts.cpp
        test_map_fitter_lifecycle.cpp
        test_output_root_lease.cpp
        test_pointing_fit_table_metrics.cpp
        test_process_resource_snapshot.cpp
        test_rawobs_adc_snap.cpp
        test_reduction_session.cpp
        test_session_failure_boundaries.cpp
        test_tod_output_selection_config.cpp
        test_utils.cpp
)
target_link_libraries(citlali_test
    PRIVATE
        citlali::citlali
        GTest::gtest_main
        GTest::gmock
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
        test_ordered_writer.cpp
        test_output_schema.cpp
        test_scan_cursor.cpp
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
