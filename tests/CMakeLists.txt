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
        test_canonical_apt_v1.cpp
        test_canonical_apt_v2.cpp
        ${CMAKE_SOURCE_DIR}/src/citlali/cli/canonical_apt_contract_protocol_v2.cpp
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
        jinc_contract_header.cpp
        test_jinc_map_contract.cpp
        test_learning_and_fruit_contracts.cpp
        test_map_fitter_lifecycle.cpp
        test_output_root_lease.cpp
        test_pointing_fit_table_metrics.cpp
        test_process_resource_snapshot.cpp
        test_rawobs_adc_snap.cpp
        test_reduction_session.cpp
        test_science_map_provenance.cpp
        test_science_map_contract.cpp
        test_science_map_truth_suite.cpp
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

add_executable(citlali_science_map_truth_test EXCLUDE_FROM_ALL
    test_science_map_contract.cpp
    test_science_map_provenance.cpp
    test_science_map_truth_suite.cpp
)

add_executable(citlali_jinc_map_contract_test EXCLUDE_FROM_ALL
    jinc_contract_header.cpp
    test_jinc_map_contract.cpp
)
set_target_properties(citlali_jinc_map_contract_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(citlali_jinc_map_contract_test
    PRIVATE
        citlali::citlali
        GTest::gtest_main
)
add_dependencies(check citlali_jinc_map_contract_test)
gtest_discover_tests(
    citlali_jinc_map_contract_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        TEST_PREFIX "citlali::jinc::"
        DISCOVERY_MODE PRE_TEST
)

add_executable(citlali_sci_align_test EXCLUDE_FROM_ALL
    canonical_apt_detector_relation_v2_header.cpp
    native_observation_carriers_header.cpp
    timestream_native_alignment_header.cpp
    timestream_measured_scan_header.cpp
    timestream_rtc_run_adapter_header.cpp
    timestream_ptc_cohort_adapter_header.cpp
    timestream_native_science_projection_header.cpp
    test_sci_align_canonical_apt_relation_v2.cpp
    test_sci_align_native_carriers.cpp
    test_sci_align_native_gap_fixture.cpp
    test_sci_align_measured_scan.cpp
    test_sci_align_rtc_run_adapter.cpp
    test_sci_align_ptc_cohort_adapter.cpp
    test_sci_align_native_science_projection.cpp
    test_sci_align_partial_cohort.cpp
    test_sci_align_pca_placeholder_invariance.cpp
    test_sci_align_native_scatter.cpp
)
set_source_files_properties(
    canonical_apt_detector_relation_v2_header.cpp
    native_observation_carriers_header.cpp
    timestream_native_alignment_header.cpp
    timestream_measured_scan_header.cpp
    timestream_rtc_run_adapter_header.cpp
    timestream_ptc_cohort_adapter_header.cpp
    timestream_native_science_projection_header.cpp
    PROPERTIES SKIP_PRECOMPILE_HEADERS ON
)
set_target_properties(citlali_sci_align_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(citlali_sci_align_test
    PRIVATE
        citlali::citlali
        GTest::gtest_main
)
target_compile_definitions(citlali_sci_align_test
    PRIVATE
        CITLALI_SCI_ALIGN_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/fixtures/sci_align"
)
add_dependencies(check citlali_sci_align_test)
gtest_discover_tests(
    citlali_sci_align_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        TEST_PREFIX "citlali::sci_align::"
        DISCOVERY_MODE PRE_TEST
)
set_target_properties(citlali_science_map_truth_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(citlali_science_map_truth_test
    PRIVATE
        citlali::citlali
        GTest::gtest_main
)

# Engine's production FITS writer is supplied by legacy implementation-detail
# headers. Isolate this test in one translation unit so it can exercise that
# real boundary without duplicating those definitions across citlali_test.
add_executable(citlali_science_map_fits_products_test EXCLUDE_FROM_ALL
    test_science_map_fits_products.cpp
)
set_target_properties(citlali_science_map_fits_products_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(citlali_science_map_fits_products_test
    PRIVATE
        citlali::citlali
        GTest::gtest_main
)
add_dependencies(check citlali_science_map_fits_products_test)
gtest_discover_tests(
    citlali_science_map_fits_products_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        TEST_PREFIX "citlali::science_map_fits::"
        DISCOVERY_MODE PRE_TEST
)

# SCI-MAP-001-F001: the repaired ordinary primitive is a header-instantiated
# template, so this focused target instruments the primitive and its
# concurrent shared-map commit directly. It is intentionally excluded from
# the default build and ordinary CTest discovery; run it explicitly to retain
# the sanitizer report as exact-SHA evidence.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_executable(citlali_science_map_tsan_test EXCLUDE_FROM_ALL
        test_science_map_truth_suite.cpp
    )
    set_target_properties(citlali_science_map_tsan_test
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    target_link_libraries(citlali_science_map_tsan_test
        PRIVATE
            citlali::citlali
            GTest::gtest_main
    )
    target_compile_options(citlali_science_map_tsan_test
        PRIVATE
            -fsanitize=thread
            -fno-omit-frame-pointer
    )
    target_link_options(citlali_science_map_tsan_test
        PRIVATE
            -fsanitize=thread
    )
endif()

add_executable(citlali_jinc_parallel_ownership_test EXCLUDE_FROM_ALL
    test_jinc_parallel_ownership.cpp
)
set_target_properties(citlali_jinc_parallel_ownership_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(citlali_jinc_parallel_ownership_test
    PRIVATE
        citlali::citlali
        GTest::gtest_main
)
add_dependencies(check citlali_jinc_parallel_ownership_test)
gtest_discover_tests(
    citlali_jinc_parallel_ownership_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        TEST_PREFIX "citlali::jinc_parallel_ownership::"
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
