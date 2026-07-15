#include <citlali/core/cli/reduction_result_reporting.h>
#include <citlali/core/session/reduction_session.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <sstream>
#include <stdexcept>

std::size_t reduction_session_header_state_from_translation_unit();
bool reduction_session_header_result_from_translation_unit();

TEST(reduction_session, returns_structured_success) {
    citlali::session::ReductionSession session;

    const auto result = session.run([] {
        auto result = citlali::session::successful_reduction_result();
        result.product_roots.emplace_back("/data/redu01");
        result.provenance_artifacts.emplace_back(
            "/data/redu01/runtime_provenance.yaml");
        return result;
    });

    EXPECT_TRUE(result.succeeded());
    ASSERT_EQ(result.product_roots.size(), 1U);
    EXPECT_EQ(result.product_roots.front(), "/data/redu01");
    ASSERT_EQ(result.provenance_artifacts.size(), 1U);
    EXPECT_EQ(
        result.provenance_artifacts.front(),
        "/data/redu01/runtime_provenance.yaml");
    EXPECT_EQ(session.state(),
              citlali::session::ReductionSessionState::succeeded);
    EXPECT_EQ(session.runs_started(), 1U);
}

TEST(reduction_session, converts_exceptions_to_diagnostics) {
    citlali::session::ReductionSession session;

    const auto result = session.run(
        []() -> citlali::session::ReductionResult {
            throw std::runtime_error("required output failed");
        });

    EXPECT_FALSE(result.succeeded());
    EXPECT_EQ(result.status,
              citlali::session::ReductionStatus::unhandled_exception);
    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics.front().code,
              "session.unhandled_exception");
    EXPECT_EQ(result.diagnostics.front().message,
              "required output failed");
    EXPECT_EQ(session.state(),
              citlali::session::ReductionSessionState::failed);
}

TEST(reduction_session, classifies_canonical_library_errors) {
    struct Case {
        citlali::error::Code code;
        citlali::session::ReductionStatus status;
        const char *diagnostic_code;
    };
    const Case cases[] = {
        {citlali::error::Code::invalid_config,
         citlali::session::ReductionStatus::invalid_request,
         "config.invalid"},
        {citlali::error::Code::io,
         citlali::session::ReductionStatus::io_failed, "io.failed"},
        {citlali::error::Code::output,
         citlali::session::ReductionStatus::output_failed, "output.failed"},
        {citlali::error::Code::runtime,
         citlali::session::ReductionStatus::execution_failed,
         "runtime.failed"},
        {citlali::error::Code::internal,
         citlali::session::ReductionStatus::execution_failed,
         "internal.failed"},
    };

    for (const auto &test_case : cases) {
        citlali::session::ReductionSession session;
        const auto result = session.run([&]()
            -> citlali::session::ReductionResult {
            throw citlali::error::Error{test_case.code, "injected failure"};
        });

        EXPECT_EQ(result.status, test_case.status);
        ASSERT_EQ(result.diagnostics.size(), 1U);
        EXPECT_EQ(result.diagnostics.front().code,
                  test_case.diagnostic_code);
        EXPECT_EQ(result.diagnostics.front().message, "injected failure");
    }
}

TEST(reduction_session, supports_success_after_failure) {
    citlali::session::ReductionSession session;

    const auto failed = session.run([] {
        return citlali::session::failed_reduction_result(
            citlali::session::ReductionStatus::execution_failed,
            "pipeline.failed", "injected failure");
    });
    const auto succeeded = session.run([] {
        return citlali::session::successful_reduction_result();
    });

    EXPECT_FALSE(failed.succeeded());
    EXPECT_TRUE(succeeded.succeeded());
    EXPECT_EQ(session.state(),
              citlali::session::ReductionSessionState::succeeded);
    EXPECT_EQ(session.runs_started(), 2U);
}

TEST(reduction_session, supports_two_sequential_successes) {
    citlali::session::ReductionSession session;

    const auto first = session.run([] {
        return citlali::session::successful_reduction_result();
    });
    const auto second = session.run([] {
        return citlali::session::successful_reduction_result();
    });

    EXPECT_TRUE(first.succeeded());
    EXPECT_TRUE(second.succeeded());
    EXPECT_EQ(session.runs_started(), 2U);
}

TEST(reduction_session, rejects_nested_run_without_losing_outer_state) {
    citlali::session::ReductionSession session;
    citlali::session::ReductionResult nested;

    const auto outer = session.run([&] {
        nested = session.run([] {
            return citlali::session::successful_reduction_result();
        });
        return citlali::session::successful_reduction_result();
    });

    EXPECT_TRUE(outer.succeeded());
    EXPECT_EQ(nested.status,
              citlali::session::ReductionStatus::invalid_session_state);
    ASSERT_EQ(nested.diagnostics.size(), 1U);
    EXPECT_EQ(nested.diagnostics.front().code, "session.already_running");
    EXPECT_EQ(session.state(),
              citlali::session::ReductionSessionState::succeeded);
    EXPECT_EQ(session.runs_started(), 1U);
}

TEST(reduction_session, keeps_cli_reporting_outside_session_policy) {
    auto result = citlali::session::failed_reduction_result(
        citlali::session::ReductionStatus::invalid_request,
        "config.invalid", "expected an integer", {"downsample", "factor"});
    std::ostringstream diagnostics;

    citlali::cli::report_reduction_result_diagnostics(result, diagnostics);

    EXPECT_EQ(citlali::cli::reduction_result_exit_code(result), EXIT_FAILURE);
    EXPECT_EQ(diagnostics.str(),
              "config.invalid [downsample.factor]: expected an integer\n");
}

TEST(reduction_session, public_headers_link_across_translation_units) {
    EXPECT_EQ(reduction_session_header_state_from_translation_unit(), 0U);
    EXPECT_TRUE(reduction_session_header_result_from_translation_unit());
}
