#include <citlali/core/config/config_error.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

namespace citlali::pipeline {
#include <citlali/core/pipeline/timestream_config_mirror_raw_filters.h>
}  // namespace citlali::pipeline

#include <gtest/gtest.h>

#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct StringConfig {
    std::string value;

    template <class T, class Key>
    bool has_typed(const Key &) const {
        return true;
    }

    template <class T, class Key>
    T get_typed(const Key &) const {
        return T{value};
    }
};

struct FakeIirRtcProc {
    bool run_tod_iir_highpass = false;
    struct {
        double iir_highpass_freq_Hz = 12.0;
        int iir_highpass_order = 4;
        bool iir_highpass_zero_phase = true;
    } filter;
};

struct FakeCorrectionRtcProc {
    bool run_calibrate = false;
    bool run_extinction = false;
    struct {
        std::string extinction_model;
    } calibration;
};

void ensure_test_logger() {
    if (spdlog::get("citlali_logger") != nullptr) {
        return;
    }
    auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    spdlog::register_logger(
        std::make_shared<spdlog::logger>("citlali_logger", sink));
}

TEST(config_safety, parsed_enum_failure_records_invalid_path) {
    ensure_test_logger();
    StringConfig config{"not-a-tod-type"};
    std::string raw_value;
    auto typed_value = citlali::config::TodType::xs;
    std::vector<std::vector<std::string>> missing_keys;
    std::vector<std::vector<std::string>> invalid_keys;
    const auto key = std::tuple{"timestream", "type"};

    citlali::pipeline::read_parsed_mirrored_config_value(
        config, key, raw_value, typed_value,
        citlali::config::parse_tod_type, missing_keys, invalid_keys);

    EXPECT_TRUE(missing_keys.empty());
    ASSERT_EQ(invalid_keys.size(), 1U);
    EXPECT_EQ(invalid_keys.front(),
              (std::vector<std::string>{"timestream", "type"}));
    EXPECT_EQ(typed_value, citlali::config::TodType::xs);
}

TEST(config_safety, range_checks_reject_nonfinite_values) {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto inf = std::numeric_limits<double>::infinity();

    citlali::config::ValidationReport minimum_report;
    citlali::config::check_minimum(
        nan, 0.0, {"value"}, minimum_report);
    ASSERT_EQ(minimum_report.error_count(), 1U);
    EXPECT_EQ(minimum_report.errors().front().message, "must be finite");

    citlali::config::ValidationReport maximum_report;
    citlali::config::check_maximum(
        inf, 1.0, {"value"}, maximum_report);
    ASSERT_EQ(maximum_report.error_count(), 1U);
    EXPECT_EQ(maximum_report.errors().front().message, "must be finite");

    citlali::config::ValidationReport greater_than_report;
    citlali::config::check_greater_than(
        -inf, 0.0, {"value"}, greater_than_report);
    ASSERT_EQ(greater_than_report.error_count(), 1U);
    EXPECT_EQ(greater_than_report.errors().front().message, "must be finite");
}

TEST(config_safety, optional_minimum_allows_nan_sentinel_but_rejects_infinity) {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto inf = std::numeric_limits<double>::infinity();
    citlali::config::ValidationReport report;

    citlali::config::check_optional_minimum(
        nan, 0.0, {"optional"}, report);
    EXPECT_TRUE(report.ok());

    citlali::config::check_optional_minimum(
        inf, 0.0, {"optional"}, report);
    ASSERT_EQ(report.error_count(), 1U);
    EXPECT_EQ(report.errors().front().message, "must be finite");
}

TEST(config_safety, authoritative_range_check_rejects_nonfinite_values) {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto inf = std::numeric_limits<double>::infinity();
    const auto key = std::tuple{"numeric", "value"};
    std::vector<std::vector<std::string>> missing_keys;
    std::vector<std::vector<std::string>> invalid_keys;

    check_range(nan, missing_keys, invalid_keys, std::vector{0.0},
                std::vector<double>{}, key);
    ASSERT_EQ(invalid_keys.size(), 1U);
    EXPECT_EQ(invalid_keys.front(),
              (std::vector<std::string>{"numeric", "value"}));

    invalid_keys.clear();
    check_range(inf, missing_keys, invalid_keys, std::vector{0.0},
                std::vector<double>{}, key);
    EXPECT_EQ(invalid_keys.size(), 1U);
}

TEST(config_safety, authoritative_optional_range_allows_only_nan_nonfinite) {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto inf = std::numeric_limits<double>::infinity();
    const auto key = std::tuple{"optional", "value"};
    std::vector<std::vector<std::string>> missing_keys;
    std::vector<std::vector<std::string>> invalid_keys;

    check_range(nan, missing_keys, invalid_keys, std::vector{0.0},
                std::vector<double>{}, key, true);
    EXPECT_TRUE(invalid_keys.empty());

    check_range(inf, missing_keys, invalid_keys, std::vector{0.0},
                std::vector<double>{}, key, true);
    EXPECT_EQ(invalid_keys.size(), 1U);
}

TEST(config_safety, disabled_iir_mirror_uses_legacy_effective_values) {
    citlali::config::RawTimeChunkIirFilterConfig target;
    target.freq_Hz = 99.0;
    target.order = 9;
    target.zero_phase = true;

    FakeIirRtcProc rtcproc;
    citlali::pipeline::mirror_raw_iir_filter_config(target, rtcproc);

    EXPECT_FALSE(target.enabled);
    EXPECT_DOUBLE_EQ(target.freq_Hz, 0.0);
    EXPECT_EQ(target.order, 1);
    EXPECT_FALSE(target.zero_phase);
}

TEST(config_safety, enabled_iir_mirror_preserves_effective_values) {
    citlali::config::RawTimeChunkIirFilterConfig target;
    FakeIirRtcProc rtcproc;
    rtcproc.run_tod_iir_highpass = true;

    citlali::pipeline::mirror_raw_iir_filter_config(target, rtcproc);

    EXPECT_TRUE(target.enabled);
    EXPECT_DOUBLE_EQ(target.freq_Hz, 12.0);
    EXPECT_EQ(target.order, 4);
    EXPECT_TRUE(target.zero_phase);
}

TEST(config_safety, disabled_extinction_mirror_uses_na_sentinel) {
    citlali::config::RawTimeChunkConfig target;
    target.extinction_model = "stale-model";
    FakeCorrectionRtcProc rtcproc;

    citlali::pipeline::mirror_raw_correction_flags(target, rtcproc);

    EXPECT_FALSE(target.extinction_correction_enabled);
    EXPECT_EQ(target.extinction_model, "N/A");
}

TEST(config_safety, enabled_extinction_mirror_preserves_model) {
    citlali::config::RawTimeChunkConfig target;
    FakeCorrectionRtcProc rtcproc;
    rtcproc.run_extinction = true;
    rtcproc.calibration.extinction_model = "am_q50";

    citlali::pipeline::mirror_raw_correction_flags(target, rtcproc);

    EXPECT_TRUE(target.extinction_correction_enabled);
    EXPECT_EQ(target.extinction_model, "am_q50");
}

}  // namespace
