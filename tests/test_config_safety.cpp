#include <citlali/core/config/config_error.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

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

}  // namespace
