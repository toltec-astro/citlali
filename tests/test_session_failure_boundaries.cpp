#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/rawobs_tone_frequency_inventory.h>
#include <citlali/core/pipeline/timestream_scan_context.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <map>
#include <memory>
#include <string>
#include <tuple>

namespace {

struct BoundaryLogger {
    template <class... Args>
    void error(const char *, Args &&...) {}

    template <class... Args>
    void warn(const char *, Args &&...) {}

    template <class... Args>
    void debug(const char *, Args &&...) {}
};

struct ToneCalibration {
    Eigen::Index n_dets = 2;
    std::map<int, std::tuple<Eigen::Index, Eigen::Index>> nw_limits{
        {0, {0, 2}}};
    std::map<std::string, Eigen::VectorXd> apt;
};

struct GapCalibration {
    std::map<int, std::tuple<Eigen::Index, Eigen::Index>> nw_limits{
        {0, {0, 1}}};
};

struct FlagData {
    Eigen::MatrixXi data;
};

struct RtcFlagData {
    FlagData flags;
};

}  // namespace

TEST(session_failure_boundaries, rejects_missing_tone_frequency_network) {
    ToneCalibration calibration;
    const citlali::pipeline::RawObsToneFrequencies frequencies;
    auto logger = std::make_shared<BoundaryLogger>();

    EXPECT_THROW(
        citlali::pipeline::assign_tone_frequencies_by_network(
            calibration, frequencies, logger),
        citlali::error::Error);
}

TEST(session_failure_boundaries, rejects_empty_tone_frequency_sweeps) {
    ToneCalibration calibration;
    citlali::pipeline::RawObsToneFrequencies frequencies;
    frequencies[0].resize(2, 0);
    auto logger = std::make_shared<BoundaryLogger>();

    EXPECT_THROW(
        citlali::pipeline::assign_tone_frequencies_by_network(
            calibration, frequencies, logger),
        citlali::error::Error);
}

TEST(session_failure_boundaries, rejects_tone_frequency_size_mismatch) {
    ToneCalibration calibration;
    citlali::pipeline::RawObsToneFrequencies frequencies;
    frequencies[0] = Eigen::MatrixXd::Zero(1, 1);
    auto logger = std::make_shared<BoundaryLogger>();

    EXPECT_THROW(
        citlali::pipeline::assign_tone_frequencies_by_network(
            calibration, frequencies, logger),
        citlali::error::Error);
}

TEST(session_failure_boundaries, rejects_missing_gap_mask) {
    RtcFlagData rtcdata;
    rtcdata.flags.data = Eigen::MatrixXi::Zero(1, 1);
    GapCalibration calibration;
    const std::map<int, Eigen::VectorXi> masks;
    auto logger = std::make_shared<BoundaryLogger>();

    EXPECT_THROW(
        citlali::pipeline::apply_gap_masks_to_rtc_flags(
            rtcdata, calibration, masks, 0, 0, logger),
        citlali::error::Error);
}
