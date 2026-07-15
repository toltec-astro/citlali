#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/beammap_fit_validation.h>
#include <citlali/core/pipeline/fits_image_metadata.h>
#include <citlali/core/pipeline/map_filtering.h>
#include <citlali/core/pipeline/mapmaking_config_policy.h>
#include <citlali/core/pipeline/observation_setup_validation.h>
#include <citlali/core/pipeline/phdu_observation_metadata.h>
#include <citlali/core/pipeline/rawobs_tone_frequency_inventory.h>
#include <citlali/core/pipeline/timestream_scan_context.h>
#include <citlali/core/utils/ecsv_io.h>
#include <citlali/core/utils/fits_io.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

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

template <class Action>
void expect_output_failure(Action action) {
    try {
        action();
        FAIL() << "expected required output validation to fail";
    } catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::output);
    }
}

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

TEST(session_failure_boundaries, rejects_detector_count_mismatch) {
    EXPECT_THROW(citlali::pipeline::require_matching_detector_count(2, 3),
                 citlali::error::Error);
}

TEST(session_failure_boundaries, reconciles_matching_sample_rates) {
    EXPECT_DOUBLE_EQ(
        citlali::pipeline::reconcile_sample_rate_hz(-1.0, 488.0, 0),
        488.0);
    EXPECT_DOUBLE_EQ(
        citlali::pipeline::reconcile_sample_rate_hz(488.0, 488.0, 1),
        488.0);
}

TEST(session_failure_boundaries, rejects_sample_rate_mismatch) {
    EXPECT_THROW(
        citlali::pipeline::reconcile_sample_rate_hz(488.0, 244.0, 1),
        citlali::error::Error);
}

TEST(session_failure_boundaries, rejects_nonpositive_sample_rate) {
    EXPECT_THROW(citlali::pipeline::require_positive_sample_rate_hz(
                     0.0, "test alignment"),
                 citlali::error::Error);
}

TEST(session_failure_boundaries, rejects_negative_extinction_tau) {
    EXPECT_THROW(citlali::pipeline::require_nonnegative_extinction_tau(
                     -0.1, "a1100"),
                 citlali::error::Error);
}

TEST(session_failure_boundaries, rejects_missing_polarization_groups) {
    EXPECT_THROW(citlali::pipeline::require_polarization_frequency_groups(true),
                 citlali::error::Error);
}

TEST(session_failure_boundaries, rejects_iir_frequency_above_nyquist) {
    EXPECT_THROW(citlali::pipeline::require_iir_below_nyquist_hz(
                     false, 300.0, 244.0),
                 citlali::error::Error);
}

TEST(session_failure_boundaries, rejects_beammap_fit_geometry_mismatch) {
    EXPECT_THROW(citlali::pipeline::require_beammap_fit_map_geometry(
                     4, 10, 20, 10, 19, 10, 20),
                 citlali::error::Error);
}

TEST(session_failure_boundaries, accepts_valid_required_output_slots) {
    auto logger = std::make_shared<BoundaryLogger>();
    const std::map<Eigen::Index, double> array_fwhms{{0, 5.0}};
    const std::vector<int> noise{1};
    const auto noise_fits = std::make_shared<std::vector<int>>(1, 1);

    EXPECT_NO_THROW(citlali::pipeline::require_map_data_slots(
        0, 1, 1, logger));
    EXPECT_NO_THROW(citlali::pipeline::require_map_write_index_slots(
        0, 0, 1, 0, 1, 0, logger));
    EXPECT_DOUBLE_EQ(citlali::pipeline::require_array_fwhm_for_id(
                         array_fwhms, 0, logger),
                     5.0);
    EXPECT_NO_THROW(citlali::pipeline::require_noise_map_write_slots(
        noise, noise_fits, 0, 0, logger));
    EXPECT_NO_THROW(citlali::pipeline::require_phdu_output_slots(
        0, 1, 1, logger));
}

TEST(session_failure_boundaries, classifies_required_output_slot_failures) {
    auto logger = std::make_shared<BoundaryLogger>();
    const std::map<Eigen::Index, double> array_fwhms{{0, 5.0}};
    const std::vector<int> noise{1};
    const auto noise_fits = std::make_shared<std::vector<int>>(1, 1);

    expect_output_failure([&] {
        citlali::pipeline::require_map_data_slots(1, 1, 1, logger);
    });
    expect_output_failure([&] {
        citlali::pipeline::require_map_write_index_slots(
            0, 1, 1, 0, 1, 0, logger);
    });
    expect_output_failure([&] {
        citlali::pipeline::require_map_write_index_slots(
            0, 0, 1, 1, 1, 0, logger);
    });
    expect_output_failure([&] {
        citlali::pipeline::require_map_write_index_slots(
            0, 0, 1, 0, 1, -1, logger);
    });
    expect_output_failure([&] {
        (void)citlali::pipeline::require_array_fwhm_for_id(
            array_fwhms, 1, logger);
    });
    expect_output_failure([&] {
        citlali::pipeline::require_noise_map_write_slots(
            noise, noise_fits, 1, 0, logger);
    });
    expect_output_failure([&] {
        citlali::pipeline::require_noise_map_write_slots(
            noise, noise_fits, 0, 1, logger);
    });
    expect_output_failure([&] {
        citlali::pipeline::require_phdu_output_slots(1, 1, 1, logger);
    });
    expect_output_failure([&] {
        citlali::pipeline::require_phdu_output_slots(0, 1, 0, logger);
    });
}

TEST(session_failure_boundaries, classifies_missing_fits_input) {
    using FitsInput = fitsIO<file_type_enum::read_fits, CCfits::ExtHDU *>;

    try {
        FitsInput input{
            "/private/tmp/citlali_phase3_missing/input.fits"};
        FAIL() << "expected missing FITS input to fail";
    } catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::io);
    }
}

TEST(session_failure_boundaries, classifies_uncreatable_fits_output) {
    using FitsOutput = fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>;

    try {
        FitsOutput output{
            "/private/tmp/citlali_phase3_missing/output"};
        FAIL() << "expected uncreatable FITS output to fail";
    } catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::output);
    }
}

TEST(session_failure_boundaries, classifies_missing_ecsv_input) {
    try {
        (void)to_map_from_ecsv_mixted_type(
            "/private/tmp/citlali_phase3_missing/input.ecsv");
        FAIL() << "expected missing ECSV input to fail";
    } catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::io);
    }
}

TEST(session_failure_boundaries, accepts_supported_mapmaking_policies) {
    EXPECT_NO_THROW(
        citlali::pipeline::enforce_map_grouping_polarization_policy(
            false, citlali::config::ReductionType::science,
            citlali::config::MapGrouping::detector));
    EXPECT_NO_THROW(citlali::pipeline::enforce_beammap_pixel_axes_policy(
        citlali::config::ReductionType::beammap,
        citlali::config::MapPixelAxes::altaz));

    const std::map<std::string, double> no_fwhm;
    EXPECT_DOUBLE_EQ(citlali::pipeline::require_map_filter_template_fwhm(
                         "kernel", no_fwhm, "a1100"),
                     0.0);
}

TEST(session_failure_boundaries, classifies_invalid_mapmaking_policies) {
    EXPECT_THROW(
        citlali::pipeline::enforce_map_grouping_polarization_policy(
            true, citlali::config::ReductionType::science,
            citlali::config::MapGrouping::detector),
        citlali::error::Error);
    EXPECT_THROW(citlali::pipeline::enforce_beammap_pixel_axes_policy(
                     citlali::config::ReductionType::beammap,
                     citlali::config::MapPixelAxes::radec),
                 citlali::error::Error);

    const std::map<std::string, double> no_fwhm;
    EXPECT_THROW(citlali::pipeline::require_map_filter_template_fwhm(
                     "gaussian", no_fwhm, "a1100"),
                 citlali::error::Error);
}
