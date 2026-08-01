#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/beammap_fit_validation.h>
#include <citlali/core/pipeline/fits_image_metadata.h>
#include <citlali/core/pipeline/fruit_loop_feedback_validation.h>
#include <citlali/core/pipeline/fruit_loop_map_input_validation.h>
#include <citlali/core/pipeline/map_filtering.h>
#include <citlali/core/pipeline/mapmaking_config_policy.h>
#include <citlali/core/pipeline/observation_setup_validation.h>
#include <citlali/core/pipeline/phdu_observation_metadata.h>
#include <citlali/core/pipeline/rawobs_tone_frequency_inventory.h>
#include <citlali/core/pipeline/timestream_scan_context.h>
#include <citlali/core/pipeline/timestream_invariant_validation.h>
#include <citlali/core/pipeline/wiener_filter_validation.h>
#include <citlali/core/session/reduction_session.h>
#include <citlali/core/utils/ecsv_io.h>
#include <citlali/core/utils/fits_io.h>

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <limits>
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

struct BoundaryWcs {
    std::vector<std::string> ctype;
    std::vector<std::string> cunit;
    std::vector<double> crval;
    std::vector<double> cdelt;
    std::vector<double> crpix;
    std::vector<int> naxis;
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

TEST(session_failure_boundaries, classifies_noncontiguous_grouping_as_io) {
    EXPECT_NO_THROW(citlali::pipeline::require_group_value_not_seen(
        false, "nw", 1));
    try {
        citlali::pipeline::require_group_value_not_seen(true, "nw", 1);
        FAIL() << "expected non-contiguous grouping to fail";
    } catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::io);
    }
}

TEST(session_failure_boundaries, classifies_invalid_weight_counters_as_internal) {
    EXPECT_NO_THROW(citlali::pipeline::require_valid_weight_counters(
        10, 8, 7, 1, 2));
    try {
        citlali::pipeline::require_valid_weight_counters(10, 8, 9, 1, 2);
        FAIL() << "expected impossible weight counters to fail";
    } catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::internal);
    }
}

TEST(session_failure_boundaries, classifies_kernel_image_count_as_config) {
    EXPECT_NO_THROW(citlali::pipeline::require_kernel_image_cardinality(1, 3));
    EXPECT_NO_THROW(citlali::pipeline::require_kernel_image_cardinality(3, 3));
    try {
        citlali::pipeline::require_kernel_image_cardinality(2, 3);
        FAIL() << "expected mismatched kernel image count to fail";
    } catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::invalid_config);
    }
}

TEST(session_failure_boundaries, classifies_fruit_loop_map_request_as_config) {
    EXPECT_NO_THROW(citlali::pipeline::require_fruit_loop_map_request(
        true, "valid request"));
    try {
        citlali::pipeline::require_fruit_loop_map_request(
            false, "unsupported grouping 'invalid'");
        FAIL() << "expected invalid fruit-loop request to fail";
    } catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::invalid_config);
        EXPECT_NE(std::string(error.what()).find("unsupported grouping"),
                  std::string::npos);
    }
}

TEST(session_failure_boundaries, fruit_loop_map_input_failure_is_recoverable) {
    citlali::session::ReductionSession session;

    const auto failed = session.run([](auto &) {
        citlali::pipeline::require_fruit_loop_map_input(
            false, "missing signal map index 2");
        return citlali::session::successful_reduction_result();
    });
    EXPECT_EQ(failed.status, citlali::session::ReductionStatus::io_failed);
    ASSERT_EQ(failed.diagnostics.size(), 1);
    EXPECT_EQ(failed.diagnostics.front().code, "io.failed");
    EXPECT_NE(failed.diagnostics.front().message.find("missing signal map"),
              std::string::npos);

    const auto succeeded = session.run([](auto &) {
        citlali::pipeline::require_fruit_loop_map_input(
            true, "valid map input");
        return citlali::session::successful_reduction_result();
    });
    EXPECT_TRUE(succeeded.succeeded());
}

TEST(session_failure_boundaries, validates_fruit_loop_feedback_identity) {
    EXPECT_NO_THROW(citlali::pipeline::require_contiguous_fruit_loop_group(
        false, "array", 1));
    EXPECT_NO_THROW(citlali::pipeline::require_fruit_loop_array_identity(
        true, 1));
    EXPECT_NO_THROW(citlali::pipeline::require_fruit_loop_map_index(2, 3));

    const auto expect_io_failure = [](auto action) {
        try {
            action();
            FAIL() << "expected fruit-loop feedback validation to fail";
        } catch (const citlali::error::Error &error) {
            EXPECT_EQ(error.code(), citlali::error::Code::io);
        }
    };
    expect_io_failure([] {
        citlali::pipeline::require_contiguous_fruit_loop_group(
            true, "array", 1);
    });
    expect_io_failure([] {
        citlali::pipeline::require_fruit_loop_array_identity(false, 4);
    });
    expect_io_failure([] {
        citlali::pipeline::require_fruit_loop_map_index(-1, 3);
    });
    expect_io_failure([] {
        citlali::pipeline::require_fruit_loop_map_index(3, 3);
    });
}

TEST(session_failure_boundaries, validates_wiener_filter_boundaries) {
    EXPECT_NO_THROW(citlali::pipeline::require_wiener_template_geometry(
        10, 20, 10, 20));
    EXPECT_NO_THROW(citlali::pipeline::require_wiener_pixel_spacing(
        1.0, 1.0));
    EXPECT_NO_THROW(citlali::pipeline::require_wiener_kernel_index(1, 2));
    EXPECT_NO_THROW(citlali::pipeline::require_wiener_kernel_weight_index(
        1, 2, 2));
    EXPECT_NO_THROW(citlali::pipeline::require_wiener_kernel_geometry(
        1, 10, 20, 10, 20, 10, 20));
    EXPECT_NO_THROW(citlali::pipeline::require_finite_wiener_kernel_peak(
        1.0, 1));
    EXPECT_DOUBLE_EQ(
        citlali::pipeline::require_wiener_unit_sum_kernel(
            1.0, 4.0, "convolve", "kernel"),
        1.0);
    EXPECT_DOUBLE_EQ(
        citlali::pipeline::require_wiener_unit_sum_kernel(
            1.0, 20.0, "wiener_filter", "kernel"),
        1.0);
    EXPECT_NO_THROW(citlali::pipeline::require_wiener_fftw_context(
        true, 10, 20));

    const auto expect_runtime_failure = [](auto action) {
        try {
            action();
            FAIL() << "expected Wiener boundary validation to fail";
        } catch (const citlali::error::Error &error) {
            EXPECT_EQ(error.code(), citlali::error::Code::runtime);
        }
    };
    expect_runtime_failure([] {
        citlali::pipeline::require_wiener_template_geometry(1, 20, 1, 20);
    });
    expect_runtime_failure([] {
        citlali::pipeline::require_wiener_pixel_spacing(0.0, 1.0);
    });
    expect_runtime_failure([] {
        citlali::pipeline::require_wiener_kernel_index(2, 2);
    });
    expect_runtime_failure([] {
        citlali::pipeline::require_wiener_kernel_weight_index(1, 2, 1);
    });
    expect_runtime_failure([] {
        citlali::pipeline::require_wiener_kernel_geometry(
            1, 10, 19, 10, 20, 10, 20);
    });
    expect_runtime_failure([] {
        citlali::pipeline::require_finite_wiener_kernel_peak(
            std::numeric_limits<double>::quiet_NaN(), 1);
    });
    expect_runtime_failure([] {
        citlali::pipeline::require_wiener_unit_sum_kernel(
            0.0, 4.0, "convolve", "kernel");
    });
    expect_runtime_failure([] {
        citlali::pipeline::require_wiener_unit_sum_kernel(
            0.19, 4.0, "wiener_filter", "kernel");
    });
    expect_runtime_failure([] {
        citlali::pipeline::require_wiener_unit_sum_kernel(
            std::numeric_limits<double>::quiet_NaN(), 4.0,
            "convolve", "kernel");
    });
    expect_runtime_failure([] {
        citlali::pipeline::require_wiener_fftw_context(false, 10, 20);
    });
}

TEST(session_failure_boundaries, accepts_valid_required_output_slots) {
    auto logger = std::make_shared<BoundaryLogger>();
    const std::map<Eigen::Index, double> array_fwhms{{0, 5.0}};
    const std::vector<int> noise{1};
    const auto noise_fits = std::make_shared<std::vector<int>>(1, 1);

    EXPECT_NO_THROW(citlali::pipeline::require_map_data_slots(
        0, 1, 1, logger));
    const std::vector<Eigen::MatrixXd> signal{
        Eigen::MatrixXd::Zero(2, 3)};
    const std::vector<Eigen::MatrixXd> weight{
        Eigen::MatrixXd::Ones(2, 3)};
    EXPECT_NO_THROW(citlali::pipeline::require_primary_map_image_shapes(
        signal, weight, 0, 2, 3, logger));
    BoundaryWcs wcs{
        {"A", "B", "FREQ", "STOKES"},
        {"u", "u", "Hz", "1"},
        {0.0, 0.0, 1.0, 0.0},
        {1.0, 1.0, 1.0, 1.0},
        {0.0, 0.0, 0.0, 0.0},
        {3, 2},
    };
    EXPECT_NO_THROW(citlali::pipeline::require_map_wcs_cardinality(
        wcs, 4, logger));
    EXPECT_NO_THROW(citlali::pipeline::require_map_write_index_slots(
        0, 0, 1, 0, 1, 0, logger));
    EXPECT_DOUBLE_EQ(citlali::pipeline::require_array_fwhm_for_id(
                         array_fwhms, 0, logger),
                     5.0);
    EXPECT_NO_THROW(citlali::pipeline::require_noise_map_write_slots(
        noise, noise_fits, 0, 0, logger));
    std::vector<Eigen::Tensor<double, 3>> noise_tensors(1);
    noise_tensors[0].resize(2, 3, 4);
    EXPECT_NO_THROW(citlali::pipeline::require_noise_map_tensor_shape(
        noise_tensors, 0, 2, 3, 4, logger));
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
    const std::vector<Eigen::MatrixXd> signal{
        Eigen::MatrixXd::Zero(2, 3)};
    const std::vector<Eigen::MatrixXd> wrong_weight{
        Eigen::MatrixXd::Ones(3, 2)};
    expect_output_failure([&] {
        citlali::pipeline::require_primary_map_image_shapes(
            signal, wrong_weight, 0, 2, 3, logger);
    });
    BoundaryWcs uneven_wcs{
        {"A", "B", "FREQ", "STOKES", "EXTRA"},
        {"u", "u", "Hz", "1"},
        {0.0, 0.0, 1.0, 0.0},
        {1.0, 1.0, 1.0, 1.0},
        {0.0, 0.0, 0.0, 0.0},
        {3, 2},
    };
    expect_output_failure([&] {
        citlali::pipeline::require_map_wcs_cardinality(
            uneven_wcs, 4, logger);
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
    std::vector<Eigen::Tensor<double, 3>> noise_tensors(1);
    noise_tensors[0].resize(2, 3, 4);
    expect_output_failure([&] {
        citlali::pipeline::require_noise_map_tensor_shape(
            noise_tensors, 0, 2, 3, -1, logger);
    });
    expect_output_failure([&] {
        citlali::pipeline::require_noise_map_tensor_shape(
            noise_tensors, 1, 2, 3, 4, logger);
    });
    expect_output_failure([&] {
        citlali::pipeline::require_noise_map_tensor_shape(
            noise_tensors, 0, 2, 3, 5, logger);
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
