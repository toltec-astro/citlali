#include <citlali/core/utils/fitting.h>
#include <citlali/core/utils/fits_io.h>

#include <gtest/gtest.h>
#include <spdlog/sinks/null_sink.h>
#include <yaml-cpp/yaml.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

std::shared_ptr<spdlog::logger> ensure_fitter_test_logger() {
    auto logger = spdlog::get("citlali_logger");
    if (logger == nullptr) {
        auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        logger = std::make_shared<spdlog::logger>("citlali_logger", sink);
        spdlog::register_logger(logger);
    }
    return logger;
}

struct PointingFitProblem {
    Eigen::MatrixXd signal;
    Eigen::MatrixXd weight;

    PointingFitProblem()
        : signal(Eigen::MatrixXd::Zero(121, 121)),
          weight(Eigen::MatrixXd::Zero(121, 121)) {
        constexpr double amplitude = 663.14;
        constexpr double center_row = 60.0;
        constexpr double center_col = 60.0;
        constexpr double sigma = 2.67536;
        for (Eigen::Index row = 0; row < signal.rows(); ++row) {
            for (Eigen::Index col = 0; col < signal.cols(); ++col) {
                const double dr = static_cast<double>(row) - center_row;
                const double dc = static_cast<double>(col) - center_col;
                const double background =
                    2.0 * std::sin(0.13 * static_cast<double>(row)) +
                    1.5 * std::cos(0.17 * static_cast<double>(col));
                signal(row, col) =
                    amplitude *
                        std::exp(-(dr * dr + dc * dc) /
                                 (2.0 * sigma * sigma)) +
                    background;
                const double noise_sigma =
                    7.0 + 0.02 * static_cast<double>((row + col) % 31);
                weight(row, col) = 1.0 / (noise_sigma * noise_sigma);
            }
        }
    }
};

engine_utils::mapFitter make_pointing_fitter() {
    engine_utils::mapFitter fitter;
    fitter.logger = ensure_fitter_test_logger();
    fitter.bounding_box_pix = 30.0;
    fitter.fitting_region_pix = 40.0;
    fitter.flux_low = 0.1;
    fitter.flux_high = 2.0;
    fitter.fwhm_low = 0.1;
    fitter.fwhm_high = 2.0;
    fitter.fit_angle = false;
    return fitter;
}

std::filesystem::path required_environment_path(const char *name) {
    const auto *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        throw std::runtime_error(std::string("missing environment variable ") +
                                 name);
    }
    return value;
}

std::string observation_number(const YAML::Node &input) {
    const auto name = input["meta"]["name"].as<std::string>();
    return name.substr(0, name.find('_'));
}

std::filesystem::path pointing_map_path(
    const std::filesystem::path &long_root,
    const std::filesystem::path &tail_root, const std::string &obsnum,
    const std::string &array_name) {
    const auto relative =
        std::filesystem::path(obsnum) / "raw" /
        ("toltec_commissioning_" + array_name + "_pointing_" + obsnum +
         "_citlali.fits");
    const auto long_path = long_root / relative;
    if (std::filesystem::exists(long_path)) {
        return long_path;
    }
    return tail_root / relative;
}

TEST(MapFitterLifecycle, RepeatedPointingFitsRemainStablePastFailureCount) {
    auto problem = PointingFitProblem{};
    auto fitter = make_pointing_fitter();

    Eigen::VectorXd expected_params;
    for (int invocation = 0; invocation < 512; ++invocation) {
        auto [params, errors, valid] =
            fitter.fit_to_gaussian<engine_utils::mapFitter::pointing>(
                problem.signal, problem.weight, 6.3, -99.0, -99.0);

        ASSERT_TRUE(valid) << "fit invocation " << invocation + 1;
        ASSERT_TRUE(params.array().isFinite().all());
        ASSERT_TRUE(errors.array().isFinite().all());
        if (invocation == 0) {
            expected_params = params;
        }
        else {
            EXPECT_TRUE(params.isApprox(expected_params, 1e-12))
                << "fit invocation " << invocation + 1;
        }
    }
}

// Opt-in forensic replay of the external 2026-07-23 failure corpus. Run with
// --gtest_also_run_disabled_tests and the three CITLALI_POINTING_REPLAY_*
// environment variables documented in the investigation note.
TEST(MapFitterLifecycle, DISABLED_ExactProductSequence) {
    const auto config_path =
        required_environment_path("CITLALI_POINTING_REPLAY_CONFIG");
    const auto long_root =
        required_environment_path("CITLALI_POINTING_REPLAY_LONG_ROOT");
    const auto tail_root =
        required_environment_path("CITLALI_POINTING_REPLAY_TAIL_ROOT");

    const auto config = YAML::LoadFile(config_path.string());
    const auto inputs = config["inputs"];
    ASSERT_TRUE(inputs.IsSequence());
    ASSERT_GE(inputs.size(), 46);

    struct ArrayFit {
        const char *name;
        double initial_fwhm_pixels;
    };
    constexpr std::array<ArrayFit, 3> arrays{{
        {"a1100", 5.0},
        {"a1400", 6.3},
        {"a2000", 9.5},
    }};

    auto fitter = make_pointing_fitter();
    std::size_t invocation = 0;
    for (std::size_t observation = 0; observation < 46; ++observation) {
        const auto obsnum = observation_number(inputs[observation]);
        for (const auto &array : arrays) {
            const auto path = pointing_map_path(
                long_root, tail_root, obsnum, array.name);
            ASSERT_TRUE(std::filesystem::exists(path))
                << "missing replay input " << path;

            fitsIO<file_type_enum::read_fits, CCfits::ExtHDU *> fits(
                path.string());
            auto signal = fits.get_hdu("signal_I");
            auto weight = fits.get_hdu("weight_I");
            auto [params, errors, valid] =
                fitter.fit_to_gaussian<engine_utils::mapFitter::pointing>(
                    signal, weight, array.initial_fwhm_pixels, -99.0, -99.0);

            ++invocation;
            ASSERT_TRUE(valid)
                << "fit invocation " << invocation << " observation "
                << obsnum << " array " << array.name;
            ASSERT_TRUE(params.array().isFinite().all());
            ASSERT_TRUE(errors.array().isFinite().all());
        }
    }
    EXPECT_EQ(invocation, 138);
}

}  // namespace
