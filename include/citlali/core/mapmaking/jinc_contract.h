#pragma once

#include <citlali/core/utils/sha256.h>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace mapmaking {

inline constexpr std::string_view jinc_contract_version =
    "SCI-MAP-002-v1";
inline constexpr std::string_view jinc_estimator_identity =
    "signed-N-over-C-formal-C2-over-Q-v1";
inline constexpr std::string_view jinc_support_identity =
    "fully-populated-square-cache-dual-use-r-max-v1";
inline constexpr std::string_view jinc_phase_identity =
    "rounded-center-residual-phase-quantized-point-sampling-v1";
inline constexpr std::string_view jinc_summation_identity =
    "naive-binary64-two-level-2gamma-n-sumabs-v1";
inline constexpr std::string_view jinc_conditioning_identity =
    "finite-positive-Q-exact-cancellation-and-rho-resolution-v1";
inline constexpr std::string_view jinc_formal_support_identity =
    "finite-signal-finite-positive-formal-weight-admitted-conditioned-v1";
inline constexpr std::string_view jinc_coverage_identity =
    "coefficient-squared-effective-integration-time-v1";
inline constexpr std::string_view jinc_kernel_identity =
    "processing-filtered-source-template-response-projected-through-JINC-v1";
inline constexpr std::string_view jinc_product_digest_identity =
    "canonical-column-major-hexfloat-sha256-v1";

inline std::optional<std::string_view> stable_jinc_array_name(
    Eigen::Index array_id) {
    switch (array_id) {
        case 0:
            return "a1100";
        case 1:
            return "a1400";
        case 2:
            return "a2000";
        default:
            return std::nullopt;
    }
}

inline double jinc_gamma(std::size_t contributor_count) {
    if (contributor_count == 0) {
        return 0.0;
    }
    const long double n = static_cast<long double>(contributor_count);
    const long double epsilon =
        static_cast<long double>(std::numeric_limits<double>::epsilon());
    const long double product = n * epsilon;
    if (!(product < 1.0L)) {
        return std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(product / (1.0L - product));
}

// JINC accumulation has one detector/sample-ordered accumulation into a
// scratch plane followed by one serialized scratch-to-live merge.  The
// conservative two-level resolution bound is therefore 2*gamma_n.
inline double jinc_rho_resolution_bound(std::size_t contributor_count) {
    const double gamma = jinc_gamma(contributor_count);
    if (!std::isfinite(gamma) || gamma > 0.5) {
        return 1.0;
    }
    return std::min(1.0, 2.0 * gamma);
}

inline int jinc_phase_bin(double residual_phase, int subpixel_n) {
    if (!std::isfinite(residual_phase) || subpixel_n < 1 ||
        residual_phase < -0.5 || residual_phase > 0.5) {
        throw std::invalid_argument("invalid JINC residual phase or subpixel_n");
    }
    int index = static_cast<int>(
        std::floor((residual_phase + 0.5) * subpixel_n));
    return std::clamp(index, 0, subpixel_n - 1);
}

struct JincSquareCrop {
    Eigen::Index map_row = 0;
    Eigen::Index map_col = 0;
    Eigen::Index cache_row = 0;
    Eigen::Index cache_col = 0;
    Eigen::Index rows = 0;
    Eigen::Index cols = 0;
};

// Crop only at the rectangular map boundary.  This deliberately contains no
// radial predicate: every coefficient in the square cache remains eligible.
inline JincSquareCrop jinc_square_crop(
    Eigen::Index map_rows, Eigen::Index map_cols, Eigen::Index center_row,
    Eigen::Index center_col, Eigen::Index cache_rows,
    Eigen::Index cache_cols) {
    if (map_rows <= 0 || map_cols <= 0 || cache_rows <= 0 ||
        cache_cols <= 0 || cache_rows % 2 == 0 || cache_cols % 2 == 0 ||
        center_row < 0 || center_row >= map_rows || center_col < 0 ||
        center_col >= map_cols) {
        throw std::invalid_argument("invalid JINC square-cache crop geometry");
    }
    const Eigen::Index half_rows = (cache_rows - 1) / 2;
    const Eigen::Index half_cols = (cache_cols - 1) / 2;
    const Eigen::Index unclipped_row = center_row - half_rows;
    const Eigen::Index unclipped_col = center_col - half_cols;
    const Eigen::Index map_row = std::max<Eigen::Index>(0, unclipped_row);
    const Eigen::Index map_col = std::max<Eigen::Index>(0, unclipped_col);
    const Eigen::Index map_row_end = std::min(
        map_rows - 1, center_row + cache_rows - 1 - half_rows);
    const Eigen::Index map_col_end = std::min(
        map_cols - 1, center_col + cache_cols - 1 - half_cols);
    return JincSquareCrop{
        map_row,
        map_col,
        std::abs(std::min<Eigen::Index>(0, unclipped_row)),
        std::abs(std::min<Eigen::Index>(0, unclipped_col)),
        map_row_end - map_row + 1,
        map_col_end - map_col + 1,
    };
}

struct JincResolvedArrayParameters {
    Eigen::Index array_id = -1;
    std::string array_name;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double r_max = 0.0;
    double pixel_size_rad = 0.0;
    double array_scale_rad = 0.0;
    Eigen::Index cache_half_width_pixels = -1;
    Eigen::Index cache_rows = 0;
    Eigen::Index cache_cols = 0;
};

inline void validate_jinc_resolved_array(
    const JincResolvedArrayParameters &parameters) {
    const auto expected_name = stable_jinc_array_name(parameters.array_id);
    if (!expected_name || parameters.array_name != *expected_name) {
        throw std::invalid_argument(
            "JINC selected array lacks stable canonical identity");
    }
    const std::array<double, 7> positive{
        parameters.a, parameters.b, parameters.c, parameters.r_max,
        parameters.pixel_size_rad, parameters.array_scale_rad,
        static_cast<double>(parameters.cache_rows)};
    if (std::any_of(positive.begin(), positive.end(), [](double value) {
            return !std::isfinite(value) || value <= 0.0;
        }) ||
        parameters.cache_half_width_pixels < 0 ||
        parameters.cache_cols <= 0 ||
        parameters.cache_rows != parameters.cache_cols ||
        parameters.cache_rows !=
            2 * parameters.cache_half_width_pixels + 1) {
        throw std::invalid_argument(
            "JINC selected array has nonphysical resolved parameters");
    }
}

struct JincConditioningResult {
    bool accumulators_finite = false;
    bool q_positive = false;
    bool exact_cancellation = false;
    bool numerically_resolved = false;
    bool formal_support = false;
    double rho = 0.0;
    double rho_resolution_bound = 0.0;
    double signal = 0.0;
    double formal_weight = 0.0;
};

inline JincConditioningResult finalize_jinc_accumulators(
    double numerator, double denominator, double variance_accumulator,
    double denominator_sum_abs, std::size_t contributor_count) {
    JincConditioningResult result;
    result.rho_resolution_bound =
        jinc_rho_resolution_bound(contributor_count);
    result.accumulators_finite =
        contributor_count > 0 && std::isfinite(numerator) &&
        std::isfinite(denominator) &&
        std::isfinite(variance_accumulator) &&
        std::isfinite(denominator_sum_abs) && denominator_sum_abs >= 0.0;
    if (!result.accumulators_finite) {
        return result;
    }
    result.q_positive = variance_accumulator > 0.0;
    result.exact_cancellation = denominator == 0.0;
    if (denominator_sum_abs > 0.0) {
        result.rho = std::abs(denominator) / denominator_sum_abs;
    }
    result.numerically_resolved =
        !result.exact_cancellation && denominator_sum_abs > 0.0 &&
        std::isfinite(result.rho) &&
        result.rho >= result.rho_resolution_bound;
    if (!result.q_positive || !result.numerically_resolved) {
        return result;
    }
    result.signal = numerator / denominator;
    const double normalized =
        denominator / std::sqrt(variance_accumulator);
    result.formal_weight = normalized * normalized;
    result.formal_support =
        std::isfinite(result.signal) &&
        std::isfinite(result.formal_weight) && result.formal_weight > 0.0;
    return result;
}

inline bool jinc_empirical_support(bool formal_support,
                                   bool empirical_admission) {
    return formal_support && empirical_admission;
}

using JincCountPlane =
    Eigen::Matrix<std::uint64_t, Eigen::Dynamic, Eigen::Dynamic>;
using JincMaskPlane =
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>;

struct JincProductJoin {
    std::string product_identity;
    std::string product_scope;
    std::string output_file;
    std::string hdu_name;
    std::string content_digest;
};

struct JincMapRealizedSummary {
    bool realized = false;
    std::size_t realization_pass = 0;
    std::size_t total_pixel_count = 0;
    std::size_t formally_supported_pixel_count = 0;
    std::size_t exact_cancellation_pixel_count = 0;
    std::size_t unresolved_cancellation_pixel_count = 0;
    std::size_t invalid_q_pixel_count = 0;
    std::size_t nonfinite_accumulator_pixel_count = 0;
    std::size_t contributor_count_max = 0;
    double rho_resolution_bound_max = 0.0;
};

struct JincRealizedSummary {
    std::size_t map_count = 0;
    std::size_t realized_map_count = 0;
    std::size_t realization_pass_count = 0;
    std::vector<std::size_t> last_pass_active_map_indices;
    std::size_t total_pixel_count = 0;
    std::size_t formally_supported_pixel_count = 0;
    std::size_t exact_cancellation_pixel_count = 0;
    std::size_t unresolved_cancellation_pixel_count = 0;
    std::size_t invalid_q_pixel_count = 0;
    std::size_t nonfinite_accumulator_pixel_count = 0;
    std::size_t contributor_count_max = 0;
    double rho_resolution_bound_max = 0.0;
    std::string summation_method{jinc_summation_identity};
    std::string conditioning_policy{jinc_conditioning_identity};
    std::vector<JincMapRealizedSummary> map_summaries;
    std::vector<JincProductJoin> product_joins;
};

struct JincObservationProvenance {
    bool available = false;
    std::string requested_digest;
    std::string effective_digest;
    double requested_r_max = 0.0;
    double effective_r_max = 0.0;
    int requested_subpixel_n = 1;
    int effective_subpixel_n = 1;
    std::map<std::string, std::array<double, 3>> requested_shape_params;
    std::map<std::string, std::array<double, 3>> effective_shape_params;
    std::vector<JincResolvedArrayParameters> resolved_arrays;
    std::string support_convention{jinc_support_identity};
    std::string phase_convention{jinc_phase_identity};
    std::string estimator{jinc_estimator_identity};
    std::string formal_support_policy{jinc_formal_support_identity};
    std::string coverage_estimator{jinc_coverage_identity};
    std::string kernel_response{jinc_kernel_identity};
    std::string kernel_template_identity = "unavailable";
    std::string processing_configuration_identity = "unavailable";
    std::string processing_realization_identity = "unavailable";
    bool processing_configuration_bound = false;
    bool processing_realization_bound = false;
    std::vector<std::pair<std::string, std::string>>
        processing_configuration_facts;
    std::vector<std::pair<std::string, std::string>>
        processing_realization_facts;
    std::string coverage_sample_frequency_identity = "unavailable";
    double coverage_sample_frequency_hz = 0.0;
    JincRealizedSummary realized;
};

struct JincProcessingScanTrace {
    std::size_t detector_count = 0;
    std::size_t detector_sample_count = 0;
    std::size_t rtc_flagged_sample_count = 0;
    std::size_t ptc_flagged_sample_count = 0;
    std::size_t apt_flagged_detector_count = 0;
    std::size_t rtc_source_masked_sample_count = 0;
    std::size_t ptc_mean_masked_sample_count = 0;
    std::size_t pca_solve_count = 0;
    std::size_t dynamic_notch_count = 0;
    std::size_t detector_notch_count = 0;
    std::string rtc_flags_digest = "unavailable";
    std::string ptc_flags_digest = "unavailable";
    std::string apt_flags_digest = "unavailable";
    std::string map_indices_digest = "unavailable";
    std::string ptc_signal_digest = "unavailable";
    std::string ptc_kernel_digest = "unavailable";
    std::string ptc_mean_mask_digest = "unavailable";
    std::string pca_realization_digest = "unavailable";
};

struct JincProducts {
    bool initialized = false;
    std::vector<Eigen::MatrixXd> denominator_sum_abs;
    std::vector<JincCountPlane> contributor_count;
    std::vector<JincMaskPlane> formal_support;
    JincObservationProvenance provenance;
    std::map<Eigen::Index, JincProcessingScanTrace> processing_scan_traces;
    std::shared_ptr<std::mutex> processing_trace_mutex =
        std::make_shared<std::mutex>();

    void clear() {
        initialized = false;
        denominator_sum_abs.clear();
        contributor_count.clear();
        formal_support.clear();
        provenance = {};
        processing_scan_traces.clear();
    }

    void allocate(Eigen::Index map_count, Eigen::Index rows,
                  Eigen::Index cols) {
        clear();
        if (map_count <= 0 || rows <= 0 || cols <= 0) {
            throw std::invalid_argument(
                "JINC product allocation requires positive shape");
        }
        initialized = true;
        const auto count = static_cast<std::size_t>(map_count);
        denominator_sum_abs.assign(
            count, Eigen::MatrixXd::Zero(rows, cols));
        contributor_count.assign(
            count, JincCountPlane::Zero(rows, cols));
        formal_support.assign(count, JincMaskPlane::Zero(rows, cols));
        provenance.available = true;
        provenance.realized.map_count = count;
        provenance.realized.map_summaries.assign(
            count, JincMapRealizedSummary{});
    }
};

inline std::string jinc_double_hex(double value) {
    std::ostringstream stream;
    stream << std::hexfloat << value;
    return stream.str();
}

template <class FilterConfig>
std::string jinc_filter_config_digest(const FilterConfig &config) {
    citlali::utils::Sha256 digest;
    auto add = [&](std::string_view value) {
        digest.update(std::to_string(value.size()));
        digest.update(":");
        digest.update(value);
        digest.update(";");
    };
    add(std::string{jinc_contract_version});
    add(jinc_double_hex(config.r_max));
    add(std::to_string(config.subpixel_n));
    for (const auto &[array_name, shape] : config.shape_params) {
        add(array_name);
        for (const auto value : shape) {
            add(jinc_double_hex(value));
        }
    }
    return "sha256:" + digest.finish();
}

inline void record_jinc_product_join(
    JincObservationProvenance &provenance, JincProductJoin join) {
    if (!provenance.available || join.product_identity.empty() ||
        join.product_scope.empty() || join.output_file.empty() ||
        join.hdu_name.empty() || join.content_digest.empty()) {
        throw std::logic_error(
            "JINC product join requires complete immutable identity");
    }
    const auto duplicate = std::find_if(
        provenance.realized.product_joins.begin(),
        provenance.realized.product_joins.end(), [&](const auto &existing) {
            return existing.output_file == join.output_file &&
                   existing.hdu_name == join.hdu_name;
        });
    if (duplicate != provenance.realized.product_joins.end()) {
        if (duplicate->product_identity != join.product_identity ||
            duplicate->product_scope != join.product_scope ||
            duplicate->content_digest != join.content_digest) {
            throw std::logic_error(
                "JINC product join cannot be changed after publication");
        }
        return;
    }
    auto next = provenance.realized.product_joins;
    next.push_back(std::move(join));
    provenance.realized.product_joins.swap(next);
}

template <class Matrix>
std::string jinc_matrix_digest(const Matrix &matrix) {
    citlali::utils::Sha256 digest;
    auto add = [&](const std::string &value) {
        digest.update(std::to_string(value.size()));
        digest.update(":");
        digest.update(value);
        digest.update(";");
    };
    add(std::string{jinc_product_digest_identity});
    add(std::to_string(matrix.rows()));
    add(std::to_string(matrix.cols()));
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
        for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
            using Scalar = std::remove_cv_t<typename Matrix::Scalar>;
            if constexpr (std::is_integral_v<Scalar>) {
                add(std::to_string(matrix(row, col)));
            }
            else {
                add(jinc_double_hex(static_cast<double>(matrix(row, col))));
            }
        }
    }
    return "sha256:" + digest.finish();
}

inline std::string jinc_realization_identity_digest(
    std::string_view identity,
    const std::vector<std::pair<std::string, std::string>> &ordered_facts) {
    citlali::utils::Sha256 digest;
    auto add = [&](std::string_view value) {
        digest.update(std::to_string(value.size()));
        digest.update(":");
        digest.update(value);
        digest.update(";");
    };
    add(std::string{jinc_contract_version});
    add(identity);
    for (const auto &[name, value] : ordered_facts) {
        add(name);
        add(value);
    }
    return std::string{identity} + ":sha256:" + digest.finish();
}

inline std::string jinc_processing_realization_identity(
    const std::string &configuration_identity, bool execution_completed,
    std::optional<std::size_t> completed_scan_count,
    std::optional<std::size_t> dynamic_notch_count) {
    const auto optional_text = [](const auto &value) {
        return value ? std::to_string(*value)
                     : std::string{"unavailable"};
    };
    return jinc_realization_identity_digest(
        "actual-processing-realization-v2",
        {{"configuration_identity", configuration_identity},
         {"raw_execution_completed",
          execution_completed ? "true" : "false"},
         {"completed_scan_count", optional_text(completed_scan_count)},
         {"dynamic_notch_count", optional_text(dynamic_notch_count)}});
}

inline bool jinc_processing_provenance_complete(
    const JincObservationProvenance &provenance) {
    return provenance.available &&
           provenance.processing_configuration_bound &&
           provenance.processing_realization_bound &&
           provenance.kernel_template_identity != "unavailable" &&
           provenance.processing_configuration_identity != "unavailable" &&
           provenance.processing_realization_identity != "unavailable" &&
           !provenance.processing_configuration_facts.empty() &&
           !provenance.processing_realization_facts.empty() &&
           provenance.coverage_sample_frequency_identity != "unavailable" &&
           std::isfinite(provenance.coverage_sample_frequency_hz) &&
           provenance.coverage_sample_frequency_hz > 0.0;
}

template <class Kernel>
std::string jinc_kernel_template_identity(const Kernel &kernel,
                                          bool enabled) {
    std::vector<std::pair<std::string, std::string>> facts{
        {"enabled", enabled ? "true" : "false"},
        {"type", kernel.type},
    };
    if (enabled) {
        std::ostringstream extensions;
        for (const auto &name : kernel.img_ext_names) {
            if (extensions.tellp() > 0) {
                extensions << ',';
            }
            extensions << name;
        }
        facts.insert(
            facts.end(),
            {{"filepath", kernel.filepath},
             {"fwhm_rad", jinc_double_hex(kernel.fwhm_rad)},
             {"sigma_rad", jinc_double_hex(kernel.sigma_rad)},
             {"sigma_limit", jinc_double_hex(kernel.sigma_limit)},
             {"map_grouping", kernel.map_grouping},
             {"image_extensions", extensions.str()},
             {"source_lat_digest", jinc_matrix_digest(kernel.source_lat)},
             {"source_lon_digest", jinc_matrix_digest(kernel.source_lon)},
             {"source_a_fwhm_digest",
              jinc_matrix_digest(kernel.source_a_fwhm_rad)},
             {"source_b_fwhm_digest",
              jinc_matrix_digest(kernel.source_b_fwhm_rad)},
             {"source_valid_digest",
              jinc_matrix_digest(kernel.source_valid)}});
        for (std::size_t image = 0; image < kernel.images.size(); ++image) {
            facts.emplace_back(
                "image_" + std::to_string(image) + "_digest",
                jinc_matrix_digest(kernel.images[image]));
        }
    }
    return jinc_realization_identity_digest(
        "actual-upstream-kernel-template-v1", facts);
}

inline bool jinc_formal_support_exact(
    const JincProducts &products, std::size_t slot, double signal,
    double formal_weight, bool admission_and_conditioning_passed) {
    return products.initialized && slot < products.formal_support.size() &&
           admission_and_conditioning_passed && std::isfinite(signal) &&
           std::isfinite(formal_weight) && formal_weight > 0.0;
}

}  // namespace mapmaking
