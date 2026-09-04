#pragma once

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mapmaking {

inline constexpr const char *jinc_accounting_schema =
    "SCI-FRUIT-EL-F10-JINC-ACCOUNTING-R0.1";

inline std::string jinc_accounting_admission_reason(
    bool final_flag, bool finite_signal, bool usable_coefficient,
    bool center_in_map) {
    if (final_flag) {
        return "final_flagged";
    }
    if (!finite_signal) {
        return "nonfinite_signal";
    }
    if (!usable_coefficient) {
        return "analysis_coefficient_unavailable";
    }
    if (!center_in_map) {
        return "center_outside_map";
    }
    return "admitted";
}

struct JincAccountingSample {
    int scan_index = -1;
    std::int64_t sample_index = -1;
    int array_id = -1;
    int uid = -1;
    double processed_signal = std::numeric_limits<double>::quiet_NaN();
    double analysis_coefficient = std::numeric_limits<double>::quiet_NaN();
    int final_flag = 1;
    int admitted = 0;
    std::string reason;
    double continuous_row = std::numeric_limits<double>::quiet_NaN();
    double continuous_col = std::numeric_limits<double>::quiet_NaN();
    std::int64_t rounded_row = -1;
    std::int64_t rounded_col = -1;
    double row_phase = std::numeric_limits<double>::quiet_NaN();
    double col_phase = std::numeric_limits<double>::quiet_NaN();
    int subpixel_index = -1;
    int center_in_map = 0;
    std::int64_t contributed_pixel_count = 0;
};

class JincAccountingState {
public:
    using CountMatrix =
        Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic>;

    bool configured = false;
    std::string array_name;
    int array_id = -1;
    int uid = -1;
    int scan_index = -1;
    Eigen::Index map_index = -1;
    std::string obsnum;
    int fruit_iteration = -1;
    int subpixel_n = 1;
    double r_max = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> kernel_shape_params;

    Eigen::MatrixXd total_n;
    Eigen::MatrixXd total_c;
    Eigen::MatrixXd total_q;
    Eigen::MatrixXd target_n;
    Eigen::MatrixXd target_c;
    Eigen::MatrixXd target_q;
    Eigen::MatrixXd total_abs_n;
    Eigen::MatrixXd total_abs_c;
    Eigen::MatrixXd target_abs_n;
    Eigen::MatrixXd target_abs_c;
    CountMatrix total_occurrence_count;
    CountMatrix target_occurrence_count;
    CountMatrix total_unique_detector_count;
    CountMatrix target_unique_detector_count;
    Eigen::MatrixXd formal_coefficient;
    Eigen::MatrixXd empirical_coefficient;
    CountMatrix normalization_support;
    CountMatrix science_policy_support;
    double normalization_threshold =
        std::numeric_limits<double>::quiet_NaN();
    double science_policy_threshold =
        std::numeric_limits<double>::quiet_NaN();
    double empirical_scale = std::numeric_limits<double>::quiet_NaN();
    bool totals_captured = false;
    bool normalization_captured = false;
    bool finalization_captured = false;
    std::vector<JincAccountingSample> target_samples;

    bool enabled() const noexcept { return configured; }

    void clear() {
        *this = JincAccountingState{};
    }

    void configure(std::string target_array_name, int target_array_id,
                   int target_uid, int target_scan_index,
                   Eigen::Index target_map_index, std::string target_obsnum,
                   int target_fruit_iteration, int target_subpixel_n,
                   double target_r_max,
                   std::vector<double> target_kernel_shape_params,
                   Eigen::Index rows,
                   Eigen::Index cols) {
        if (rows <= 0 || cols <= 0 || target_map_index < 0 ||
            target_array_id < 0 || target_uid < 0 ||
            target_scan_index < 0) {
            throw std::runtime_error(
                "invalid JINC accounting target or map geometry");
        }
        clear();
        configured = true;
        array_name = std::move(target_array_name);
        array_id = target_array_id;
        uid = target_uid;
        scan_index = target_scan_index;
        map_index = target_map_index;
        obsnum = std::move(target_obsnum);
        fruit_iteration = target_fruit_iteration;
        subpixel_n = target_subpixel_n;
        r_max = target_r_max;
        kernel_shape_params = std::move(target_kernel_shape_params);
        const auto zero = Eigen::MatrixXd::Zero(rows, cols);
        target_n = zero;
        target_c = zero;
        target_q = zero;
        total_abs_n = zero;
        total_abs_c = zero;
        target_abs_n = zero;
        target_abs_c = zero;
        total_occurrence_count = CountMatrix::Zero(rows, cols);
        target_occurrence_count = CountMatrix::Zero(rows, cols);
        total_unique_detector_count = CountMatrix::Zero(rows, cols);
        target_unique_detector_count = CountMatrix::Zero(rows, cols);
    }

    template <class Apt>
    void prepare_uid_inventory(const Apt &apt) {
        if (!configured) {
            return;
        }
        const auto array_it = apt.find("array");
        const auto uid_it = apt.find("uid");
        if (array_it == apt.end() || uid_it == apt.end()) {
            throw std::runtime_error(
                "JINC accounting requires APT array and uid columns");
        }
        const Eigen::Index n = std::min<Eigen::Index>(
            array_it->second.size(), uid_it->second.size());
        const auto old_words_per_pixel = words_per_pixel_;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (!std::isfinite(array_it->second(i)) ||
                !std::isfinite(uid_it->second(i)) ||
                static_cast<int>(std::llround(array_it->second(i))) !=
                    array_id) {
                continue;
            }
            const int detector_uid =
                static_cast<int>(std::llround(uid_it->second(i)));
            if (uid_slots_.find(detector_uid) == uid_slots_.end()) {
                uid_slots_.emplace(detector_uid, uid_slots_.size());
            }
            target_uid_seen_in_inventory_ =
                target_uid_seen_in_inventory_ || detector_uid == uid;
        }
        words_per_pixel_ = (uid_slots_.size() + 63u) / 64u;
        if (words_per_pixel_ == old_words_per_pixel) {
            return;
        }
        const auto pixels = static_cast<std::size_t>(target_n.size());
        std::vector<std::uint64_t> expanded(
            pixels * words_per_pixel_, 0u);
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            for (std::size_t word = 0; word < old_words_per_pixel; ++word) {
                expanded[pixel * words_per_pixel_ + word] =
                    total_unique_bits_[pixel * old_words_per_pixel + word];
            }
        }
        total_unique_bits_.swap(expanded);
    }

    bool is_target(int detector_uid, int detector_scan_index,
                   int detector_array_id, Eigen::Index detector_map_index) const {
        return configured && detector_uid == uid &&
               detector_scan_index == scan_index &&
               detector_array_id == array_id &&
               detector_map_index == map_index;
    }

    void record_sample(JincAccountingSample sample) {
        if (configured) {
            target_samples.push_back(std::move(sample));
        }
    }

    void record_contribution(Eigen::Index row, Eigen::Index col,
                             double n_term, double c_term, double q_term,
                             int detector_uid, bool target) {
        if (!configured || row < 0 || col < 0 || row >= target_n.rows() ||
            col >= target_n.cols()) {
            return;
        }
        total_abs_n(row, col) += std::abs(n_term);
        total_abs_c(row, col) += std::abs(c_term);
        total_occurrence_count(row, col)++;
        record_unique_detector(row, col, detector_uid);
        if (!target) {
            return;
        }
        target_n(row, col) += n_term;
        target_c(row, col) += c_term;
        target_q(row, col) += q_term;
        target_abs_n(row, col) += std::abs(n_term);
        target_abs_c(row, col) += std::abs(c_term);
        target_occurrence_count(row, col)++;
        target_unique_detector_count(row, col) = 1;
    }

    void capture_totals(const Eigen::MatrixXd &n, const Eigen::MatrixXd &c,
                        const Eigen::MatrixXd &q) {
        if (!configured) {
            return;
        }
        require_shape(n, "N");
        require_shape(c, "C");
        require_shape(q, "Q");
        total_n = n;
        total_c = c;
        total_q = q;
        totals_captured = true;
    }

    void capture_normalization(const Eigen::MatrixXd &coefficient,
                               const Eigen::ArrayXXd &support,
                               double threshold) {
        if (!configured) {
            return;
        }
        require_shape(coefficient, "formal coefficient");
        formal_coefficient = coefficient;
        normalization_support =
            (support > 0.0).template cast<std::int64_t>().matrix();
        normalization_threshold = threshold;
        normalization_captured = true;
    }

    void capture_finalization(const Eigen::MatrixXd &coefficient,
                              const Eigen::ArrayXXd &support,
                              double threshold, double scale) {
        if (!configured) {
            return;
        }
        require_shape(coefficient, "empirical coefficient");
        empirical_coefficient = coefficient;
        science_policy_support =
            (support > 0.0).template cast<std::int64_t>().matrix();
        science_policy_threshold = threshold;
        empirical_scale = scale;
        finalization_captured = true;
    }

    void require_complete() const {
        if (!configured || !totals_captured || !normalization_captured ||
            !finalization_captured || !target_uid_seen_in_inventory_ ||
            target_samples.empty()) {
            throw std::runtime_error(
                "incomplete required JINC accounting receipt");
        }
    }

private:
    std::unordered_map<int, std::size_t> uid_slots_;
    std::size_t words_per_pixel_ = 0;
    std::vector<std::uint64_t> total_unique_bits_;
    bool target_uid_seen_in_inventory_ = false;

    void require_shape(const Eigen::MatrixXd &matrix,
                       const char *name) const {
        if (matrix.rows() != target_n.rows() ||
            matrix.cols() != target_n.cols()) {
            throw std::runtime_error(
                std::string{"JINC accounting shape mismatch for "} + name);
        }
    }

    void record_unique_detector(Eigen::Index row, Eigen::Index col,
                                int detector_uid) {
        const auto slot_it = uid_slots_.find(detector_uid);
        if (slot_it == uid_slots_.end() || words_per_pixel_ == 0) {
            throw std::runtime_error(
                "JINC accounting contribution has an unregistered UID");
        }
        const auto pixel = static_cast<std::size_t>(
            row * target_n.cols() + col);
        const auto slot = slot_it->second;
        const auto word = slot / 64u;
        const std::uint64_t bit = std::uint64_t{1} << (slot % 64u);
        auto &bits = total_unique_bits_.at(pixel * words_per_pixel_ + word);
        if ((bits & bit) == 0u) {
            bits |= bit;
            total_unique_detector_count(row, col)++;
        }
    }
};

}  // namespace mapmaking
