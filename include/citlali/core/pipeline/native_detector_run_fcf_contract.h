#pragma once

// Detector/run-local flux-conversion-factor reconciliation for native RTC
// dispatch. Extinction is evaluated on each packet-contiguous run, so its
// scalar detector metadata is the sample-count-weighted mean across exactly
// those run-local realizations. Non-extinction execution retains the stricter
// exact-equality contract.

#include <citlali/core/pipeline/timestream_measured_scan.h>

#include <Eigen/Core>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace citlali::pipeline {

inline constexpr std::string_view native_detector_run_fcf_contract_v1 =
    "citlali-native-detector-run-local-fcf-contract-v1";

class NativeDetectorRunFcfContract {
public:
    NativeDetectorRunFcfContract(std::size_t detector_count,
                                 bool extinction_active)
        : extinction_active_(extinction_active),
          exact_(static_cast<Eigen::Index>(detector_count)),
          weighted_sum_(detector_count, 0.0),
          compensation_(detector_count, 0.0),
          sample_count_(detector_count, 0) {
        if (detector_count == 0 ||
            detector_count > static_cast<std::size_t>(
                                 std::numeric_limits<Eigen::Index>::max())) {
            throw std::invalid_argument(
                "native detector/run FCF contract has invalid cardinality");
        }
        exact_.setConstant(std::numeric_limits<double>::quiet_NaN());
    }

    void observe(const std::vector<TimestreamDetectorColumn> &detector_columns,
                 const Eigen::VectorXd &run_fcf,
                 std::size_t run_sample_count) {
        if (detector_columns.empty() || run_sample_count == 0 ||
            run_fcf.size() !=
                static_cast<Eigen::Index>(detector_columns.size()) ||
            !run_fcf.array().isFinite().all()) {
            throw std::invalid_argument(
                "native detector/run FCF realization is incomplete");
        }
        for (std::size_t local = 0; local < detector_columns.size();
             ++local) {
            const auto detector = detector_columns[local];
            if (detector < 0 || detector >= exact_.size()) {
                throw std::out_of_range(
                    "native detector/run FCF column is out of range");
            }
            const auto index = static_cast<std::size_t>(detector);
            const auto value = run_fcf(static_cast<Eigen::Index>(local));
            if (run_sample_count >
                std::numeric_limits<std::size_t>::max() -
                    sample_count_[index]) {
                throw std::overflow_error(
                    "native detector/run FCF sample count overflow");
            }
            if (!extinction_active_) {
                if (sample_count_[index] != 0 &&
                    std::bit_cast<std::uint64_t>(exact_(detector)) !=
                        std::bit_cast<std::uint64_t>(value)) {
                    throw std::logic_error(
                        "native detector FCF differs between non-extinction runs");
                }
                exact_(detector) = value;
            }
            else {
                const auto term = value *
                    static_cast<double>(run_sample_count);
                if (!std::isfinite(term)) {
                    throw std::overflow_error(
                        "native detector/run FCF weighted value is nonfinite");
                }
                // Kahan accumulation fixes one deterministic order and limits
                // loss when run lengths differ substantially.
                const auto adjusted = term - compensation_[index];
                const auto next = weighted_sum_[index] + adjusted;
                compensation_[index] =
                    (next - weighted_sum_[index]) - adjusted;
                weighted_sum_[index] = next;
            }
            sample_count_[index] += run_sample_count;
        }
    }

    Eigen::VectorXd finish() const {
        Eigen::VectorXd result(exact_.size());
        for (Eigen::Index detector = 0; detector < exact_.size();
             ++detector) {
            const auto index = static_cast<std::size_t>(detector);
            if (sample_count_[index] == 0) {
                throw std::logic_error(
                    "native detector/run FCF contract missed a detector");
            }
            result(detector) = extinction_active_
                ? weighted_sum_[index] /
                    static_cast<double>(sample_count_[index])
                : exact_(detector);
            if (!std::isfinite(result(detector))) {
                throw std::logic_error(
                    "native detector/run FCF contract produced a nonfinite value");
            }
        }
        return result;
    }

private:
    bool extinction_active_ = false;
    Eigen::VectorXd exact_;
    std::vector<double> weighted_sum_;
    std::vector<double> compensation_;
    std::vector<std::size_t> sample_count_;
};

}  // namespace citlali::pipeline
