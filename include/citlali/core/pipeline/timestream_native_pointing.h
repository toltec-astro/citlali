#pragma once

#include <citlali/core/config/calibration_config.h>
#include <citlali/core/pipeline/timestream_native_alignment.h>

#include <Eigen/Core>
#include <tula/algorithm/mlinterp/mlinterp.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

using NativeTelescopeData = std::map<std::string, Eigen::VectorXd>;
using NativePointingOffsetsArcsec =
    std::map<std::string, Eigen::VectorXd>;

namespace native_pointing_detail {

inline void require_finite_vector(const Eigen::VectorXd &values,
                                  const std::string &label) {
    if (values.size() <= 0 || !values.array().isFinite().all()) {
        throw std::invalid_argument(label + " must be nonempty and finite");
    }
}

inline void require_strictly_increasing(const Eigen::VectorXd &values,
                                        const std::string &label) {
    require_finite_vector(values, label);
    for (Eigen::Index index = 1; index < values.size(); ++index) {
        if (!(values(index) > values(index - 1))) {
            throw std::invalid_argument(label + " must increase strictly");
        }
    }
}

inline bool exactly_equal(const Eigen::VectorXd &lhs,
                          const Eigen::VectorXd &rhs) noexcept {
    return lhs.size() == rhs.size() &&
           (lhs.array() == rhs.array()).all();
}

}  // namespace native_pointing_detail

// Immutable copy of the measured telescope trajectory before legacy
// common-slot interpolation.  TelTime is telescope support, not detector time.
class RawTelescopeTrajectory {
public:
    explicit RawTelescopeTrajectory(NativeTelescopeData telescope_data)
        : telescope_data_{std::move(telescope_data)} {
        const auto time = telescope_data_.find("TelTime");
        if (time == telescope_data_.end()) {
            throw std::invalid_argument(
                "raw telescope trajectory requires TelTime");
        }
        native_pointing_detail::require_strictly_increasing(
            time->second, "raw telescope TelTime");
        if (time->second.size() < 2) {
            throw std::invalid_argument(
                "raw telescope trajectory requires two support samples");
        }
        for (const auto &[name, values] : telescope_data_) {
            if (values.size() != time->second.size() ||
                !values.array().isFinite().all()) {
                throw std::invalid_argument(
                    "raw telescope series '" + name +
                    "' must be finite and match TelTime");
            }
        }
        for (const auto *required : {"TelAzAct", "TelElAct"}) {
            if (!telescope_data_.contains(required)) {
                throw std::invalid_argument(
                    std::string{"raw telescope trajectory lacks "} +
                    required);
            }
        }
    }

    const NativeTelescopeData &telescope_data() const noexcept {
        return telescope_data_;
    }
    const Eigen::VectorXd &support_times_unix_sec() const noexcept {
        return telescope_data_.at("TelTime");
    }
    double support_start_unix_sec() const noexcept {
        return support_times_unix_sec()(0);
    }
    double support_end_unix_sec() const noexcept {
        const auto &times = support_times_unix_sec();
        return times(times.size() - 1);
    }

private:
    NativeTelescopeData telescope_data_;
};

inline NativeTelescopeData evaluate_raw_telescope_trajectory_at(
    const RawTelescopeTrajectory &trajectory,
    const Eigen::VectorXd &target_times_unix_sec) {
    native_pointing_detail::require_strictly_increasing(
        target_times_unix_sec, "native telescope target times");
    if (target_times_unix_sec(0) < trajectory.support_start_unix_sec() ||
        target_times_unix_sec(target_times_unix_sec.size() - 1) >
            trajectory.support_end_unix_sec()) {
        throw std::out_of_range(
            "native telescope target is outside measured support");
    }

    const auto &raw = trajectory.telescope_data();
    const auto &support = trajectory.support_times_unix_sec();
    Eigen::Matrix<Eigen::Index, 1, 1> support_shape;
    support_shape << support.size();
    NativeTelescopeData result;
    for (const auto &[name, source] : raw) {
        if (name == "TelTime" || name == "TelUTC") continue;
        Eigen::VectorXd evaluated(target_times_unix_sec.size());
        mlinterp::interp(
            support_shape.data(), evaluated.size(), source.data(),
            evaluated.data(), support.data(),
            target_times_unix_sec.data());
        if (!evaluated.array().isFinite().all()) {
            throw std::logic_error(
                "native telescope interpolation produced nonfinite '" +
                name + "'");
        }
        result.emplace(name, std::move(evaluated));
    }
    result["TelTime"] = target_times_unix_sec;
    result["TelUTC"] = target_times_unix_sec;
    return result;
}

// One- or two-value observation offset model.  Values are evaluated at each
// network's native timestamps and retain arcsecond units.
class NativePointingOffsetModel {
public:
    NativePointingOffsetModel(
        NativePointingOffsetsArcsec source_arcsec,
        Eigen::VectorXd support_times_unix_sec)
        : source_arcsec_{std::move(source_arcsec)},
          support_times_unix_sec_{std::move(support_times_unix_sec)} {
        const auto az =
            source_arcsec_.find(citlali::config::pointing_axis_az());
        const auto alt =
            source_arcsec_.find(citlali::config::pointing_axis_alt());
        if (az == source_arcsec_.end() || alt == source_arcsec_.end()) {
            throw std::invalid_argument(
                "native pointing offsets require az and alt");
        }
        source_value_count_ = az->second.size();
        if (source_value_count_ != alt->second.size() ||
            (source_value_count_ != 1 && source_value_count_ != 2) ||
            !az->second.array().isFinite().all() ||
            !alt->second.array().isFinite().all()) {
            throw std::invalid_argument(
                "native pointing offsets require equal finite one- or two-value axes");
        }
        if (support_times_unix_sec_.size() != 2 ||
            !support_times_unix_sec_.array().isFinite().all() ||
            !(support_times_unix_sec_(1) >
              support_times_unix_sec_(0))) {
            throw std::invalid_argument(
                "native pointing offsets require increasing two-point support");
        }
    }

    const Eigen::VectorXd &support_times_unix_sec() const noexcept {
        return support_times_unix_sec_;
    }
    Eigen::Index source_value_count() const noexcept {
        return source_value_count_;
    }

    NativePointingOffsetsArcsec evaluate_at(
        const Eigen::VectorXd &target_times_unix_sec) const {
        native_pointing_detail::require_strictly_increasing(
            target_times_unix_sec, "native pointing-offset target times");
        // A single configured value is a constant observation model, not an
        // interpolation model. Exact native detector samples may extend past
        // the legacy common-slot grid used to supply these nominal bounds;
        // the constant remains defined there. Two-value models retain strict
        // no-extrapolation support.
        if (source_value_count_ == 2 &&
            (target_times_unix_sec(0) < support_times_unix_sec_(0) ||
             target_times_unix_sec(target_times_unix_sec.size() - 1) >
                 support_times_unix_sec_(1))) {
            throw std::out_of_range(
                "native pointing-offset target is outside support");
        }

        NativePointingOffsetsArcsec result;
        for (const auto *axis : {citlali::config::pointing_axis_az(),
                                 citlali::config::pointing_axis_alt()}) {
            const auto &source = source_arcsec_.at(axis);
            Eigen::VectorXd evaluated(target_times_unix_sec.size());
            if (source_value_count_ == 1) {
                evaluated.setConstant(source(0));
            }
            else {
                Eigen::Matrix<Eigen::Index, 1, 1> source_shape;
                source_shape << source_value_count_;
                mlinterp::interp(
                    source_shape.data(), evaluated.size(), source.data(),
                    evaluated.data(), support_times_unix_sec_.data(),
                    target_times_unix_sec.data());
            }
            if (!evaluated.array().isFinite().all()) {
                throw std::logic_error(
                    "native pointing-offset interpolation produced nonfinite values");
            }
            result.emplace(axis, std::move(evaluated));
        }
        return result;
    }

private:
    NativePointingOffsetsArcsec source_arcsec_;
    Eigen::VectorXd support_times_unix_sec_;
    Eigen::Index source_value_count_ = 0;
};

class NativeNetworkPointing {
public:
    NativeNetworkPointing(
        TimestreamNetworkId network_id,
        TimestreamNativeRow first_native_row,
        Eigen::VectorXd reconstructed_times_unix_sec,
        NativeTelescopeData telescope_data,
        NativePointingOffsetsArcsec pointing_offsets_arcsec)
        : network_id_{network_id}, first_native_row_{first_native_row},
          reconstructed_times_unix_sec_{
              std::move(reconstructed_times_unix_sec)},
          telescope_data_{std::move(telescope_data)},
          pointing_offsets_arcsec_{std::move(pointing_offsets_arcsec)} {
        if (network_id_ < 0 || first_native_row_ < 0) {
            throw std::invalid_argument(
                "native network pointing requires nonnegative identity");
        }
        native_pointing_detail::require_strictly_increasing(
            reconstructed_times_unix_sec_,
            "native network pointing times");
        if (reconstructed_times_unix_sec_.size() >
            std::numeric_limits<TimestreamNativeRow>::max() -
                first_native_row_) {
            throw std::length_error(
                "native pointing row interval would overflow");
        }
        const auto tel_time = telescope_data_.find("TelTime");
        if (tel_time == telescope_data_.end() ||
            !native_pointing_detail::exactly_equal(
                tel_time->second, reconstructed_times_unix_sec_)) {
            throw std::invalid_argument(
                "native telescope TelTime must exactly equal detector target times");
        }
        for (const auto &[name, values] : telescope_data_) {
            require_series(values, "native telescope series '" + name + "'");
        }
        for (const auto *required : {"TelAzAct", "TelElAct"}) {
            if (!telescope_data_.contains(required)) {
                throw std::invalid_argument(
                    std::string{"native telescope pointing lacks "} +
                    required);
            }
        }
        for (const auto *axis : {citlali::config::pointing_axis_az(),
                                 citlali::config::pointing_axis_alt()}) {
            const auto found = pointing_offsets_arcsec_.find(axis);
            if (found == pointing_offsets_arcsec_.end()) {
                throw std::invalid_argument(
                    "native pointing carrier lacks an offset axis");
            }
            require_series(found->second,
                           std::string{"native pointing offset '"} + axis +
                               "'");
        }
    }

    TimestreamNetworkId network_id() const noexcept { return network_id_; }
    TimestreamNativeRow first_native_row() const noexcept {
        return first_native_row_;
    }
    TimestreamNativeRow past_last_native_row() const noexcept {
        return first_native_row_ + static_cast<TimestreamNativeRow>(
            reconstructed_times_unix_sec_.size());
    }
    Eigen::Index row_count() const noexcept {
        return reconstructed_times_unix_sec_.size();
    }
    const Eigen::VectorXd &reconstructed_times_unix_sec() const noexcept {
        return reconstructed_times_unix_sec_;
    }
    NativeSampleIdentity identity(TimestreamNativeRow native_row) const {
        const auto row = local_row(native_row);
        return NativeSampleIdentity{
            network_id_, native_row,
            reconstructed_times_unix_sec_(row)};
    }
    Eigen::Index local_row(TimestreamNativeRow native_row) const {
        if (native_row < first_native_row_ ||
            native_row >= past_last_native_row()) {
            throw std::out_of_range(
                "native row is outside the pointing carrier");
        }
        return static_cast<Eigen::Index>(native_row - first_native_row_);
    }
    const Eigen::VectorXd &telescope_series(const std::string &name) const {
        return telescope_data_.at(name);
    }
    const Eigen::VectorXd &pointing_offset_arcsec(
        const std::string &axis) const {
        return pointing_offsets_arcsec_.at(axis);
    }
    const NativeTelescopeData &telescope_data() const noexcept {
        return telescope_data_;
    }
    const NativePointingOffsetsArcsec &pointing_offsets_arcsec() const
        noexcept {
        return pointing_offsets_arcsec_;
    }

private:
    void require_series(const Eigen::VectorXd &values,
                        const std::string &label) const {
        if (values.size() != reconstructed_times_unix_sec_.size() ||
            !values.array().isFinite().all()) {
            throw std::invalid_argument(
                label + " must be finite and match native rows");
        }
    }

    TimestreamNetworkId network_id_;
    TimestreamNativeRow first_native_row_;
    Eigen::VectorXd reconstructed_times_unix_sec_;
    NativeTelescopeData telescope_data_;
    NativePointingOffsetsArcsec pointing_offsets_arcsec_;
};

class NativePointingPlan {
public:
    NativePointingPlan(
        std::shared_ptr<const NativeAlignmentPlan> alignment_plan,
        std::shared_ptr<const RawTelescopeTrajectory> raw_trajectory,
        std::vector<NativeNetworkPointing> networks)
        : alignment_plan_{std::move(alignment_plan)},
          raw_trajectory_{std::move(raw_trajectory)},
          networks_{std::move(networks)} {
        if (!alignment_plan_ || !raw_trajectory_) {
            throw std::invalid_argument(
                "native pointing plan requires alignment and telescope handles");
        }
        std::sort(networks_.begin(), networks_.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.network_id() < rhs.network_id();
                  });
        if (networks_.size() != alignment_plan_->networks().size()) {
            throw std::invalid_argument(
                "native pointing and alignment network counts differ");
        }
        for (std::size_t index = 0; index < networks_.size(); ++index) {
            const auto &pointing = networks_[index];
            const auto &alignment =
                alignment_plan_->networks().at(index);
            if (pointing.network_id() != alignment.network_id() ||
                pointing.first_native_row() !=
                    alignment.first_native_row() ||
                pointing.past_last_native_row() !=
                    alignment.past_last_native_row()) {
                throw std::invalid_argument(
                    "native pointing interval differs from alignment authority");
            }
            for (TimestreamNativeRow row =
                     alignment.first_native_row();
                 row < alignment.past_last_native_row(); ++row) {
                if (!(pointing.identity(row) == alignment.identity(row))) {
                    throw std::invalid_argument(
                        "native pointing identity differs from alignment authority");
                }
            }
            if (!network_index_by_id_
                     .emplace(pointing.network_id(), index)
                     .second) {
                throw std::invalid_argument(
                    "native pointing plan repeats a network ID");
            }
            participant_network_ids_.push_back(pointing.network_id());
        }
    }

    const NativeObservationScope &scope() const noexcept {
        return alignment_plan_->scope();
    }
    const std::shared_ptr<const NativeAlignmentPlan> &
    alignment_plan_handle() const noexcept {
        return alignment_plan_;
    }
    bool bound_to(const std::shared_ptr<const NativeAlignmentPlan> &plan)
        const noexcept {
        return alignment_plan_.get() == plan.get();
    }
    const std::shared_ptr<const RawTelescopeTrajectory> &
    raw_trajectory_handle() const noexcept {
        return raw_trajectory_;
    }
    const std::vector<TimestreamNetworkId> &participant_network_ids() const
        noexcept {
        return participant_network_ids_;
    }
    const NativeNetworkPointing &network(
        TimestreamNetworkId network_id) const {
        const auto found = network_index_by_id_.find(network_id);
        if (found == network_index_by_id_.end()) {
            throw std::out_of_range(
                "network is absent from native pointing plan");
        }
        return networks_.at(found->second);
    }

private:
    std::shared_ptr<const NativeAlignmentPlan> alignment_plan_;
    std::shared_ptr<const RawTelescopeTrajectory> raw_trajectory_;
    std::vector<NativeNetworkPointing> networks_;
    std::vector<TimestreamNetworkId> participant_network_ids_;
    std::map<TimestreamNetworkId, std::size_t> network_index_by_id_;
};

inline std::shared_ptr<const NativePointingPlan>
make_native_pointing_plan(
    std::shared_ptr<const NativeAlignmentPlan> alignment_plan,
    std::shared_ptr<const RawTelescopeTrajectory> raw_trajectory,
    const NativePointingOffsetModel &offset_model) {
    if (!alignment_plan || !raw_trajectory) {
        throw std::invalid_argument(
            "native pointing candidate lacks required handles");
    }
    std::vector<NativeNetworkPointing> networks;
    networks.reserve(alignment_plan->networks().size());
    for (const auto &alignment : alignment_plan->networks()) {
        Eigen::VectorXd times =
            alignment.reconstructed_times_unix_sec();
        auto telescope = evaluate_raw_telescope_trajectory_at(
            *raw_trajectory, times);
        auto offsets = offset_model.evaluate_at(times);
        networks.emplace_back(
            alignment.network_id(), alignment.first_native_row(),
            std::move(times), std::move(telescope), std::move(offsets));
    }
    return std::make_shared<const NativePointingPlan>(
        std::move(alignment_plan), std::move(raw_trajectory),
        std::move(networks));
}

}  // namespace citlali::pipeline
