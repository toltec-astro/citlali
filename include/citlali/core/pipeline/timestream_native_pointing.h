#pragma once

#include <citlali/core/config/calibration_config.h>
#include <citlali/core/pipeline/timestream_native_consumer_bridge.h>

#include <Eigen/Core>
#include <tula/algorithm/mlinterp/mlinterp.hpp>

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
    for (Eigen::Index i = 1; i < values.size(); ++i) {
        if (!(values(i) > values(i - 1))) {
            throw std::invalid_argument(label + " must increase strictly");
        }
    }
}

inline bool vectors_are_exactly_equal(const Eigen::VectorXd &lhs,
                                      const Eigen::VectorXd &rhs) noexcept {
    return lhs.size() == rhs.size() &&
           (lhs.array() == rhs.array()).all();
}

// Preserve the established conversion, including its integer-second
// truncation.  This helper only resolves the already accepted two-point
// pointing-offset support; it is not a detector timing correction.
inline double configured_mjd_to_unix_support(double modified_julian_date) {
    constexpr double unix_epoch_julian_date = 2440587.5;
    constexpr double modified_to_julian_offset = 2400000.5;
    const double days = modified_julian_date - unix_epoch_julian_date +
                        modified_to_julian_offset;
    if (!std::isfinite(days) ||
        days < static_cast<double>(std::numeric_limits<long long>::min()) /
                   86400.0 ||
        days > static_cast<double>(std::numeric_limits<long long>::max()) /
                   86400.0) {
        throw std::invalid_argument(
            "pointing-offset MJD support is not representable in Unix seconds");
    }
    return static_cast<double>(
        static_cast<long long>(days * 86400.0));
}

}  // namespace native_pointing_detail

// Immutable snapshot of the one measured telescope trajectory loaded for the
// observation, captured before any common-slot interpolation.  Its TelTime is
// telescope support time; it is distinct from every detector network's
// reconstructed native timestamp.
class RawTelescopeTrajectory {
public:
    explicit RawTelescopeTrajectory(NativeTelescopeData telescope_data)
        : telescope_data_{std::move(telescope_data)} {
        const auto time_it = telescope_data_.find("TelTime");
        if (time_it == telescope_data_.end()) {
            throw std::invalid_argument(
                "raw telescope trajectory requires TelTime support");
        }
        native_pointing_detail::require_strictly_increasing(
            time_it->second, "raw telescope TelTime support");
        if (time_it->second.size() < 2) {
            throw std::invalid_argument(
                "raw telescope trajectory requires at least two support samples");
        }

        const Eigen::Index sample_count = time_it->second.size();
        for (const auto &[key, values] : telescope_data_) {
            if (values.size() != sample_count) {
                throw std::invalid_argument(
                    "raw telescope series '" + key +
                    "' does not match TelTime cardinality");
            }
            if (!values.array().isFinite().all()) {
                throw std::invalid_argument(
                    "raw telescope series '" + key + "' must be finite");
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

// Frozen one- or two-value pointing-offset model.  In the two-value case the
// support is either the same established common-observation endpoint pair or
// the explicitly configured MJD pair.  It is evaluated independently at each
// network's reconstructed native timestamps.
class NativePointingOffsetModel {
public:
    NativePointingOffsetModel(NativePointingOffsetsArcsec source_arcsec,
                              Eigen::VectorXd support_times_unix_sec,
                              bool uses_explicit_mjd_support)
        : source_arcsec_{std::move(source_arcsec)},
          support_times_unix_sec_{std::move(support_times_unix_sec)},
          uses_explicit_mjd_support_{uses_explicit_mjd_support} {
        const auto az_it =
            source_arcsec_.find(citlali::config::pointing_axis_az());
        const auto alt_it =
            source_arcsec_.find(citlali::config::pointing_axis_alt());
        if (az_it == source_arcsec_.end() || alt_it == source_arcsec_.end()) {
            throw std::invalid_argument(
                "pointing-offset model requires both az and alt sources");
        }
        source_value_count_ = az_it->second.size();
        if (source_value_count_ != alt_it->second.size() ||
            (source_value_count_ != 1 && source_value_count_ != 2)) {
            throw std::invalid_argument(
                "pointing-offset az/alt sources must have equal one- or two-value cardinality");
        }
        if (!az_it->second.array().isFinite().all() ||
            !alt_it->second.array().isFinite().all()) {
            throw std::invalid_argument(
                "pointing-offset az/alt sources must be finite");
        }
        if (support_times_unix_sec_.size() != 2 ||
            !support_times_unix_sec_.array().isFinite().all()) {
            throw std::invalid_argument(
                "pointing-offset model requires a finite two-value support interval");
        }
        if (source_value_count_ == 2 &&
            !(support_times_unix_sec_(1) > support_times_unix_sec_(0))) {
            throw std::invalid_argument(
                "two-value pointing-offset support must increase strictly");
        }
    }

    Eigen::Index source_value_count() const noexcept {
        return source_value_count_;
    }
    const Eigen::VectorXd &support_times_unix_sec() const noexcept {
        return support_times_unix_sec_;
    }
    double support_start_unix_sec() const noexcept {
        return support_times_unix_sec_(0);
    }
    double support_end_unix_sec() const noexcept {
        return support_times_unix_sec_(1);
    }
    bool uses_explicit_mjd_support() const noexcept {
        return uses_explicit_mjd_support_;
    }

    NativePointingOffsetsArcsec evaluate_at(
        const Eigen::VectorXd &target_reconstructed_times_unix_sec) const {
        native_pointing_detail::require_strictly_increasing(
            target_reconstructed_times_unix_sec,
            "pointing-offset target reconstructed native timestamps");
        if (source_value_count_ == 2 &&
            (target_reconstructed_times_unix_sec(0) <
                 support_start_unix_sec() ||
             target_reconstructed_times_unix_sec(
                 target_reconstructed_times_unix_sec.size() - 1) >
                 support_end_unix_sec())) {
            throw std::out_of_range(
                "native pointing-offset target is outside configured support");
        }

        NativePointingOffsetsArcsec result;
        for (const auto *axis : {citlali::config::pointing_axis_alt(),
                                 citlali::config::pointing_axis_az()}) {
            Eigen::VectorXd evaluated(
                target_reconstructed_times_unix_sec.size());
            const auto &source = source_arcsec_.at(axis);
            if (source_value_count_ == 1) {
                evaluated.setConstant(source(0));
            }
            else {
                Eigen::Matrix<Eigen::Index, 1, 1> support_shape;
                support_shape << source_value_count_;
                mlinterp::interp(
                    support_shape.data(), evaluated.size(), source.data(),
                    evaluated.data(), support_times_unix_sec_.data(),
                    target_reconstructed_times_unix_sec.data());
            }
            if (!evaluated.array().isFinite().all()) {
                throw std::logic_error(
                    "native pointing-offset evaluation produced a nonfinite value");
            }
            result.emplace(axis, std::move(evaluated));
        }
        return result;
    }

private:
    NativePointingOffsetsArcsec source_arcsec_;
    Eigen::VectorXd support_times_unix_sec_;
    Eigen::Index source_value_count_ = 0;
    bool uses_explicit_mjd_support_ = false;
};

template <class PointingOffsets>
NativePointingOffsetModel make_native_pointing_offset_model(
    const PointingOffsets &pointing_offsets,
    const Eigen::VectorXd &common_observation_times_unix_sec) {
    native_pointing_detail::require_strictly_increasing(
        common_observation_times_unix_sec,
        "common observation times used only for pointing-offset support");

    NativePointingOffsetsArcsec source;
    for (const auto *axis : {citlali::config::pointing_axis_alt(),
                             citlali::config::pointing_axis_az()}) {
        const auto it = pointing_offsets.arcsec.find(axis);
        if (it == pointing_offsets.arcsec.end()) {
            throw std::invalid_argument(
                "pointing_offsets must include both az and alt vectors");
        }
        source.emplace(axis, it->second);
    }

    const bool use_explicit_mjd =
        pointing_offsets.modified_julian_date.size() == 2 &&
        (pointing_offsets.modified_julian_date.array() > 0.0).all();
    Eigen::VectorXd support(2);
    if (use_explicit_mjd) {
        support << native_pointing_detail::configured_mjd_to_unix_support(
                       pointing_offsets.modified_julian_date(0)),
            native_pointing_detail::configured_mjd_to_unix_support(
                pointing_offsets.modified_julian_date(1));
        if (!(support(1) > support(0)) ||
            support(0) > common_observation_times_unix_sec(0) ||
            support(1) < common_observation_times_unix_sec(
                             common_observation_times_unix_sec.size() - 1)) {
            throw std::invalid_argument(
                "pointing-offset MJD support does not bracket the common observation");
        }
    }
    else {
        support << common_observation_times_unix_sec(0),
            common_observation_times_unix_sec(
                common_observation_times_unix_sec.size() - 1);
    }
    return NativePointingOffsetModel{
        std::move(source), std::move(support), use_explicit_mjd};
}

// Immutable telescope and pointing-offset evaluation for the contiguous
// delivered-row interval that contains every mapped row of one network.  The
// target timestamp is reconstructed raw-data provenance; this carrier does not
// declare it to be a physical detector integration time.
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
            "native network pointing reconstructed timestamps");
        if (reconstructed_times_unix_sec_.size() >
            std::numeric_limits<TimestreamNativeRow>::max() -
                first_native_row_) {
            throw std::length_error(
                "native network pointing row interval would overflow");
        }

        const auto tel_time_it = telescope_data_.find("TelTime");
        if (tel_time_it == telescope_data_.end() ||
            !native_pointing_detail::vectors_are_exactly_equal(
                tel_time_it->second, reconstructed_times_unix_sec_)) {
            throw std::invalid_argument(
                "native telescope TelTime must exactly equal its reconstructed target timestamps");
        }
        for (const auto &[key, values] : telescope_data_) {
            require_carrier_series(values, "native telescope series '" + key + "'");
        }
        for (const auto *required : {"lat_phys", "lon_phys", "TelAzAct",
                                     "TelElAct"}) {
            if (telescope_data_.find(required) == telescope_data_.end()) {
                throw std::invalid_argument(
                    std::string{"native telescope pointing is missing required series '"} +
                    required + "'");
            }
        }

        for (const auto *axis : {citlali::config::pointing_axis_alt(),
                                 citlali::config::pointing_axis_az()}) {
            const auto it = pointing_offsets_arcsec_.find(axis);
            if (it == pointing_offsets_arcsec_.end()) {
                throw std::invalid_argument(
                    "native pointing carrier requires both az and alt offsets");
            }
            require_carrier_series(
                it->second,
                std::string{"native pointing-offset series '"} + axis + "'");
        }
    }

    TimestreamNetworkId network_id() const noexcept { return network_id_; }
    TimestreamNativeRow first_native_row() const noexcept {
        return first_native_row_;
    }
    TimestreamNativeRow past_last_native_row() const noexcept {
        return first_native_row_ +
               static_cast<TimestreamNativeRow>(
                   reconstructed_times_unix_sec_.size());
    }
    Eigen::Index row_count() const noexcept {
        return reconstructed_times_unix_sec_.size();
    }
    const Eigen::VectorXd &reconstructed_times_unix_sec() const noexcept {
        return reconstructed_times_unix_sec_;
    }
    NativeSampleIdentity identity(TimestreamNativeRow native_row) const {
        const Eigen::Index offset = local_row(native_row);
        return NativeSampleIdentity{
            network_id_, native_row,
            reconstructed_times_unix_sec_(offset)};
    }
    Eigen::Index local_row(TimestreamNativeRow native_row) const {
        if (native_row < first_native_row_ ||
            native_row >= past_last_native_row()) {
            throw std::out_of_range(
                "native row is outside the network pointing carrier");
        }
        return static_cast<Eigen::Index>(native_row - first_native_row_);
    }
    const NativeTelescopeData &telescope_data() const noexcept {
        return telescope_data_;
    }
    const Eigen::VectorXd &telescope_series(const std::string &key) const {
        return telescope_data_.at(key);
    }
    const NativePointingOffsetsArcsec &pointing_offsets_arcsec() const
        noexcept {
        return pointing_offsets_arcsec_;
    }
    const Eigen::VectorXd &pointing_offset_arcsec(
        const std::string &axis) const {
        return pointing_offsets_arcsec_.at(axis);
    }

private:
    void require_carrier_series(const Eigen::VectorXd &values,
                                const std::string &label) const {
        if (values.size() != reconstructed_times_unix_sec_.size() ||
            !values.array().isFinite().all()) {
            throw std::invalid_argument(
                label + " must be finite and match native row cardinality");
        }
    }

    TimestreamNetworkId network_id_;
    TimestreamNativeRow first_native_row_;
    Eigen::VectorXd reconstructed_times_unix_sec_;
    NativeTelescopeData telescope_data_;
    NativePointingOffsetsArcsec pointing_offsets_arcsec_;
};

// Observation-owned immutable carrier, bound to the exact alignment-plan
// object that supplied every native identity.  Common slots are absent here by
// design: consumers receive temporal eligibility from their admitted cohort.
class NativePointingPlan {
public:
    NativePointingPlan(
        std::shared_ptr<const NativeAlignmentPlan> alignment_plan,
        std::shared_ptr<const RawTelescopeTrajectory>
            raw_telescope_trajectory,
        std::vector<NativeNetworkPointing> networks)
        : alignment_plan_{std::move(alignment_plan)},
          raw_telescope_trajectory_{
              std::move(raw_telescope_trajectory)},
          networks_{std::move(networks)} {
        if (!alignment_plan_ || !raw_telescope_trajectory_) {
            throw std::invalid_argument(
                "native pointing plan requires exact alignment and raw telescope handles");
        }
        if (networks_.size() != alignment_plan_->networks().size()) {
            throw std::invalid_argument(
                "native pointing networks must match the alignment plan");
        }

        const auto &expected_ids =
            alignment_plan_->participant_network_ids();
        participant_network_ids_.reserve(networks_.size());
        for (std::size_t index = 0; index < networks_.size(); ++index) {
            const auto &pointing = networks_[index];
            const auto &alignment = alignment_plan_->network(
                expected_ids.at(index));
            if (pointing.network_id() != alignment.network_id() ||
                pointing.first_native_row() < alignment.first_native_row() ||
                pointing.past_last_native_row() >
                    alignment.past_last_native_row()) {
                throw std::invalid_argument(
                    "native pointing row interval is outside its alignment network");
            }
            for (Eigen::Index local_row = 0;
                 local_row < pointing.row_count(); ++local_row) {
                const auto native_row =
                    pointing.first_native_row() +
                    static_cast<TimestreamNativeRow>(local_row);
                if (!(pointing.identity(native_row) ==
                      alignment.identity(native_row))) {
                    throw std::invalid_argument(
                        "native pointing identity does not exactly match its alignment network");
                }
            }
            for (std::size_t slot = 0;
                 slot < alignment_plan_->slot_count(); ++slot) {
                const auto &association = alignment_plan_->association(
                    alignment.network_id(), slot);
                if (association.mapped() &&
                    (association.native_row < pointing.first_native_row() ||
                     association.native_row >=
                         pointing.past_last_native_row())) {
                    throw std::invalid_argument(
                        "mapped native row lacks network-native telescope pointing");
                }
            }
            if (!network_index_by_id_
                     .emplace(pointing.network_id(), index)
                     .second) {
                throw std::invalid_argument(
                    "native pointing plan contains a duplicate network ID");
            }
            participant_network_ids_.push_back(pointing.network_id());
        }
        if (participant_network_ids_ != expected_ids) {
            throw std::invalid_argument(
                "native pointing participant order changed from the alignment plan");
        }
    }

    const std::shared_ptr<const NativeAlignmentPlan> &alignment_plan_handle()
        const noexcept {
        return alignment_plan_;
    }
    const std::shared_ptr<const RawTelescopeTrajectory> &
    raw_telescope_trajectory_handle() const noexcept {
        return raw_telescope_trajectory_;
    }
    bool bound_to(
        const std::shared_ptr<const NativeAlignmentPlan> &alignment_plan)
        const noexcept {
        return alignment_plan_.get() == alignment_plan.get();
    }
    const std::vector<TimestreamNetworkId> &participant_network_ids() const
        noexcept {
        return participant_network_ids_;
    }
    const NativeNetworkPointing &network(
        TimestreamNetworkId network_id) const {
        const auto it = network_index_by_id_.find(network_id);
        if (it == network_index_by_id_.end()) {
            throw std::out_of_range(
                "network is absent from the native pointing plan");
        }
        return networks_.at(it->second);
    }

private:
    std::shared_ptr<const NativeAlignmentPlan> alignment_plan_;
    std::shared_ptr<const RawTelescopeTrajectory>
        raw_telescope_trajectory_;
    std::vector<NativeNetworkPointing> networks_;
    std::vector<TimestreamNetworkId> participant_network_ids_;
    std::map<TimestreamNetworkId, std::size_t> network_index_by_id_;
};

}  // namespace citlali::pipeline
