#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/timestream_ptc_cohort_adapter.h>
#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/pointing.h>

#include <Eigen/Core>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct NativeScienceDetectorProjection {
    TimestreamDetectorColumn detector_column = -1;
    std::int64_t output_uid = -1;
    std::int64_t array = -1;
    TimestreamNetworkId network_id = -1;
    std::optional<std::int64_t> apt_flag;
    Eigen::Index map_index = -1;
    double az_offset_arcsec = 0.0;
    double el_offset_arcsec = 0.0;

    friend bool operator==(const NativeScienceDetectorProjection &,
                           const NativeScienceDetectorProjection &) =
        default;
};

struct NativeScienceProjectionRequest {
    std::string pixel_axes;
    std::string map_grouping;
    std::vector<NativeScienceDetectorProjection> detectors;
};

struct NativeScienceProjectionCell {
    CoincidenceCellState state = CoincidenceCellState::absent;
    std::size_t segment_ordinal = 0;
    Eigen::Index segment_output_row = -1;
    std::vector<std::size_t> exact_common_slots;
    NativeSampleIdentity identity;
    TimestreamNativeRevision revision = 0;
    double value = 0.0;
    double latitude_rad = 0.0;
    double longitude_rad = 0.0;

    bool projects() const noexcept {
        return state == CoincidenceCellState::mapped_valid;
    }
};

namespace native_science_projection_detail {

inline bool exact_double_equal(double lhs, double rhs) noexcept {
    return std::bit_cast<std::uint64_t>(lhs) ==
           std::bit_cast<std::uint64_t>(rhs);
}

inline double resolve_detector_offset_arcsec(
    double apt_offset_arcsec,
    const std::optional<std::int64_t> &apt_flag) {
    if (std::isfinite(apt_offset_arcsec)) {
        return apt_offset_arcsec;
    }
    if (!apt_flag.has_value() || *apt_flag != 0) {
        // Canonical APT v2 permits science-ineligible detector rows to carry
        // typed-null geometry. They remain excluded by the native PTC flags,
        // but projection still needs a finite, deterministic coordinate in
        // order to preserve the rectangular detector inventory.
        return 0.0;
    }
    throw std::logic_error(
        "eligible native science detector has nonfinite APT offset");
}

inline std::vector<std::string> required_telescope_series(
    const std::string &pixel_axes) {
    if (citlali::config::is_radec_map_pixel_axes(pixel_axes)) {
        return {"TelElAct", "ActParAng", "dec_phys", "ra_phys"};
    }
    if (citlali::config::is_altaz_map_pixel_axes(pixel_axes)) {
        return {"TelElAct", "alt_phys", "az_phys"};
    }
    if (citlali::config::is_galactic_map_pixel_axes(pixel_axes)) {
        return {"TelElAct", "ActParAng", "ActGalAng", "b_phys", "l_phys"};
    }
    throw std::invalid_argument(
        "native science projection pixel axes are unsupported");
}

inline std::pair<double, double> project_native_pointing(
    const NativeNetworkPointing &pointing,
    const NativeSampleIdentity &identity,
    const NativeScienceDetectorProjection &detector,
    const std::string &pixel_axes, const std::string &map_grouping) {
    if (!(pointing.identity(identity.native_row()) == identity)) {
        throw std::logic_error(
            "native science pointing identity is stale or unequal");
    }
    const auto local_row = pointing.local_row(identity.native_row());
    NativeTelescopeData telescope;
    for (const auto &name : required_telescope_series(pixel_axes)) {
        Eigen::VectorXd value(1);
        value(0) = pointing.telescope_series(name)(local_row);
        telescope.emplace(name, std::move(value));
    }
    NativePointingOffsetsArcsec offsets;
    for (const auto *axis : {citlali::config::pointing_axis_az(),
                             citlali::config::pointing_axis_alt()}) {
        Eigen::VectorXd value(1);
        value(0) = pointing.pointing_offset_arcsec(axis)(local_row);
        offsets.emplace(axis, std::move(value));
    }
    auto [latitude, longitude] = engine_utils::calc_det_pointing(
        telescope, detector.az_offset_arcsec,
        detector.el_offset_arcsec, pixel_axes, offsets, map_grouping);
    if (latitude.size() != 1 || longitude.size() != 1 ||
        !std::isfinite(latitude(0)) || !std::isfinite(longitude(0))) {
        throw std::logic_error(
            "native science pointing projection is incomplete or nonfinite");
    }
    return {latitude(0), longitude(0)};
}

}  // namespace native_science_projection_detail

// Immutable, operation-bound bridge from the committed native PTC ledger to
// science consumers. Relational rows only organize the rectangular calling
// surface; every cell retains its own network-native identity and pointing.
class NativeScienceProjection {
public:
    const NativeOperationIdentity &operation() const noexcept {
        return operation_;
    }
    const NativeScanChunkScope &scope() const noexcept { return scope_; }
    const std::string &pixel_axes() const noexcept { return pixel_axes_; }
    const std::string &map_grouping() const noexcept {
        return map_grouping_;
    }
    Eigen::Index row_count() const noexcept { return values_.rows(); }
    Eigen::Index detector_count() const noexcept { return values_.cols(); }
    const Eigen::MatrixXd &values() const noexcept { return values_; }
    const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> &flags()
        const noexcept {
        return flags_;
    }
    const Eigen::MatrixXd &latitudes_rad() const noexcept {
        return latitudes_rad_;
    }
    const Eigen::MatrixXd &longitudes_rad() const noexcept {
        return longitudes_rad_;
    }
    const std::vector<NativeScienceDetectorProjection> &detectors()
        const noexcept {
        return detectors_;
    }
    const NativeScienceProjectionCell &cell(
        Eigen::Index row, Eigen::Index detector) const {
        require_cell(row, detector);
        return cells_.at(static_cast<std::size_t>(row) *
                             static_cast<std::size_t>(detector_count()) +
                         static_cast<std::size_t>(detector));
    }
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> detector_pointing(
        Eigen::Index detector) const {
        if (detector < 0 || detector >= detector_count()) {
            throw std::out_of_range(
                "native science detector pointing is out of range");
        }
        return {latitudes_rad_.col(detector),
                longitudes_rad_.col(detector)};
    }

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>
    map_center_source_mask(double radius_arcsec) const {
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> result(
            row_count(), detector_count());
        result.setConstant(false);
        if (!(radius_arcsec > 0.0) || !std::isfinite(radius_arcsec)) {
            return result;
        }
        const double radius2 =
            radius_arcsec * ASEC_TO_RAD * radius_arcsec * ASEC_TO_RAD;
        for (Eigen::Index row = 0; row < row_count(); ++row) {
            for (Eigen::Index detector = 0;
                 detector < detector_count(); ++detector) {
                const auto &candidate = cell(row, detector);
                if (!candidate.projects()) continue;
                result(row, detector) =
                    candidate.latitude_rad * candidate.latitude_rad +
                        candidate.longitude_rad * candidate.longitude_rad <=
                    radius2;
            }
        }
        return result;
    }

    template <typename input_t, typename Derived, typename apt_t>
    void require_compatible_mapmaking_input(
        const input_t &input,
        const Eigen::DenseBase<Derived> &map_indices,
        const std::string &pixel_axes, const std::string &map_grouping,
        const apt_t &apt) const {
        if (pixel_axes != pixel_axes_ || map_grouping != map_grouping_ ||
            input.scans.data.rows() != row_count() ||
            input.scans.data.cols() != detector_count() ||
            input.flags.data.rows() != row_count() ||
            input.flags.data.cols() != detector_count() ||
            map_indices.size() != detector_count()) {
            throw std::logic_error(
                "native science mapmaking candidate is incomplete or foreign");
        }
        const auto require_apt = [&](const char *name) -> const auto & {
            const auto found = apt.find(name);
            if (found == apt.end() || found->second.size() != detector_count()) {
                throw std::logic_error(
                    std::string{"native science mapmaking APT lacks exact "} +
                    name + " inventory");
            }
            return found->second;
        };
        const auto &uid = require_apt("uid");
        const auto &array = require_apt("array");
        const auto &flag = require_apt("flag");
        const auto &x_t = require_apt("x_t");
        const auto &y_t = require_apt("y_t");
        for (Eigen::Index detector = 0;
             detector < detector_count(); ++detector) {
            const auto &binding = detectors_.at(
                static_cast<std::size_t>(detector));
            const auto resolved_x =
                native_science_projection_detail::resolve_detector_offset_arcsec(
                    x_t(detector), binding.apt_flag);
            const auto resolved_y =
                native_science_projection_detail::resolve_detector_offset_arcsec(
                    y_t(detector), binding.apt_flag);
            if (binding.detector_column != detector ||
                map_indices(detector) != binding.map_index ||
                !std::isfinite(uid(detector)) ||
                uid(detector) != static_cast<double>(binding.output_uid) ||
                !std::isfinite(array(detector)) ||
                array(detector) != static_cast<double>(binding.array) ||
                (binding.apt_flag.has_value()
                     ? (!std::isfinite(flag(detector)) ||
                        flag(detector) !=
                            static_cast<double>(*binding.apt_flag))
                     : (std::isfinite(flag(detector)) &&
                        flag(detector) == 0.0)) ||
                !native_science_projection_detail::exact_double_equal(
                    resolved_x, binding.az_offset_arcsec) ||
                !native_science_projection_detail::exact_double_equal(
                    resolved_y, binding.el_offset_arcsec)) {
                throw std::logic_error(
                    "native science mapmaking detector authority is unequal or synthetic");
            }
            for (Eigen::Index row = 0; row < row_count(); ++row) {
                if (input.flags.data(row, detector) !=
                        flags_(row, detector) ||
                    !native_science_projection_detail::exact_double_equal(
                        input.scans.data(row, detector),
                        values_(row, detector))) {
                    throw std::logic_error(
                        "native science mapmaking samples are stale, unequal, or synthetic");
                }
            }
        }
    }

private:
    friend NativeScienceProjection make_native_science_projection(
        const NativeMeasuredDetectorLedger &,
        const NativePtcPreparedOperation &,
        NativeScienceProjectionRequest);

    NativeScienceProjection(
        NativeOperationIdentity operation, NativeScanChunkScope scope,
        std::string pixel_axes, std::string map_grouping,
        std::vector<NativeScienceDetectorProjection> detectors,
        Eigen::MatrixXd values,
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> flags,
        Eigen::MatrixXd latitudes_rad, Eigen::MatrixXd longitudes_rad,
        std::vector<NativeScienceProjectionCell> cells)
        : operation_{operation}, scope_{std::move(scope)},
          pixel_axes_{std::move(pixel_axes)},
          map_grouping_{std::move(map_grouping)},
          detectors_{std::move(detectors)}, values_{std::move(values)},
          flags_{std::move(flags)},
          latitudes_rad_{std::move(latitudes_rad)},
          longitudes_rad_{std::move(longitudes_rad)},
          cells_{std::move(cells)} {}

    void require_cell(Eigen::Index row, Eigen::Index detector) const {
        if (row < 0 || detector < 0 || row >= row_count() ||
            detector >= detector_count()) {
            throw std::out_of_range(
                "native science projection cell is out of range");
        }
    }

    NativeOperationIdentity operation_;
    NativeScanChunkScope scope_;
    std::string pixel_axes_;
    std::string map_grouping_;
    std::vector<NativeScienceDetectorProjection> detectors_;
    Eigen::MatrixXd values_;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> flags_;
    Eigen::MatrixXd latitudes_rad_;
    Eigen::MatrixXd longitudes_rad_;
    std::vector<NativeScienceProjectionCell> cells_;
};

inline NativeScienceProjection make_native_science_projection(
    const NativeMeasuredDetectorLedger &ledger,
    const NativePtcPreparedOperation &prepared,
    NativeScienceProjectionRequest request) {
    const auto mapping = ledger.mapping_handle();
    if (!mapping || prepared.mapping_handle().get() != mapping.get() ||
        !ledger.last_operation().has_value() ||
        !ledger.last_committed_operation().has_value() ||
        !(*ledger.last_operation() == prepared.operation()) ||
        !(*ledger.last_committed_operation() == prepared.operation())) {
        throw std::logic_error(
            "native science projection requires the exact committed PTC operation");
    }
    if (prepared.detector_count() != mapping->detector_count() ||
        prepared.segment_count() == 0 || prepared.groups().empty() ||
        mapping->detector_count() == 0 ||
        request.detectors.size() != mapping->detector_count()) {
        throw std::invalid_argument(
            "native science projection inventory is incomplete");
    }
    (void)native_science_projection_detail::required_telescope_series(
        request.pixel_axes);
    const auto grouping =
        citlali::config::parse_map_grouping(request.map_grouping);
    if (!grouping.has_value() ||
        citlali::config::is_automatic_map_grouping(*grouping)) {
        throw std::invalid_argument(
            "native science projection requires a resolved map grouping");
    }

    std::vector<bool> detector_seen(mapping->detector_count(), false);
    for (const auto &detector : request.detectors) {
        if (detector.detector_column < 0 ||
            static_cast<std::size_t>(detector.detector_column) >=
                mapping->detector_count() ||
            detector_seen.at(static_cast<std::size_t>(
                detector.detector_column)) ||
            detector.map_index < 0 ||
            !std::isfinite(detector.az_offset_arcsec) ||
            !std::isfinite(detector.el_offset_arcsec) ||
            mapping->binding(detector.detector_column).output_uid !=
                detector.output_uid ||
            mapping->binding(detector.detector_column).array !=
                detector.array ||
            mapping->binding(detector.detector_column).network_id !=
                detector.network_id ||
            mapping->binding(detector.detector_column).apt_flag !=
                detector.apt_flag) {
            throw std::invalid_argument(
                "native science detector projection is duplicate, foreign, or nonfinite");
        }
        detector_seen[static_cast<std::size_t>(detector.detector_column)] =
            true;
    }
    std::sort(request.detectors.begin(), request.detectors.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.detector_column < rhs.detector_column;
              });

    std::vector<Eigen::Index> rows_per_segment(prepared.segment_count(), -1);
    for (const auto &group : prepared.groups()) {
        if (group.segment_ordinal() >= prepared.segment_count() ||
            group.slot_count() <= 0) {
            throw std::logic_error(
                "native science PTC group has a foreign or empty segment");
        }
        auto &count = rows_per_segment[group.segment_ordinal()];
        if (count < 0) count = group.slot_count();
        if (count != group.slot_count()) {
            throw std::logic_error(
                "native science PTC segment row counts are unequal");
        }
    }
    if (std::any_of(rows_per_segment.begin(), rows_per_segment.end(),
                    [](Eigen::Index count) { return count <= 0; })) {
        throw std::logic_error(
            "native science PTC segment inventory is incomplete");
    }
    std::vector<Eigen::Index> row_offsets(prepared.segment_count(), 0);
    Eigen::Index total_rows = 0;
    for (std::size_t segment = 0; segment < rows_per_segment.size();
         ++segment) {
        row_offsets[segment] = total_rows;
        if (rows_per_segment[segment] >
            std::numeric_limits<Eigen::Index>::max() - total_rows) {
            throw std::length_error(
                "native science projection row inventory overflows");
        }
        total_rows += rows_per_segment[segment];
    }

    const auto detector_count = static_cast<Eigen::Index>(
        mapping->detector_count());
    Eigen::MatrixXd values(total_rows, detector_count);
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> flags(
        total_rows, detector_count);
    Eigen::MatrixXd latitudes(total_rows, detector_count);
    Eigen::MatrixXd longitudes(total_rows, detector_count);
    flags.setConstant(true);
    if (static_cast<std::size_t>(total_rows) >
        std::numeric_limits<std::size_t>::max() /
            mapping->detector_count()) {
        throw std::length_error(
            "native science projection cell inventory overflows");
    }
    std::vector<std::optional<NativeScienceProjectionCell>> staged(
        static_cast<std::size_t>(total_rows) * mapping->detector_count());
    std::set<NativeDetectorSampleKey> destinations;
    std::vector<std::optional<std::vector<std::size_t>>> row_support(
        static_cast<std::size_t>(total_rows));

    const auto &pointing_plan =
        *mapping->carriers_handle()->pointing_handle();
    for (const auto &group : prepared.groups()) {
        for (Eigen::Index local = 0;
             local < group.detector_count(); ++local) {
            const auto detector_column = group.detector_columns().at(
                static_cast<std::size_t>(local));
            if (detector_column < 0 || detector_column >= detector_count) {
                throw std::logic_error(
                    "native science PTC detector partition is foreign");
            }
            const auto &detector = request.detectors.at(
                static_cast<std::size_t>(detector_column));
            for (Eigen::Index row = 0; row < group.slot_count(); ++row) {
                const auto &source = group.cell(row, local);
                const Eigen::Index output_row =
                    row_offsets[group.segment_ordinal()] + row;
                const auto index = static_cast<std::size_t>(output_row) *
                        mapping->detector_count() +
                    static_cast<std::size_t>(detector_column);
                if (staged[index].has_value() ||
                    !source.identity.has_value() ||
                    source.segment_ordinal != group.segment_ordinal() ||
                    source.segment_output_row != row ||
                    source.exact_common_slots.empty()) {
                    throw std::logic_error(
                        "native science PTC candidate is duplicate or incomplete");
                }
                auto &support = row_support.at(
                    static_cast<std::size_t>(output_row));
                if (!support.has_value()) support = source.exact_common_slots;
                if (*support != source.exact_common_slots) {
                    throw std::logic_error(
                        "native science PTC relational support is unequal");
                }
                const NativeDetectorSampleKey key{
                    source.identity->key(), detector_column};
                if (!destinations.insert(key).second) {
                    throw std::logic_error(
                        "native science PTC destination is duplicated");
                }
                const auto record = ledger.record(key);
                if (!(record.identity == *source.identity) ||
                    source.expected_revision ==
                        std::numeric_limits<TimestreamNativeRevision>::max() ||
                    record.revision != source.expected_revision + 1 ||
                    !std::isfinite(record.current_value)) {
                    throw std::logic_error(
                        "native science PTC cell is stale, unequal, or nonfinite");
                }
                const bool valid =
                    source.state == CoincidenceCellState::mapped_valid;
                const bool invalid =
                    source.state == CoincidenceCellState::mapped_invalid;
                if ((!valid && !invalid) ||
                    (valid && (source.delivered_flag_bits != 0 ||
                               source.operation_exclusion_bits != 0 ||
                               !source.apt_flag.has_value() ||
                               *source.apt_flag != 0)) ||
                    (invalid && source.delivered_flag_bits == 0 &&
                               source.operation_exclusion_bits == 0 &&
                               source.apt_flag.has_value() &&
                               *source.apt_flag == 0)) {
                    throw std::logic_error(
                        "native science PTC validity authority is unequal");
                }
                const auto [latitude, longitude] =
                    native_science_projection_detail::project_native_pointing(
                        pointing_plan.network(
                            source.identity->network_id()),
                        *source.identity, detector, request.pixel_axes,
                        request.map_grouping);
                values(output_row, detector_column) = record.current_value;
                flags(output_row, detector_column) = !valid;
                latitudes(output_row, detector_column) = latitude;
                longitudes(output_row, detector_column) = longitude;
                staged[index] = NativeScienceProjectionCell{
                    source.state, source.segment_ordinal,
                    source.segment_output_row, source.exact_common_slots,
                    *source.identity, record.revision,
                    record.current_value, latitude, longitude};
            }
        }
    }
    if (destinations.size() != staged.size() ||
        std::any_of(staged.begin(), staged.end(),
                    [](const auto &cell) { return !cell.has_value(); })) {
        throw std::logic_error(
            "native science PTC detector/sample partition is incomplete");
    }
    std::vector<NativeScienceProjectionCell> cells;
    cells.reserve(staged.size());
    for (auto &cell : staged) cells.push_back(std::move(*cell));
    return NativeScienceProjection{
        prepared.operation(), mapping->scope(),
        std::move(request.pixel_axes), std::move(request.map_grouping),
        std::move(request.detectors), std::move(values), std::move(flags),
        std::move(latitudes), std::move(longitudes), std::move(cells)};
}

}  // namespace citlali::pipeline
