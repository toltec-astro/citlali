#include <citlali/core/pipeline/ast_scan_motion.h>

#include <Eigen/QR>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace citlali::pipeline {
namespace {

constexpr std::size_t half_window = 5;
constexpr std::size_t window_size = 2 * half_window + 1;
constexpr std::size_t pair_count =
    window_size * (window_size - 1) / 2;
constexpr double radians_to_arcsec =
    180.0 * 3600.0 / std::numbers::pi_v<double>;

bool has_any_cause(AstScanMotionCause value) noexcept {
    return value != AstScanMotionCause::none;
}

double shortest_signed_angle(double value) noexcept {
    double wrapped = std::remainder(value, 2.0 * std::numbers::pi_v<double>);
    if (wrapped >= std::numbers::pi_v<double>) {
        wrapped -= 2.0 * std::numbers::pi_v<double>;
    }
    return wrapped;
}

bool direction_is_structurally_valid(double ra, double dec) noexcept {
    return std::isfinite(ra) && std::isfinite(dec) &&
           std::abs(dec) <= std::numbers::pi_v<double> / 2.0;
}

bool exact_coordinate_antipode(double center_ra, double center_dec,
                               double target_ra, double target_dec) noexcept {
    return target_dec == -center_dec &&
           std::abs(shortest_signed_angle(target_ra - center_ra)) ==
               std::numbers::pi_v<double>;
}

struct TangentCoordinate {
    double east_rad = 0.0;
    double north_rad = 0.0;
};

std::optional<TangentCoordinate> spherical_log_j2000(
    double center_ra, double center_dec, double target_ra,
    double target_dec) {
    if (!direction_is_structurally_valid(center_ra, center_dec) ||
        !direction_is_structurally_valid(target_ra, target_dec) ||
        exact_coordinate_antipode(center_ra, center_dec, target_ra,
                                  target_dec)) {
        return std::nullopt;
    }

    const double center_cos_dec = std::cos(center_dec);
    const std::array<double, 3> center{
        center_cos_dec * std::cos(center_ra),
        center_cos_dec * std::sin(center_ra), std::sin(center_dec)};
    const double target_cos_dec = std::cos(target_dec);
    const std::array<double, 3> target{
        target_cos_dec * std::cos(target_ra),
        target_cos_dec * std::sin(target_ra), std::sin(target_dec)};
    const double dot = std::clamp(
        center[0] * target[0] + center[1] * target[1] +
            center[2] * target[2],
        -1.0, 1.0);
    const std::array<double, 3> tangent{
        target[0] - dot * center[0], target[1] - dot * center[1],
        target[2] - dot * center[2]};
    const double tangent_norm = std::hypot(
        std::hypot(tangent[0], tangent[1]), tangent[2]);

    if (tangent_norm == 0.0) {
        if (dot < 0.0) return std::nullopt;
        return TangentCoordinate{};
    }

    const double angle = std::atan2(tangent_norm, dot);
    const std::array<double, 3> unit_tangent{
        tangent[0] / tangent_norm, tangent[1] / tangent_norm,
        tangent[2] / tangent_norm};
    const std::array<double, 3> east{-std::sin(center_ra),
                                     std::cos(center_ra), 0.0};
    const std::array<double, 3> north{
        -std::sin(center_dec) * std::cos(center_ra),
        -std::sin(center_dec) * std::sin(center_ra),
        std::cos(center_dec)};
    TangentCoordinate result{
        angle * (unit_tangent[0] * east[0] +
                 unit_tangent[1] * east[1] +
                 unit_tangent[2] * east[2]),
        angle * (unit_tangent[0] * north[0] +
                 unit_tangent[1] * north[1] +
                 unit_tangent[2] * north[2])};
    if (!std::isfinite(result.east_rad) ||
        !std::isfinite(result.north_rad)) {
        return std::nullopt;
    }
    return result;
}

template <std::size_t count>
double median(std::array<double, count> values) {
    static_assert(count > 0);
    const auto middle = values.begin() +
        static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    if ((values.size() % 2U) != 0U) return *middle;
    const auto lower = *std::max_element(values.begin(), middle);
    return 0.5 * (lower + *middle);
}

bool same_complete_window(
    const std::vector<AstScanMotionDerivedRecord> &records,
    std::size_t center) noexcept {
    if (center < half_window ||
        center + half_window >= records.size()) {
        return false;
    }
    const auto run = records[center].continuity_run();
    if (run < 0) return false;
    for (std::size_t index = center - half_window;
         index <= center + half_window; ++index) {
        if (records[index].continuity_run() != run ||
            !records[index].raw_direction_structurally_valid()) {
            return false;
        }
    }
    return true;
}

std::optional<std::array<TangentCoordinate, window_size>> tangent_window(
    const AstScanMotionSource &source, std::size_t center) {
    std::array<TangentCoordinate, window_size> result;
    const auto &ra = source.source_ra_act_rad();
    const auto &dec = source.source_dec_act_rad();
    for (std::size_t offset = 0; offset < window_size; ++offset) {
        const auto index = center - half_window + offset;
        const auto value = spherical_log_j2000(
            ra(static_cast<Eigen::Index>(center)),
            dec(static_cast<Eigen::Index>(center)),
            ra(static_cast<Eigen::Index>(index)),
            dec(static_cast<Eigen::Index>(index)));
        if (!value) return std::nullopt;
        result[offset] = *value;
    }
    return result;
}

AstScanMotionCause source_family_causes(
    const AstScanMotionSource &source) noexcept {
    AstScanMotionCause causes = AstScanMotionCause::none;
    const auto &metadata = source.metadata();
    if (metadata.producer_kind != AstScanMotionProducerKind::real_toltec) {
        causes |= AstScanMotionCause::unsupported_producer_kind;
    }
    if (metadata.dcs_observation_goal != "Science") {
        causes |= AstScanMotionCause::not_science_observation;
    }
    if (metadata.dcs_observation_program != "Lissajous") {
        causes |= AstScanMotionCause::unsupported_scan_program;
    }
    if (metadata.scan_file_valid != 1) {
        causes |= AstScanMotionCause::invalid_scan_file;
    }
    if (!std::isfinite(metadata.source_epoch) ||
        metadata.source_epoch != 2000.0 ||
        metadata.source_coordinate_system != 0) {
        causes |= AstScanMotionCause::unsupported_source_frame;
    }
    if (!std::isfinite(metadata.nominal_producer_cadence_hz) ||
        metadata.nominal_producer_cadence_hz != 50.0) {
        causes |= AstScanMotionCause::unsupported_producer_cadence;
    }
    if (metadata.field_registry != AstScanMotionFieldRegistry::
                                       source_ra_act_source_dec_act_j2000_radians) {
        causes |= AstScanMotionCause::unsupported_field_registry;
    }
    if (!(source.telescope_scope() == source.admitted_detector_scope())) {
        causes |= AstScanMotionCause::observation_scope_mismatch;
    }
    return causes;
}

void validate_engineering_schedule(
    const AstScanMotionSource &source,
    std::span<const AstScanMotionProcessingSpan> schedule) {
    if (schedule.empty()) return;
    std::vector<AstScanMotionProcessingSpan> ordered(schedule.begin(),
                                                      schedule.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto &lhs,
                                                 const auto &rhs) {
        return lhs.first_record < rhs.first_record;
    });
    AstTelescopeRecord expected = source.first_record();
    for (const auto &span : ordered) {
        if (span.first_record != expected ||
            span.first_record >= span.past_last_record ||
            span.past_last_record > source.past_last_record()) {
            throw std::invalid_argument(
                "AST processing schedule must exactly partition source support");
        }
        expected = span.past_last_record;
    }
    if (expected != source.past_last_record()) {
        throw std::invalid_argument(
            "AST processing schedule omits source support");
    }
}

}  // namespace

std::shared_ptr<const AstScanMotionSource> AstScanMotionSource::admit(
    NativeObservationScope telescope_scope,
    NativeObservationScope admitted_detector_scope,
    AstTelescopeRecord first_record, AstScanMotionSourceMetadata metadata,
    Eigen::VectorXd producer_times_unix_sec,
    Eigen::VectorXd source_ra_act_rad,
    Eigen::VectorXd source_dec_act_rad) {
    return std::shared_ptr<const AstScanMotionSource>(new AstScanMotionSource{
        telescope_scope, admitted_detector_scope, first_record,
        std::move(metadata), std::move(producer_times_unix_sec),
        std::move(source_ra_act_rad), std::move(source_dec_act_rad)});
}

AstScanMotionSource::AstScanMotionSource(
    NativeObservationScope telescope_scope,
    NativeObservationScope admitted_detector_scope,
    AstTelescopeRecord first_record, AstScanMotionSourceMetadata metadata,
    Eigen::VectorXd producer_times_unix_sec,
    Eigen::VectorXd source_ra_act_rad,
    Eigen::VectorXd source_dec_act_rad)
    : telescope_scope_{telescope_scope},
      admitted_detector_scope_{admitted_detector_scope},
      first_record_{first_record}, metadata_{std::move(metadata)},
      producer_times_unix_sec_{std::move(producer_times_unix_sec)},
      source_ra_act_rad_{std::move(source_ra_act_rad)},
      source_dec_act_rad_{std::move(source_dec_act_rad)} {
    if (first_record_ < 0 || producer_times_unix_sec_.size() <= 0 ||
        producer_times_unix_sec_.size() != source_ra_act_rad_.size() ||
        producer_times_unix_sec_.size() != source_dec_act_rad_.size() ||
        metadata_.source_artifact_identity.empty() ||
        static_cast<std::uint64_t>(producer_times_unix_sec_.size()) >
            static_cast<std::uint64_t>(
                std::numeric_limits<AstTelescopeRecord>::max() -
                first_record_)) {
        throw std::invalid_argument(
            "AST scan-motion source identity or shape is incomplete");
    }
}

const NativeObservationScope &AstScanMotionSource::telescope_scope()
    const noexcept {
    return telescope_scope_;
}

const NativeObservationScope &AstScanMotionSource::admitted_detector_scope()
    const noexcept {
    return admitted_detector_scope_;
}

AstTelescopeRecord AstScanMotionSource::first_record() const noexcept {
    return first_record_;
}

AstTelescopeRecord AstScanMotionSource::past_last_record() const noexcept {
    return first_record_ +
        static_cast<AstTelescopeRecord>(producer_times_unix_sec_.size());
}

std::size_t AstScanMotionSource::record_count() const noexcept {
    return static_cast<std::size_t>(producer_times_unix_sec_.size());
}

const AstScanMotionSourceMetadata &AstScanMotionSource::metadata()
    const noexcept {
    return metadata_;
}

const Eigen::VectorXd &AstScanMotionSource::producer_times_unix_sec()
    const noexcept {
    return producer_times_unix_sec_;
}

const Eigen::VectorXd &AstScanMotionSource::source_ra_act_rad()
    const noexcept {
    return source_ra_act_rad_;
}

const Eigen::VectorXd &AstScanMotionSource::source_dec_act_rad()
    const noexcept {
    return source_dec_act_rad_;
}

AstTelescopeRecordIdentity AstScanMotionSource::identity(
    AstTelescopeRecord record) const {
    (void)local_index(record);
    return {telescope_scope_, record};
}

std::size_t AstScanMotionSource::local_index(
    AstTelescopeRecord record) const {
    if (record < first_record_ || record >= past_last_record()) {
        throw std::out_of_range(
            "telescope record is outside AST source support");
    }
    return static_cast<std::size_t>(record - first_record_);
}

bool AstScanMotionIdentityBinding::complete() const noexcept {
    return requested != 0 && effective != 0 && observation_resolved != 0 &&
           realized != 0;
}

AstScanMotionCause AstScanMotionDerivedRecord::causes() const noexcept {
    return causes_;
}

std::int32_t AstScanMotionDerivedRecord::continuity_run() const noexcept {
    return continuity_run_;
}

bool AstScanMotionDerivedRecord::raw_direction_structurally_valid()
    const noexcept {
    return (flags_ & raw_direction_valid_bit) != 0;
}

bool AstScanMotionDerivedRecord::telemetry_quality_classified()
    const noexcept {
    return (flags_ & quality_classified_bit) != 0;
}

bool AstScanMotionDerivedRecord::telemetry_defect() const noexcept {
    return (flags_ & telemetry_defect_bit) != 0;
}

bool AstScanMotionDerivedRecord::realized_direction_valid() const noexcept {
    return raw_direction_structurally_valid() &&
           telemetry_quality_classified() && !telemetry_defect();
}

bool AstScanMotionDerivedRecord::derivative_valid() const noexcept {
    return (flags_ & derivative_valid_bit) != 0;
}

double AstScanMotionDerivedRecord::telemetry_residual_arcsec()
    const noexcept {
    return telemetry_residual_arcsec_;
}

double AstScanMotionDerivedRecord::east_velocity_arcsec_per_sec()
    const noexcept {
    return east_velocity_arcsec_per_sec_;
}

double AstScanMotionDerivedRecord::north_velocity_arcsec_per_sec()
    const noexcept {
    return north_velocity_arcsec_per_sec_;
}

double AstScanMotionDerivedRecord::scalar_speed_arcsec_per_sec()
    const noexcept {
    return scalar_speed_arcsec_per_sec_;
}

AstScanMotionProduct::AstScanMotionProduct(
    std::shared_ptr<const AstScanMotionSource> source,
    AstScanMotionIdentityBinding identity_binding,
    std::vector<AstScanMotionDerivedRecord> records,
    AstScanMotionScanSummary scan_summary,
    bool source_time_axis_mapping_eligible)
    : source_{std::move(source)}, identity_binding_{identity_binding},
      records_{std::move(records)}, scan_summary_{scan_summary},
      source_time_axis_mapping_eligible_{source_time_axis_mapping_eligible} {}

const std::shared_ptr<const AstScanMotionSource> &
AstScanMotionProduct::source_handle() const noexcept {
    return source_;
}

const NativeObservationScope &AstScanMotionProduct::scope() const noexcept {
    return source_->telescope_scope();
}

const AstScanMotionIdentityBinding &
AstScanMotionProduct::identity_binding() const noexcept {
    return identity_binding_;
}

std::size_t AstScanMotionProduct::record_count() const noexcept {
    return records_.size();
}

const AstScanMotionDerivedRecord &AstScanMotionProduct::record(
    AstTelescopeRecord record) const {
    return records_.at(source_->local_index(record));
}

const AstScanMotionDerivedRecord &AstScanMotionProduct::record_at_local(
    std::size_t local_index) const {
    return records_.at(local_index);
}

AstTelescopeRecord AstScanMotionProduct::record_identity(
    std::size_t local_index) const {
    if (local_index >= records_.size()) {
        throw std::out_of_range("AST local record is outside product support");
    }
    return source_->first_record() +
        static_cast<AstTelescopeRecord>(local_index);
}

std::optional<AstScanMotionSupport> AstScanMotionProduct::telemetry_support(
    AstTelescopeRecord record) const {
    return support(record, false);
}

std::optional<AstScanMotionSupport> AstScanMotionProduct::derivative_support(
    AstTelescopeRecord record) const {
    return support(record, true);
}

std::optional<AstScanMotionSupport> AstScanMotionProduct::support(
    AstTelescopeRecord record, bool require_derivative) const {
    const auto center = source_->local_index(record);
    const auto &derived = records_.at(center);
    if (!same_complete_window(records_, center) ||
        (!require_derivative &&
         has_cause(derived.causes(), AstScanMotionCause::
                                          telemetry_quality_support_unavailable)) ||
        (require_derivative &&
         has_cause(derived.causes(), AstScanMotionCause::
                                          derivative_support_intersects_invalidity))) {
        return std::nullopt;
    }
    const auto first = center - half_window;
    const auto last = center + half_window;
    return AstScanMotionSupport{
        record_identity(first), record_identity(last) + 1,
        source_->producer_times_unix_sec()(static_cast<Eigen::Index>(first)),
        source_->producer_times_unix_sec()(static_cast<Eigen::Index>(last))};
}

const AstScanMotionScanSummary &AstScanMotionProduct::scan_summary()
    const noexcept {
    return scan_summary_;
}

AstScanMotionMemoryEvidence AstScanMotionProduct::memory_evidence()
    const noexcept {
    return {records_.size() * sizeof(AstScanMotionDerivedRecord), 1, 2};
}

bool AstScanMotionProduct::source_time_axis_mapping_eligible()
    const noexcept {
    return source_time_axis_mapping_eligible_;
}

std::shared_ptr<const AstScanMotionProduct> build_ast_scan_motion_product(
    std::shared_ptr<const AstScanMotionSource> source,
    AstScanMotionIdentityBinding identity_binding,
    std::span<const AstScanMotionProcessingSpan> engineering_schedule) {
    if (!source || !identity_binding.complete()) {
        throw std::invalid_argument(
            "AST scan-motion build requires complete source and lifecycle identity");
    }
    validate_engineering_schedule(*source, engineering_schedule);

    std::vector<AstScanMotionDerivedRecord> records(source->record_count());
    AstScanMotionScanSummary summary;
    summary.record_count = records.size();
    const auto family_causes = source_family_causes(*source);
    if (has_any_cause(family_causes)) {
        for (auto &record : records) record.causes_ = family_causes;
        summary.causes = family_causes |
            AstScanMotionCause::scan_maximum_incomplete;
        return std::shared_ptr<const AstScanMotionProduct>(
            new AstScanMotionProduct{
                std::move(source), identity_binding, std::move(records),
                summary, false});
    }

    const auto &times = source->producer_times_unix_sec();
    const auto &ra = source->source_ra_act_rad();
    const auto &dec = source->source_dec_act_rad();
    bool source_times_strictly_increasing = true;
    bool coverage_incomplete = false;
    std::int32_t next_run = 0;

    for (std::size_t index = 0; index < records.size(); ++index) {
        auto &record = records[index];
        const auto row = static_cast<Eigen::Index>(index);
        if (!std::isfinite(times(row))) {
            record.causes_ |= AstScanMotionCause::nonfinite_telescope_time;
            source_times_strictly_increasing = false;
            coverage_incomplete = true;
            continue;
        }
        if (!direction_is_structurally_valid(ra(row), dec(row))) {
            record.causes_ |= AstScanMotionCause::
                                  nonfinite_or_unnormalizable_direction;
            coverage_incomplete = true;
            continue;
        }
        bool begins_run = index == 0 ||
            records[index - 1].continuity_run_ < 0;
        if (index > 0 && std::isfinite(times(row - 1))) {
            const double dt = times(row) - times(row - 1);
            if (!(dt > 0.0)) {
                record.causes_ |=
                    AstScanMotionCause::nonmonotonic_telescope_time;
                source_times_strictly_increasing = false;
                coverage_incomplete = true;
                continue;
            }
            if (!ast_scan_motion_continuous_interval(dt)) {
                record.causes_ |= AstScanMotionCause::telescope_gap;
                begins_run = true;
                coverage_incomplete = true;
            }
        }
        if (begins_run) {
            record.continuity_run_ = next_run++;
        }
        else {
            record.continuity_run_ =
                records[index - 1].continuity_run_;
        }
        record.flags_ |= AstScanMotionDerivedRecord::raw_direction_valid_bit;
    }
    summary.continuity_run_count = static_cast<std::size_t>(next_run);

    for (std::size_t center = 0; center < records.size(); ++center) {
        auto &record = records[center];
        if (!record.raw_direction_structurally_valid()) continue;
        if (!same_complete_window(records, center)) {
            record.causes_ |= AstScanMotionCause::
                                  telemetry_quality_support_unavailable;
            continue;
        }
        const auto tangent = tangent_window(*source, center);
        if (!tangent) {
            record.causes_ |=
                AstScanMotionCause::spherical_topology_unavailable;
            coverage_incomplete = true;
            continue;
        }
        std::array<double, pair_count> east_slopes;
        std::array<double, pair_count> north_slopes;
        std::size_t pair = 0;
        const auto first = center - half_window;
        for (std::size_t left = 0; left < window_size; ++left) {
            for (std::size_t right = left + 1; right < window_size; ++right) {
                const double dt =
                    times(static_cast<Eigen::Index>(first + right)) -
                    times(static_cast<Eigen::Index>(first + left));
                east_slopes[pair] =
                    ((*tangent)[right].east_rad -
                     (*tangent)[left].east_rad) /
                    dt;
                north_slopes[pair] =
                    ((*tangent)[right].north_rad -
                     (*tangent)[left].north_rad) /
                    dt;
                ++pair;
            }
        }
        const double east_slope = median(std::move(east_slopes));
        const double north_slope = median(std::move(north_slopes));
        std::array<double, window_size> east_intercepts;
        std::array<double, window_size> north_intercepts;
        for (std::size_t offset = 0; offset < window_size; ++offset) {
            const double dt =
                times(static_cast<Eigen::Index>(first + offset)) -
                times(static_cast<Eigen::Index>(center));
            east_intercepts[offset] =
                (*tangent)[offset].east_rad - east_slope * dt;
            north_intercepts[offset] =
                (*tangent)[offset].north_rad - north_slope * dt;
        }
        record.telemetry_residual_arcsec_ =
            std::hypot(median(std::move(east_intercepts)),
                       median(std::move(north_intercepts))) *
            radians_to_arcsec;
        if (!std::isfinite(record.telemetry_residual_arcsec_)) {
            record.causes_ |=
                AstScanMotionCause::spherical_topology_unavailable;
            coverage_incomplete = true;
            continue;
        }
        record.flags_ |=
            AstScanMotionDerivedRecord::quality_classified_bit;
        if (ast_scan_motion_telemetry_defect(
                record.telemetry_residual_arcsec_)) {
            record.flags_ |=
                AstScanMotionDerivedRecord::telemetry_defect_bit;
            record.causes_ |= AstScanMotionCause::telemetry_defect;
            ++summary.telemetry_defect_count;
        }
    }

    for (std::size_t center = 0; center < records.size(); ++center) {
        auto &record = records[center];
        bool support_valid = same_complete_window(records, center) &&
            record.telemetry_quality_classified() &&
            !record.telemetry_defect();
        if (support_valid) {
            for (std::size_t index = center - half_window;
                 index <= center + half_window; ++index) {
                if (!records[index].telemetry_quality_classified() ||
                    records[index].telemetry_defect()) {
                    support_valid = false;
                    break;
                }
            }
        }
        if (!support_valid) {
            if (record.raw_direction_structurally_valid()) {
                record.causes_ |= AstScanMotionCause::
                                      derivative_support_intersects_invalidity;
            }
            continue;
        }
        const auto tangent = tangent_window(*source, center);
        if (!tangent) {
            record.causes_ |=
                AstScanMotionCause::spherical_topology_unavailable |
                AstScanMotionCause::derivative_support_intersects_invalidity;
            coverage_incomplete = true;
            continue;
        }

        Eigen::Matrix<double, window_size, 3> design;
        Eigen::Matrix<double, window_size, 2> response;
        const auto first = center - half_window;
        for (std::size_t offset = 0; offset < window_size; ++offset) {
            const double dt =
                times(static_cast<Eigen::Index>(first + offset)) -
                times(static_cast<Eigen::Index>(center));
            design(static_cast<Eigen::Index>(offset), 0) = 1.0;
            design(static_cast<Eigen::Index>(offset), 1) = dt;
            design(static_cast<Eigen::Index>(offset), 2) = dt * dt;
            response(static_cast<Eigen::Index>(offset), 0) =
                (*tangent)[offset].east_rad;
            response(static_cast<Eigen::Index>(offset), 1) =
                (*tangent)[offset].north_rad;
        }
        Eigen::ColPivHouseholderQR<decltype(design)> qr(design);
        if (qr.rank() != 3) {
            record.causes_ |=
                AstScanMotionCause::rank_deficient_derivative_fit;
            coverage_incomplete = true;
            continue;
        }
        const Eigen::Matrix<double, 3, 2> coefficients = qr.solve(response);
        const double east = coefficients(1, 0) * radians_to_arcsec;
        const double north = coefficients(1, 1) * radians_to_arcsec;
        const double speed = std::hypot(east, north);
        if (!std::isfinite(east) || !std::isfinite(north) ||
            !std::isfinite(speed)) {
            record.causes_ |= AstScanMotionCause::nonfinite_derivative;
            coverage_incomplete = true;
            continue;
        }
        record.east_velocity_arcsec_per_sec_ = east;
        record.north_velocity_arcsec_per_sec_ = north;
        record.scalar_speed_arcsec_per_sec_ = speed;
        record.flags_ |= AstScanMotionDerivedRecord::derivative_valid_bit;
        ++summary.derivative_valid_record_count;
        if (ast_scan_motion_speed_admitted(speed)) {
            ++summary.admitted_candidate_count;
            if (summary.maximizing_record < 0 ||
                speed > summary.maximum_speed_arcsec_per_sec) {
                summary.maximizing_record = source->first_record() +
                    static_cast<AstTelescopeRecord>(center);
                summary.maximum_speed_arcsec_per_sec = speed;
            }
        }
    }

    if (coverage_incomplete) {
        summary.causes |= AstScanMotionCause::scan_maximum_incomplete;
        summary.maximizing_record = -1;
        summary.maximum_speed_arcsec_per_sec = 0.0;
    }
    else if (summary.admitted_candidate_count == 0) {
        summary.causes |= AstScanMotionCause::no_admitted_scan_motion;
    }
    else {
        summary.maximum_available = true;
    }

    return std::shared_ptr<const AstScanMotionProduct>(
        new AstScanMotionProduct{
            std::move(source), identity_binding, std::move(records), summary,
            source_times_strictly_increasing});
}

}  // namespace citlali::pipeline
