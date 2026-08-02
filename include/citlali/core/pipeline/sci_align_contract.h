#pragma once

#include <Eigen/Dense>

#include <citlali/core/config/interface_sync_config.h>
#include <citlali/core/utils/slot_assignment.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline::sci_align {

// The types in this header describe the bounded SCI-ALIGN-001 contract.  They
// deliberately do not assign physical meaning to the legacy timestamp or its
// sample event.  The reconstructed coordinate is suitable only for preserving
// the admitted relative-timing behavior.

inline constexpr double reference_sample_frequency_hz = 122.0703125;
inline constexpr double legacy_counter_adjustment_ticks = 4294967295.0;

inline int parse_toltec_interface_identity(const std::string &identity) {
    constexpr const char *prefix = "toltec";
    if (identity.rfind(prefix, 0) != 0 || identity.size() <= 6) {
        throw std::runtime_error(
            "detector input interface must be an exact toltecN identity: " +
            identity);
    }
    const std::string suffix = identity.substr(6);
    if (!std::all_of(suffix.begin(), suffix.end(),
                     [](unsigned char value) {
                         return std::isdigit(value) != 0;
                     })) {
        throw std::runtime_error(
            "detector input interface has a nonnumeric TolTEC suffix: " +
            identity);
    }
    const int parsed = std::stoi(suffix);
    if (parsed < 0 || parsed >= static_cast<int>(
            citlali::config::toltec_interface_count) ||
        identity != "toltec" + std::to_string(parsed)) {
        throw std::runtime_error(
            "detector input interface is outside canonical toltec0..toltec12: " +
            identity);
    }
    return parsed;
}

inline bool machine_equal(double lhs, double rhs, double ulp_scale = 64.0) {
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return false;
    }
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <=
           ulp_scale * std::numeric_limits<double>::epsilon() * scale;
}

inline void require_finite_positive(double value, const char *name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(std::string{name} +
                                    " must be finite and positive");
    }
}

inline void require_strictly_increasing(const Eigen::VectorXd &coordinates,
                                        const char *name) {
    if (coordinates.size() == 0) {
        throw std::invalid_argument(std::string{name} + " must not be empty");
    }
    for (Eigen::Index i = 0; i < coordinates.size(); ++i) {
        if (!std::isfinite(coordinates[i])) {
            throw std::invalid_argument(std::string{name} +
                                        " contains a non-finite coordinate");
        }
        if (i > 0 && !(coordinates[i] > coordinates[i - 1])) {
            throw std::invalid_argument(std::string{name} +
                                        " must be strictly increasing");
        }
    }
}

enum class NativeRateFactor { half, one, two, four };

inline double rate_multiplier(NativeRateFactor factor) {
    switch (factor) {
        case NativeRateFactor::half:
            return 0.5;
        case NativeRateFactor::one:
            return 1.0;
        case NativeRateFactor::two:
            return 2.0;
        case NativeRateFactor::four:
            return 4.0;
    }
    throw std::logic_error("unknown native rate factor");
}

struct NativeTimingHeader {
    double fpga_frequency_hz = std::numeric_limits<double>::quiet_NaN();
    double accumulation_length_ticks =
        std::numeric_limits<double>::quiet_NaN();
    double sample_frequency_hz = std::numeric_limits<double>::quiet_NaN();
};

struct ValidatedNativeRate {
    NativeRateFactor factor = NativeRateFactor::one;
    double multiplier = 1.0;
    double sample_frequency_hz = reference_sample_frequency_hz;
    double cadence_seconds = 1.0 / reference_sample_frequency_hz;
    double exclusive_half_cell_seconds =
        0.5 / reference_sample_frequency_hz;
    std::uint64_t accumulation_length_ticks = 0;
};

inline ValidatedNativeRate validate_native_timing_header(
    const NativeTimingHeader &header) {
    require_finite_positive(header.fpga_frequency_hz,
                            "FpgaFreq");
    require_finite_positive(header.accumulation_length_ticks,
                            "AccumLen");
    require_finite_positive(header.sample_frequency_hz,
                            "SampleFreq");

    const double rounded_accumulation =
        std::floor(header.accumulation_length_ticks + 0.5);
    const double exclusive_uint64_upper =
        std::ldexp(1.0, std::numeric_limits<std::uint64_t>::digits);
    if (!machine_equal(header.accumulation_length_ticks,
                       rounded_accumulation) ||
        rounded_accumulation >= exclusive_uint64_upper) {
        throw std::invalid_argument("AccumLen must be an integer tick count");
    }

    const double cadence_from_counters =
        header.accumulation_length_ticks / header.fpga_frequency_hz;
    const double frequency_from_counters = 1.0 / cadence_from_counters;
    if (!machine_equal(header.sample_frequency_hz,
                       frequency_from_counters)) {
        throw std::invalid_argument(
            "FpgaFreq, AccumLen, and SampleFreq are inconsistent");
    }

    const NativeRateFactor factors[] = {
        NativeRateFactor::half, NativeRateFactor::one,
        NativeRateFactor::two, NativeRateFactor::four};
    std::optional<NativeRateFactor> selected;
    for (const auto factor : factors) {
        const double expected =
            rate_multiplier(factor) * reference_sample_frequency_hz;
        if (machine_equal(header.sample_frequency_hz, expected)) {
            selected = factor;
            break;
        }
    }
    if (!selected.has_value()) {
        throw std::invalid_argument(
            "SampleFreq is outside the admitted 0.5x/1x/2x/4x family");
    }

    const double multiplier = rate_multiplier(*selected);
    const double cadence =
        1.0 / (multiplier * reference_sample_frequency_hz);
    if (!machine_equal(cadence, cadence_from_counters)) {
        throw std::invalid_argument(
            "counter-derived cadence does not select the declared rate");
    }

    return {*selected,
            multiplier,
            multiplier * reference_sample_frequency_hz,
            cadence,
            cadence / 2.0,
            static_cast<std::uint64_t>(rounded_accumulation)};
}

inline ValidatedNativeRate validate_bounded_production_native_timing_header(
    const NativeTimingHeader &header) {
    const auto validated = validate_native_timing_header(header);
    if (validated.factor != NativeRateFactor::one) {
        throw std::invalid_argument(
            "native detector rates other than 1x remain "
            "production-evidence-pending");
    }
    return validated;
}

inline ValidatedNativeRate require_common_native_rate(
    const std::vector<NativeTimingHeader> &headers) {
    if (headers.empty()) {
        throw std::invalid_argument("at least one detector header is required");
    }
    const auto first = validate_native_timing_header(headers.front());
    for (std::size_t i = 1; i < headers.size(); ++i) {
        const auto current = validate_native_timing_header(headers[i]);
        if (current.factor != first.factor) {
            throw std::invalid_argument(
                "mixed native detector rates are not admitted");
        }
    }
    return first;
}

struct PacketGap {
    Eigen::Index row_before = 0;
    std::int64_t first_missing_packet = 0;
    std::uint64_t missing_packet_count = 0;
};

struct LegacyTimestampResult {
    Eigen::VectorXd seconds;
    std::vector<PacketGap> packet_gaps;
    bool producer_clock_authority_available = false;
    bool integration_event_authority_available = false;
    bool absolute_timing_precision_available = false;
};

inline LegacyTimestampResult reconstruct_legacy_detector_timestamps(
    const Eigen::Ref<const Eigen::MatrixXd> &timestamp_fields,
    const std::vector<std::int64_t> &packet_counts,
    double fpga_frequency_hz) {
    require_finite_positive(fpga_frequency_hz, "FpgaFreq");
    if (timestamp_fields.rows() == 0 || timestamp_fields.cols() != 6) {
        throw std::invalid_argument(
            "legacy detector timestamp input must have nonzero rows and six columns");
    }
    if (static_cast<std::size_t>(timestamp_fields.rows()) !=
        packet_counts.size()) {
        throw std::invalid_argument(
            "PacketCount and legacy timestamp rows have different lengths");
    }
    if (!timestamp_fields.allFinite()) {
        throw std::invalid_argument(
            "legacy detector timestamp input contains non-finite fields");
    }

    const double anchor_expression =
        timestamp_fields(0, 0) + timestamp_fields(0, 5) * 1.0e-9 - 0.5;
    if (anchor_expression <
            static_cast<double>(std::numeric_limits<int>::min()) ||
        anchor_expression >
            static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("legacy timestamp anchor exceeds C++ int");
    }
    // Deliberately preserve the admitted C++ int truncation compatibility.
    const int anchor = static_cast<int>(anchor_expression);

    LegacyTimestampResult result;
    result.seconds.resize(timestamp_fields.rows());
    for (Eigen::Index row = 0; row < timestamp_fields.rows(); ++row) {
        double delta_ticks =
            timestamp_fields(row, 2) - timestamp_fields(row, 4);
        if (timestamp_fields(row, 2) < timestamp_fields(row, 4)) {
            delta_ticks += legacy_counter_adjustment_ticks;
        }
        result.seconds[row] = static_cast<double>(anchor) +
                              timestamp_fields(row, 1) +
                              delta_ticks / fpga_frequency_hz;
        if (!std::isfinite(result.seconds[row])) {
            throw std::invalid_argument(
                "legacy timestamp reconstruction produced a non-finite value");
        }

        if (row > 0) {
            if (packet_counts[static_cast<std::size_t>(row)] <=
                packet_counts[static_cast<std::size_t>(row - 1)]) {
                throw std::invalid_argument(
                    "PacketCount is duplicate, decreasing, or reset");
            }
            const auto difference =
                static_cast<std::uint64_t>(
                    packet_counts[static_cast<std::size_t>(row)]) -
                static_cast<std::uint64_t>(
                    packet_counts[static_cast<std::size_t>(row - 1)]);
            if (difference > 1) {
                result.packet_gaps.push_back(
                    {row - 1,
                     packet_counts[static_cast<std::size_t>(row - 1)] + 1,
                     difference - 1});
            }
        }
    }
    require_strictly_increasing(result.seconds,
                                "reconstructed detector timestamps");
    return result;
}

enum class ClockCoordinateStage { native_legacy, reference_clock };

struct ClockCoordinates {
    Eigen::VectorXd seconds;
    ClockCoordinateStage stage = ClockCoordinateStage::native_legacy;
};

struct InterfaceOffset {
    double seconds = 0.0;
    bool authority_resolved = false;
    std::string source;
    std::string reference_interface;
    std::string unit = "s";
    std::string sign_convention = "positive_add";
};

inline ClockCoordinates apply_interface_offset_once(
    ClockCoordinates native, const InterfaceOffset &offset) {
    if (native.stage != ClockCoordinateStage::native_legacy) {
        throw std::invalid_argument(
            "interface offset has already been applied");
    }
    require_strictly_increasing(native.seconds, "native interface time");
    if (!std::isfinite(offset.seconds)) {
        throw std::invalid_argument("interface offset must be finite");
    }
    if (offset.unit != "s" || offset.sign_convention != "positive_add") {
        throw std::invalid_argument(
            "interface offset requires seconds and positive-add convention");
    }
    if (offset.seconds != 0.0 &&
        (!offset.authority_resolved || offset.source.empty() ||
         offset.reference_interface.empty())) {
        throw std::invalid_argument(
            "nonzero interface offset lacks resolved authority");
    }
    native.seconds.array() += offset.seconds;
    require_strictly_increasing(native.seconds, "reference-clock time");
    native.stage = ClockCoordinateStage::reference_clock;
    return native;
}

inline std::int64_t round_half_up_slot(double coordinate) {
    return citlali::utils::round_half_up_slot(coordinate);
}

struct DetectorInterfaceCoordinates {
    std::string interface_id;
    NativeTimingHeader timing_header;
    ClockCoordinates corrected_time;
};

struct SlotAssignment {
    Eigen::Index native_row = 0;
    std::int64_t global_slot = 0;
    double native_timestamp_seconds = 0.0;
    double residual_seconds = 0.0;
};

struct HalfOpenSlotInterval {
    std::int64_t begin = 0;
    std::int64_t end = 0;

    std::uint64_t size() const {
        return end > begin
            ? static_cast<std::uint64_t>(end) -
                  static_cast<std::uint64_t>(begin)
            : 0;
    }
};

struct DetectorInterfaceMapping {
    std::string interface_id;
    Eigen::Index native_row_count = 0;
    std::int64_t first_global_slot = 0;
    std::int64_t last_global_slot = -1;
    double minimum_residual_seconds = 0.0;
    double maximum_residual_seconds = 0.0;
    double maximum_absolute_residual_seconds = 0.0;
    // Expanded row mappings are diagnostic/as-requested. Production setup
    // retains only the compact generative fields above.
    std::vector<SlotAssignment> assignments;
    std::optional<HalfOpenSlotInterval> leading_unavailable;
    std::optional<HalfOpenSlotInterval> trailing_unavailable;
};

struct DetectorLattice {
    double phase_seconds = 0.0;
    double cadence_seconds = 0.0;
    double exclusive_half_cell_seconds = 0.0;
    std::int64_t first_global_slot = 0;
    std::int64_t last_global_slot = -1;
    ValidatedNativeRate native_rate;
    std::vector<DetectorInterfaceMapping> interfaces;

    std::uint64_t slot_count() const {
        if (last_global_slot < first_global_slot) {
            return 0;
        }
        // Unsigned subtraction gives the exact nonnegative distance for the
        // ordered int64 endpoints without first overflowing signed int64.
        const auto distance =
            static_cast<std::uint64_t>(last_global_slot) -
            static_cast<std::uint64_t>(first_global_slot);
        if (distance == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error(
                "detector lattice slot count exceeds uint64 range");
        }
        return distance + 1;
    }

    double time_for_global_slot(std::int64_t slot) const {
        return phase_seconds + static_cast<double>(slot) * cadence_seconds;
    }
};

inline DetectorLattice build_detector_union_lattice(
    const std::vector<DetectorInterfaceCoordinates> &interfaces,
    bool retain_expanded_assignments = true) {
    if (interfaces.empty()) {
        throw std::invalid_argument(
            "at least one detector interface is required");
    }
    std::vector<NativeTimingHeader> headers;
    headers.reserve(interfaces.size());
    double phase = std::numeric_limits<double>::lowest();
    for (const auto &interface : interfaces) {
        if (interface.interface_id.empty()) {
            throw std::invalid_argument("detector interface identity is empty");
        }
        if (interface.corrected_time.stage !=
            ClockCoordinateStage::reference_clock) {
            throw std::invalid_argument(
                "detector lattice requires offset-resolved reference-clock time");
        }
        require_strictly_increasing(interface.corrected_time.seconds,
                                    "detector interface time");
        headers.push_back(interface.timing_header);
        phase = std::max(phase, interface.corrected_time.seconds[0]);
    }
    const auto rate = require_common_native_rate(headers);

    DetectorLattice lattice;
    lattice.phase_seconds = phase;
    lattice.cadence_seconds = rate.cadence_seconds;
    lattice.exclusive_half_cell_seconds =
        rate.exclusive_half_cell_seconds;
    lattice.native_rate = rate;
    lattice.first_global_slot = std::numeric_limits<std::int64_t>::max();
    lattice.last_global_slot = std::numeric_limits<std::int64_t>::min();
    lattice.interfaces.reserve(interfaces.size());

    for (const auto &interface : interfaces) {
        DetectorInterfaceMapping mapping;
        mapping.interface_id = interface.interface_id;
        mapping.native_row_count = interface.corrected_time.seconds.size();
        mapping.minimum_residual_seconds =
            std::numeric_limits<double>::max();
        mapping.maximum_residual_seconds =
            std::numeric_limits<double>::lowest();
        if (retain_expanded_assignments) {
            mapping.assignments.reserve(static_cast<std::size_t>(
                interface.corrected_time.seconds.size()));
        }
        std::optional<std::int64_t> previous_slot;
        for (Eigen::Index row = 0;
             row < interface.corrected_time.seconds.size(); ++row) {
            const double timestamp = interface.corrected_time.seconds[row];
            const double coordinate = (timestamp - phase) /
                                      lattice.cadence_seconds;
            const auto slot = round_half_up_slot(coordinate);
            const double nominal = phase + static_cast<double>(slot) *
                                               lattice.cadence_seconds;
            const double residual = timestamp - nominal;
            if (!(std::abs(residual) <
                  lattice.exclusive_half_cell_seconds) ||
                machine_equal(std::abs(residual),
                              lattice.exclusive_half_cell_seconds)) {
                throw std::invalid_argument(
                    "detector row is on or outside the exclusive half-cell boundary");
            }
            if (previous_slot.has_value() && slot == *previous_slot) {
                throw std::invalid_argument(
                    "multiple native rows from one interface occupy one slot");
            }
            if (previous_slot.has_value() && slot < *previous_slot) {
                throw std::invalid_argument(
                    "detector slot identities are not increasing");
            }
            if (retain_expanded_assignments) {
                mapping.assignments.push_back(
                    {row, slot, timestamp, residual});
            }
            if (!previous_slot.has_value()) {
                mapping.first_global_slot = slot;
            }
            mapping.last_global_slot = slot;
            mapping.minimum_residual_seconds = std::min(
                mapping.minimum_residual_seconds, residual);
            mapping.maximum_residual_seconds = std::max(
                mapping.maximum_residual_seconds, residual);
            mapping.maximum_absolute_residual_seconds = std::max(
                mapping.maximum_absolute_residual_seconds,
                std::abs(residual));
            previous_slot = slot;
            lattice.first_global_slot =
                std::min(lattice.first_global_slot, slot);
            lattice.last_global_slot =
                std::max(lattice.last_global_slot, slot);
        }
        lattice.interfaces.push_back(std::move(mapping));
    }

    for (auto &mapping : lattice.interfaces) {
        if (lattice.last_global_slot ==
            std::numeric_limits<std::int64_t>::max()) {
            throw std::overflow_error(
                "detector union cannot represent its half-open stop slot");
        }
        const auto first = mapping.first_global_slot;
        const auto last = mapping.last_global_slot;
        if (first > lattice.first_global_slot) {
            mapping.leading_unavailable =
                HalfOpenSlotInterval{lattice.first_global_slot, first};
        }
        if (last < lattice.last_global_slot) {
            mapping.trailing_unavailable =
                HalfOpenSlotInterval{last + 1,
                                     lattice.last_global_slot + 1};
        }
    }
    return lattice;
}

enum class FieldTopology {
    continuous_scalar,
    circular,
    declared_half_open_step,
    exact_only
};

enum class Origin { original, synthesized, unavailable };
enum class Validity { valid, invalid };
enum class Method { exact, linear, circular, held, none };
enum class Reason {
    none,
    missing_stream,
    outside_support,
    nonfinite_source,
    support_span_exceeded,
    antipodal_ambiguous,
    operator_not_permitted,
    missing_detector_row,
    leading_gap,
    trailing_gap,
    gap_limit_exceeded,
    invalid_gap_endpoint
};
enum class DetailLevel { compact, expanded };

struct SourceWeight {
    Eigen::Index source_row = 0;
    double weight = 0.0;
};

struct CellQuality {
    Origin origin = Origin::unavailable;
    Validity validity = Validity::invalid;
    Method method = Method::none;
    Reason reason = Reason::outside_support;
    std::vector<SourceWeight> expanded_sources;

    bool available() const {
        return origin != Origin::unavailable && validity == Validity::valid;
    }
};

struct AlignedValue {
    double value = std::numeric_limits<double>::quiet_NaN();
    CellQuality quality;
};

struct FieldContract {
    FieldTopology topology = FieldTopology::exact_only;
    double maximum_support_span_seconds = 0.0;
    double circular_period = 0.0;
    std::optional<double> acquisition_end_seconds;
};

inline CellQuality unavailable_quality(Reason reason) {
    return {Origin::unavailable, Validity::invalid, Method::none, reason, {}};
}

inline void add_source_if_requested(CellQuality &quality,
                                    DetailLevel detail, Eigen::Index row,
                                    double weight) {
    if (detail == DetailLevel::expanded) {
        quality.expanded_sources.push_back({row, weight});
    }
}

inline double wrap_period(double value, double period) {
    double wrapped = std::fmod(value, period);
    if (wrapped < 0.0) {
        wrapped += period;
    }
    if (wrapped >= period) {
        wrapped -= period;
    }
    return wrapped;
}

inline double nearest_periodic_equivalent(double value, double reference,
                                          double period) {
    if (!std::isfinite(value) || !std::isfinite(reference) ||
        !std::isfinite(period) || !(period > 0.0)) {
        throw std::invalid_argument(
            "periodic representation requires finite values and positive period");
    }
    return value + std::round((reference - value) / period) * period;
}

inline AlignedValue align_field_at(
    const Eigen::Ref<const Eigen::VectorXd> &source_time,
    const Eigen::Ref<const Eigen::VectorXd> &source_value,
    double target_time, const FieldContract &contract,
    DetailLevel detail = DetailLevel::compact) {
    if (!std::isfinite(target_time)) {
        throw std::invalid_argument("target time must be finite");
    }
    if (source_time.size() != source_value.size()) {
        throw std::invalid_argument(
            "field coordinates and values have different lengths");
    }
    if (source_time.size() == 0) {
        return {std::numeric_limits<double>::quiet_NaN(),
                unavailable_quality(Reason::missing_stream)};
    }
    require_strictly_increasing(source_time, "field source time");

    if ((contract.topology == FieldTopology::continuous_scalar ||
         contract.topology == FieldTopology::circular ||
         contract.topology == FieldTopology::declared_half_open_step) &&
        (!std::isfinite(contract.maximum_support_span_seconds) ||
         contract.maximum_support_span_seconds <= 0.0)) {
        throw std::invalid_argument(
            "interpolated/held field requires a finite positive maximum support span");
    }
    if (contract.topology == FieldTopology::circular) {
        require_finite_positive(contract.circular_period, "circular period");
    }

    const double *first = source_time.data();
    const double *last = first + source_time.size();
    const double *position = std::lower_bound(first, last, target_time);
    const Eigen::Index upper = static_cast<Eigen::Index>(position - first);
    if (upper < source_time.size() && source_time[upper] == target_time) {
        CellQuality quality{Origin::original,
                            std::isfinite(source_value[upper])
                                ? Validity::valid
                                : Validity::invalid,
                            Method::exact,
                            std::isfinite(source_value[upper])
                                ? Reason::none
                                : Reason::nonfinite_source,
                            {}};
        add_source_if_requested(quality, detail, upper, 1.0);
        return {source_value[upper], std::move(quality)};
    }

    if (contract.topology == FieldTopology::declared_half_open_step) {
        if (upper == 0) {
            return {std::numeric_limits<double>::quiet_NaN(),
                    unavailable_quality(Reason::outside_support)};
        }
        const Eigen::Index held_row = upper - 1;
        const double interval_end =
            upper < source_time.size()
                ? source_time[upper]
                : contract.acquisition_end_seconds.value_or(
                      std::numeric_limits<double>::quiet_NaN());
        if (!std::isfinite(interval_end) || !(target_time < interval_end)) {
            return {std::numeric_limits<double>::quiet_NaN(),
                    unavailable_quality(Reason::outside_support)};
        }
        if (target_time - source_time[held_row] >
            contract.maximum_support_span_seconds) {
            return {std::numeric_limits<double>::quiet_NaN(),
                    unavailable_quality(Reason::support_span_exceeded)};
        }
        if (!std::isfinite(source_value[held_row])) {
            return {std::numeric_limits<double>::quiet_NaN(),
                    unavailable_quality(Reason::nonfinite_source)};
        }
        CellQuality quality{Origin::synthesized, Validity::valid,
                            Method::held, Reason::none, {}};
        add_source_if_requested(quality, detail, held_row, 1.0);
        return {source_value[held_row], std::move(quality)};
    }

    if (contract.topology == FieldTopology::exact_only) {
        return {std::numeric_limits<double>::quiet_NaN(),
                unavailable_quality(Reason::operator_not_permitted)};
    }
    if (upper == 0 || upper >= source_time.size()) {
        return {std::numeric_limits<double>::quiet_NaN(),
                unavailable_quality(Reason::outside_support)};
    }

    const Eigen::Index lower = upper - 1;
    const double span = source_time[upper] - source_time[lower];
    if (span > contract.maximum_support_span_seconds) {
        return {std::numeric_limits<double>::quiet_NaN(),
                unavailable_quality(Reason::support_span_exceeded)};
    }
    if (!std::isfinite(source_value[lower]) ||
        !std::isfinite(source_value[upper])) {
        return {std::numeric_limits<double>::quiet_NaN(),
                unavailable_quality(Reason::nonfinite_source)};
    }
    const double lambda = (target_time - source_time[lower]) / span;

    CellQuality quality{Origin::synthesized, Validity::valid,
                        contract.topology == FieldTopology::circular
                            ? Method::circular
                            : Method::linear,
                        Reason::none,
                        {}};
    add_source_if_requested(quality, detail, lower, 1.0 - lambda);
    add_source_if_requested(quality, detail, upper, lambda);

    if (contract.topology == FieldTopology::continuous_scalar) {
        return {(1.0 - lambda) * source_value[lower] +
                    lambda * source_value[upper],
                std::move(quality)};
    }

    double difference =
        std::fmod(source_value[upper] - source_value[lower],
                  contract.circular_period);
    if (difference <= -contract.circular_period / 2.0) {
        difference += contract.circular_period;
    } else if (difference > contract.circular_period / 2.0) {
        difference -= contract.circular_period;
    }
    if (machine_equal(std::abs(difference),
                      contract.circular_period / 2.0)) {
        return {std::numeric_limits<double>::quiet_NaN(),
                unavailable_quality(Reason::antipodal_ambiguous)};
    }
    const double shortest_arc_value =
        source_value[lower] + lambda * difference;
    const double source_representation_reference =
        (1.0 - lambda) * source_value[lower] +
        lambda * source_value[upper];
    return {nearest_periodic_equivalent(
                shortest_arc_value, source_representation_reference,
                contract.circular_period),
            std::move(quality)};
}

inline std::vector<AlignedValue> align_field_series(
    const Eigen::Ref<const Eigen::VectorXd> &source_time,
    const Eigen::Ref<const Eigen::VectorXd> &source_value,
    const Eigen::Ref<const Eigen::VectorXd> &target_time,
    const FieldContract &contract,
    DetailLevel detail = DetailLevel::compact) {
    std::vector<AlignedValue> result;
    result.reserve(static_cast<std::size_t>(target_time.size()));
    for (Eigen::Index i = 0; i < target_time.size(); ++i) {
        result.push_back(align_field_at(source_time, source_value,
                                        target_time[i], contract, detail));
    }
    return result;
}

struct DetectorSlotCell {
    bool has_native_row = false;
    double value = std::numeric_limits<double>::quiet_NaN();
    bool native_valid = false;
    Eigen::Index native_row = -1;
    double native_exposure_seconds = 0.0;
};

struct GapLimits {
    std::size_t maximum_missing_slots = 0;
    double maximum_missing_duration_seconds = 0.0;
};

enum class GapLocation { internal, leading, trailing };
enum class GapAction { linear_fill, unavailable };

struct GapRun {
    std::size_t begin = 0;
    std::size_t end = 0;
    GapLocation location = GapLocation::internal;
    GapAction action = GapAction::unavailable;
    Reason reason = Reason::none;
    double missing_duration_seconds = 0.0;
    std::optional<std::size_t> left_endpoint_slot;
    std::optional<std::size_t> right_endpoint_slot;

    std::size_t missing_slot_count() const { return end - begin; }
};

struct GapPlan {
    std::vector<GapRun> runs;
    std::size_t missing_slot_count = 0;
    std::size_t synthesized_slot_count = 0;
    std::size_t unavailable_slot_count = 0;
};

inline GapPlan plan_detector_gaps(const std::vector<DetectorSlotCell> &cells,
                                  double cadence_seconds,
                                  const GapLimits &limits) {
    require_finite_positive(cadence_seconds, "detector cadence");
    if (!std::isfinite(limits.maximum_missing_duration_seconds) ||
        limits.maximum_missing_duration_seconds < 0.0) {
        throw std::invalid_argument(
            "maximum missing duration must be finite and nonnegative");
    }

    GapPlan plan;
    std::size_t i = 0;
    while (i < cells.size()) {
        if (cells[i].has_native_row) {
            ++i;
            continue;
        }
        const std::size_t begin = i;
        while (i < cells.size() && !cells[i].has_native_row) {
            ++i;
        }
        const std::size_t end = i;
        GapRun run;
        run.begin = begin;
        run.end = end;
        run.missing_duration_seconds =
            static_cast<double>(end - begin) * cadence_seconds;
        run.location = begin == 0
                           ? GapLocation::leading
                           : (end == cells.size() ? GapLocation::trailing
                                                  : GapLocation::internal);
        if (run.location == GapLocation::leading) {
            run.reason = Reason::leading_gap;
        } else if (run.location == GapLocation::trailing) {
            run.reason = Reason::trailing_gap;
        } else {
            run.left_endpoint_slot = begin - 1;
            run.right_endpoint_slot = end;
            const bool within_limits =
                run.missing_slot_count() <= limits.maximum_missing_slots &&
                run.missing_duration_seconds <=
                    limits.maximum_missing_duration_seconds;
            if (!within_limits) {
                run.reason = Reason::gap_limit_exceeded;
            } else if (!cells[begin - 1].native_valid ||
                       !std::isfinite(cells[begin - 1].value) ||
                       !cells[end].native_valid ||
                       !std::isfinite(cells[end].value)) {
                run.reason = Reason::invalid_gap_endpoint;
            } else {
                run.action = GapAction::linear_fill;
                run.reason = Reason::none;
            }
        }
        plan.missing_slot_count += run.missing_slot_count();
        if (run.action == GapAction::linear_fill) {
            plan.synthesized_slot_count += run.missing_slot_count();
        } else {
            plan.unavailable_slot_count += run.missing_slot_count();
        }
        plan.runs.push_back(std::move(run));
    }
    return plan;
}

inline const GapRun *find_gap_run(const GapPlan &plan, std::size_t slot) {
    const auto it = std::lower_bound(
        plan.runs.begin(), plan.runs.end(), slot,
        [](const GapRun &run, std::size_t candidate) {
            return run.end <= candidate;
        });
    return it != plan.runs.end() && slot >= it->begin && slot < it->end
               ? &*it
               : nullptr;
}

inline AlignedValue detector_slot_value_at(
    const std::vector<DetectorSlotCell> &cells, const GapPlan &plan,
    std::size_t slot, DetailLevel detail = DetailLevel::compact) {
    if (slot >= cells.size()) {
        throw std::out_of_range("detector slot index is out of range");
    }
    const auto &cell = cells[slot];
    if (cell.has_native_row) {
        CellQuality quality{Origin::original,
                            cell.native_valid && std::isfinite(cell.value)
                                ? Validity::valid
                                : Validity::invalid,
                            Method::exact,
                            cell.native_valid && std::isfinite(cell.value)
                                ? Reason::none
                                : Reason::nonfinite_source,
                            {}};
        add_source_if_requested(quality, detail, cell.native_row, 1.0);
        return {cell.value, std::move(quality)};
    }

    const GapRun *run = find_gap_run(plan, slot);
    if (run == nullptr) {
        throw std::logic_error("missing detector slot is absent from gap plan");
    }
    if (run->action != GapAction::linear_fill) {
        return {std::numeric_limits<double>::quiet_NaN(),
                unavailable_quality(run->reason)};
    }
    const std::size_t left = *run->left_endpoint_slot;
    const std::size_t right = *run->right_endpoint_slot;
    const double lambda = static_cast<double>(slot - left) /
                          static_cast<double>(right - left);
    CellQuality quality{Origin::synthesized, Validity::valid, Method::linear,
                        Reason::none, {}};
    add_source_if_requested(quality, detail, cells[left].native_row,
                            1.0 - lambda);
    add_source_if_requested(quality, detail, cells[right].native_row, lambda);
    return {(1.0 - lambda) * cells[left].value +
                lambda * cells[right].value,
            std::move(quality)};
}

struct TemporalSupport {
    double begin_seconds = 0.0;
    double end_seconds = 0.0;
};

inline double support_duration(const TemporalSupport &support) {
    if (!std::isfinite(support.begin_seconds) ||
        !std::isfinite(support.end_seconds) ||
        support.end_seconds < support.begin_seconds) {
        throw std::invalid_argument("temporal support is invalid");
    }
    return support.end_seconds - support.begin_seconds;
}

inline double support_intersection_duration(const TemporalSupport &lhs,
                                            const TemporalSupport &rhs) {
    (void)support_duration(lhs);
    (void)support_duration(rhs);
    return std::max(0.0, std::min(lhs.end_seconds, rhs.end_seconds) -
                             std::max(lhs.begin_seconds, rhs.begin_seconds));
}

inline double acquired_detector_exposure(
    const TemporalSupport &nominal_cell,
    const std::optional<TemporalSupport> &native_integration,
    const CellQuality &quality) {
    if (quality.origin != Origin::original ||
        quality.validity != Validity::valid ||
        !native_integration.has_value()) {
        return 0.0;
    }
    return support_intersection_duration(nominal_cell, *native_integration);
}

struct ExposureCell {
    TemporalSupport nominal_cell;
    std::optional<TemporalSupport> native_integration;
    CellQuality quality;
};

struct ExposureSummary {
    double nominal_span_seconds = 0.0;
    double acquired_exposure_seconds = 0.0;
    std::size_t original_valid_count = 0;
    std::size_t synthesized_count = 0;
    std::size_t unavailable_count = 0;
};

inline ExposureSummary summarize_exposure(
    const std::vector<ExposureCell> &cells, std::size_t begin,
    std::size_t end) {
    if (begin > end || end > cells.size()) {
        throw std::out_of_range("exposure slice is outside the cell vector");
    }
    ExposureSummary result;
    for (std::size_t i = begin; i < end; ++i) {
        result.nominal_span_seconds += support_duration(cells[i].nominal_cell);
        result.acquired_exposure_seconds += acquired_detector_exposure(
            cells[i].nominal_cell, cells[i].native_integration,
            cells[i].quality);
        if (cells[i].quality.origin == Origin::original &&
            cells[i].quality.validity == Validity::valid) {
            ++result.original_valid_count;
        } else if (cells[i].quality.origin == Origin::synthesized) {
            ++result.synthesized_count;
        } else if (cells[i].quality.origin == Origin::unavailable) {
            ++result.unavailable_count;
        }
    }
    return result;
}

inline void require_small_diagnostic_matrix(const Eigen::MatrixXd &matrix,
                                            Eigen::Index maximum_dimension) {
    if (maximum_dimension <= 0 || matrix.rows() > maximum_dimension ||
        matrix.cols() > maximum_dimension) {
        throw std::invalid_argument(
            "expanded response/covariance request exceeds the bounded diagnostic size");
    }
    if (!matrix.allFinite()) {
        throw std::invalid_argument(
            "response/covariance input contains non-finite values");
    }
}

inline Eigen::MatrixXd conditional_covariance_small(
    const Eigen::Ref<const Eigen::MatrixXd> &mapping,
    const Eigen::Ref<const Eigen::MatrixXd> &native_covariance,
    Eigen::Index maximum_dimension = 64) {
    require_small_diagnostic_matrix(mapping, maximum_dimension);
    require_small_diagnostic_matrix(native_covariance, maximum_dimension);
    if (native_covariance.rows() != native_covariance.cols() ||
        mapping.cols() != native_covariance.rows()) {
        throw std::invalid_argument(
            "mapping and native covariance dimensions are incompatible");
    }
    return mapping * native_covariance * mapping.transpose();
}

inline Eigen::VectorXd conditional_response_small(
    const Eigen::Ref<const Eigen::MatrixXd> &mapping,
    const Eigen::Ref<const Eigen::VectorXd> &native_template,
    Eigen::Index maximum_dimension = 64) {
    require_small_diagnostic_matrix(mapping, maximum_dimension);
    if (native_template.size() > maximum_dimension ||
        mapping.cols() != native_template.size() ||
        !native_template.allFinite()) {
        throw std::invalid_argument(
            "mapping and native response template are incompatible");
    }
    return mapping * native_template;
}

inline Eigen::MatrixXd timing_covariance_small(
    const Eigen::Ref<const Eigen::MatrixXd> &timing_jacobian,
    const Eigen::Ref<const Eigen::MatrixXd> &timing_parameter_covariance,
    Eigen::Index maximum_dimension = 64) {
    return conditional_covariance_small(timing_jacobian,
                                        timing_parameter_covariance,
                                        maximum_dimension);
}

inline double linear_interpolation_variance(double lambda,
                                            double variance_left,
                                            double variance_right,
                                            double covariance) {
    if (!std::isfinite(lambda) || lambda < 0.0 || lambda > 1.0 ||
        !std::isfinite(variance_left) || !std::isfinite(variance_right) ||
        !std::isfinite(covariance)) {
        throw std::invalid_argument(
            "linear variance inputs are outside their finite domain");
    }
    return (1.0 - lambda) * (1.0 - lambda) * variance_left +
           lambda * lambda * variance_right +
           2.0 * lambda * (1.0 - lambda) * covariance;
}

inline std::complex<double> fractional_linear_response(
    double alpha, double angular_frequency_radians_per_second,
    double cadence_seconds) {
    if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0 ||
        !std::isfinite(angular_frequency_radians_per_second)) {
        throw std::invalid_argument(
            "fractional response inputs are outside their finite domain");
    }
    require_finite_positive(cadence_seconds, "detector cadence");
    const std::complex<double> imaginary{0.0, 1.0};
    const double phase = angular_frequency_radians_per_second *
                         cadence_seconds;
    return std::exp(-imaginary * phase * alpha) *
           ((1.0 - alpha) + alpha * std::exp(imaginary * phase));
}

inline double linear_interpolation_error_bound(double bracket_span_seconds,
                                               double curvature_supremum) {
    require_finite_positive(bracket_span_seconds, "interpolation bracket");
    if (!std::isfinite(curvature_supremum) || curvature_supremum < 0.0) {
        throw std::invalid_argument(
            "curvature supremum must be finite and nonnegative");
    }
    return bracket_span_seconds * bracket_span_seconds *
           curvature_supremum / 8.0;
}

enum class StreamRole { telescope, hwpr };

struct MissingStreamDisposition {
    StreamRole role = StreamRole::telescope;
    bool present = false;
    bool detector_lattice_preserved = true;
    bool intensity_eligible = false;
    bool polarization_eligible = false;
    Reason reason = Reason::missing_stream;
};

inline MissingStreamDisposition missing_stream_disposition(StreamRole role) {
    if (role == StreamRole::hwpr) {
        return {role, false, true, true, false, Reason::missing_stream};
    }
    return {role, false, true, false, false, Reason::missing_stream};
}

}  // namespace citlali::pipeline::sci_align
