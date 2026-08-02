#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline::sci_align {

// ALIGN stores authoritative windows as zero-based, half-open detector-slot
// intervals.  The existing inclusive Eigen matrix is a one-way compatibility
// surface for numerical processors; it is not the identity authority.
struct HalfOpenInterval {
    Eigen::Index start = 0;
    Eigen::Index stop = 0;

    Eigen::Index size() const {
        if (!valid()) {
            throw std::logic_error("invalid half-open interval has no size");
        }
        return stop - start;
    }
    bool empty() const noexcept { return start == stop; }
    bool valid() const noexcept { return start >= 0 && stop >= start; }
    bool contains(Eigen::Index index) const noexcept {
        return start <= index && index < stop;
    }

    friend bool operator==(HalfOpenInterval left,
                           HalfOpenInterval right) noexcept {
        return left.start == right.start && left.stop == right.stop;
    }
};

enum class ScanStatus {
    usable,
    short_support,
    partial_support,
    empty,
    rejected,
    unusable,
};

inline const char *to_string(ScanStatus status) noexcept {
    switch (status) {
        case ScanStatus::usable:
            return "usable";
        case ScanStatus::short_support:
            return "short";
        case ScanStatus::partial_support:
            return "partial";
        case ScanStatus::empty:
            return "empty";
        case ScanStatus::rejected:
            return "rejected";
        case ScanStatus::unusable:
            return "unusable";
    }
    return "unusable";
}

struct ScanWindowRecord {
    Eigen::Index stable_id = 0;
    std::optional<Eigen::Index> physical_id;
    std::string identity_authority;
    HalfOpenInterval processing;
    HalfOpenInterval science;
    HalfOpenInterval context;
    ScanStatus status = ScanStatus::usable;
    bool legacy_processing_admitted = true;
    Eigen::Index compatibility_ordinal = -1;
    // Optional existing-use adapter windows. Candidate OD5 identity remains
    // in processing/science/context even when the 9aae numerical-consumer
    // cohort used a different boundary or discarded remainder.
    std::optional<HalfOpenInterval> compatibility_science;
    std::optional<HalfOpenInterval> compatibility_context;
};

inline const HalfOpenInterval &compatibility_science_window(
    const ScanWindowRecord &record) {
    if (!record.legacy_processing_admitted) {
        throw std::logic_error(
            "nonadmitted scan has no compatibility science window");
    }
    return record.compatibility_science.has_value()
               ? *record.compatibility_science
               : record.science;
}

inline const HalfOpenInterval &compatibility_context_window(
    const ScanWindowRecord &record) {
    if (!record.legacy_processing_admitted) {
        throw std::logic_error(
            "nonadmitted scan has no compatibility context window");
    }
    return record.compatibility_context.has_value()
               ? *record.compatibility_context
               : record.context;
}

struct PhysicalWindowRecord {
    Eigen::Index stable_id = 0;
    HalfOpenInterval interval;
    std::string authority;
};

struct ScanWindowPlan {
    std::string policy;
    double requested_value = 0.0;
    double effective_duration_sec = 0.0;
    Eigen::Index observation_sample_count = 0;
    std::vector<PhysicalWindowRecord> physical_records;
    std::vector<ScanWindowRecord> records;
    std::vector<Eigen::Index> compatibility_to_stable_id;

    void clear() {
        policy.clear();
        requested_value = 0.0;
        effective_duration_sec = 0.0;
        observation_sample_count = 0;
        physical_records.clear();
        records.clear();
        compatibility_to_stable_id.clear();
    }
};

inline Eigen::Index round_half_up_positive(double value,
                                           const char *label) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(std::string{"invalid "} + label);
    }
    const double rounded_value = std::floor(value + 0.5);
    const double exclusive_upper =
        -static_cast<double>(std::numeric_limits<Eigen::Index>::min());
    if (!std::isfinite(rounded_value) ||
        rounded_value >= exclusive_upper) {
        throw std::runtime_error(std::string{"invalid "} + label);
    }
    const auto rounded = static_cast<Eigen::Index>(rounded_value);
    if (rounded <= 0) {
        throw std::runtime_error(std::string{"nonpositive effective "} + label);
    }
    return rounded;
}

inline HalfOpenInterval clipped_context(HalfOpenInterval science,
                                        Eigen::Index context_samples,
                                        Eigen::Index sample_count) {
    if (!science.valid() || science.stop > sample_count || sample_count < 0) {
        throw std::runtime_error("invalid science interval for context");
    }
    const Eigen::Index context = std::max<Eigen::Index>(0, context_samples);
    const Eigen::Index context_start =
        context >= science.start ? 0 : science.start - context;
    const Eigen::Index context_stop =
        context >= sample_count - science.stop
            ? sample_count
            : science.stop + context;
    return {
        context_start,
        context_stop,
    };
}

inline HalfOpenInterval clipped_context(HalfOpenInterval science,
                                        Eigen::Index context_samples,
                                        HalfOpenInterval support,
                                        Eigen::Index sample_count) {
    if (!support.valid() || support.empty() ||
        support.stop > sample_count || !science.valid() || science.empty() ||
        science.start < support.start || science.stop > support.stop) {
        throw std::runtime_error(
            "invalid governing support interval for scan context");
    }
    const Eigen::Index context = std::max<Eigen::Index>(0, context_samples);
    const Eigen::Index leading_support =
        science.start - support.start;
    const Eigen::Index trailing_support =
        support.stop - science.stop;
    return {
        context >= leading_support
            ? support.start
            : science.start - context,
        context >= trailing_support
            ? support.stop
            : science.stop + context,
    };
}

inline Eigen::Index checked_scan_container_size(std::size_t size,
                                                const char *label) {
    if (size > static_cast<std::size_t>(
                   std::numeric_limits<Eigen::Index>::max())) {
        throw std::overflow_error(std::string{label} +
                                  " exceeds Eigen index range");
    }
    return static_cast<Eigen::Index>(size);
}

inline void validate_scan_window_plan(const ScanWindowPlan &plan) {
    (void)checked_scan_container_size(plan.records.size(),
                                      "scan record count");
    const auto physical_record_count = checked_scan_container_size(
        plan.physical_records.size(), "physical scan record count");
    const auto compatibility_record_count = checked_scan_container_size(
        plan.compatibility_to_stable_id.size(),
        "compatibility scan record count");
    Eigen::Index previous_physical_stop = 0;
    Eigen::Index expected_physical_id = 0;
    bool first_physical = true;
    for (const auto &physical : plan.physical_records) {
        if (physical.stable_id != expected_physical_id++ ||
            !physical.interval.valid() || physical.interval.empty() ||
            physical.interval.stop > plan.observation_sample_count ||
            physical.authority.empty() ||
            (!first_physical &&
             physical.interval.start < previous_physical_stop)) {
            throw std::runtime_error(
                "invalid or overlapping physical-window identity");
        }
        first_physical = false;
        previous_physical_stop = physical.interval.stop;
    }

    HalfOpenInterval processing_hull;
    HalfOpenInterval context_hull;
    if (!plan.records.empty()) {
        processing_hull.start = plan.records.front().processing.start;
        processing_hull.stop = plan.records.front().processing.stop;
        context_hull.start = plan.records.front().context.start;
        context_hull.stop = plan.records.front().context.stop;
        for (const auto &record : plan.records) {
            processing_hull.start = std::min(
                processing_hull.start, record.processing.start);
            processing_hull.stop = std::max(
                processing_hull.stop, record.processing.stop);
            context_hull.start = std::min(
                context_hull.start, record.context.start);
            context_hull.stop = std::max(
                context_hull.stop, record.context.stop);
        }
    }

    Eigen::Index previous_stop = 0;
    bool first = true;
    Eigen::Index expected_id = 0;
    Eigen::Index expected_ordinal = 0;
    Eigen::Index previous_compatibility_stop = 0;
    bool first_compatibility = true;
    for (const auto &record : plan.records) {
        if (record.stable_id != expected_id++) {
            throw std::runtime_error("scan stable identities are not contiguous from zero");
        }
        if (record.identity_authority.empty() ||
            !record.processing.valid() || !record.science.valid() ||
            !record.context.valid() ||
            record.processing.empty() ||
            record.processing.stop > plan.observation_sample_count ||
            record.science.start < record.processing.start ||
            record.science.stop > record.processing.stop ||
            record.context.start > record.science.start ||
            record.context.stop < record.science.stop ||
            record.context.stop > plan.observation_sample_count) {
            throw std::runtime_error("invalid scan window support");
        }
        if (!first && record.processing.start < previous_stop) {
            throw std::runtime_error("overlapping processing chunk windows");
        }
        first = false;
        previous_stop = record.processing.stop;
        if (record.physical_id.has_value()) {
            if (*record.physical_id < 0 ||
                *record.physical_id >= physical_record_count) {
                throw std::runtime_error(
                    "processing chunk references an unknown physical window");
            }
            const auto &physical = plan.physical_records.at(
                static_cast<std::size_t>(*record.physical_id));
            if (record.processing.start < physical.interval.start ||
                record.processing.stop > physical.interval.stop) {
                throw std::runtime_error(
                    "processing chunk is outside its physical window");
            }
        }
        else {
            const bool bounded_raster_adapter =
                plan.policy ==
                    "legacy_4x_linear_any_nonzero_plus_outside_v1" &&
                record.identity_authority ==
                    "legacy_inferred_raster_compatibility_segment_not_physical";
            const bool requested_continuous_chunk =
                (plan.policy == "fixed_duration_round_half_up_v1" ||
                 plan.policy == "fixed_count_balanced_v1") &&
                record.identity_authority ==
                    "requested_processing_chunk_under_continuous_observation_no_physical_scan_authority";
            if (!bounded_raster_adapter && !requested_continuous_chunk) {
                throw std::runtime_error(
                    "missing physical-window authority outside a bounded unavailable-authority policy");
            }
        }
        if (record.legacy_processing_admitted) {
            if (record.compatibility_ordinal != expected_ordinal ||
                expected_ordinal >= compatibility_record_count ||
                plan.compatibility_to_stable_id[expected_ordinal] !=
                    record.stable_id) {
                throw std::runtime_error("invalid compatibility scan mapping");
            }
            const auto &compatibility_science =
                compatibility_science_window(record);
            const auto &compatibility_context =
                compatibility_context_window(record);
            if (!compatibility_science.valid() ||
                compatibility_science.empty() ||
                compatibility_science.stop >
                    plan.observation_sample_count ||
                compatibility_science.start < processing_hull.start ||
                compatibility_science.stop > processing_hull.stop ||
                !compatibility_context.valid() ||
                compatibility_context.empty() ||
                compatibility_context.start >
                    compatibility_science.start ||
                compatibility_context.stop <
                    compatibility_science.stop ||
                compatibility_context.stop >
                    plan.observation_sample_count ||
                compatibility_context.start < context_hull.start ||
                compatibility_context.stop > context_hull.stop ||
                (!first_compatibility &&
                 compatibility_science.start <
                     previous_compatibility_stop)) {
                throw std::runtime_error(
                    "invalid or overlapping compatibility scan support");
            }
            first_compatibility = false;
            previous_compatibility_stop = compatibility_science.stop;
            ++expected_ordinal;
        }
        else if (record.compatibility_ordinal != -1 ||
                 record.compatibility_science.has_value() ||
                 record.compatibility_context.has_value()) {
            throw std::runtime_error(
                "non-admitted scan retains a compatibility adapter");
        }
    }
    if (expected_ordinal != compatibility_record_count) {
        throw std::runtime_error("incomplete compatibility scan mapping");
    }
}

inline void append_scan_record(ScanWindowPlan &plan,
                               HalfOpenInterval interval,
                               Eigen::Index context_samples,
                               ScanStatus status,
                               bool legacy_processing_admitted,
                               std::optional<Eigen::Index> physical_id,
                               std::string identity_authority,
                               std::optional<HalfOpenInterval>
                                   context_support = std::nullopt) {
    ScanWindowRecord record;
    record.stable_id = checked_scan_container_size(
        plan.records.size(), "scan record count");
    record.physical_id = physical_id;
    record.identity_authority = std::move(identity_authority);
    record.processing = interval;
    record.science = interval;
    record.context = context_support.has_value()
        ? clipped_context(interval, context_samples, *context_support,
                          plan.observation_sample_count)
        : clipped_context(interval, context_samples,
                          plan.observation_sample_count);
    record.status = interval.empty() ? ScanStatus::empty : status;
    record.legacy_processing_admitted =
        legacy_processing_admitted && !interval.empty();
    if (record.legacy_processing_admitted) {
        record.compatibility_ordinal = checked_scan_container_size(
            plan.compatibility_to_stable_id.size(),
            "compatibility scan record count");
        plan.compatibility_to_stable_id.push_back(record.stable_id);
        record.compatibility_science = record.science;
        record.compatibility_context = record.context;
    }
    plan.records.push_back(record);
}

inline ScanWindowPlan make_fixed_duration_scan_plan(
    Eigen::Index sample_count, HalfOpenInterval support,
    double requested_duration_sec, double cadence_sec,
    Eigen::Index context_samples = 0) {
    if (sample_count <= 0 || !support.valid() || support.empty() ||
        support.stop > sample_count || !std::isfinite(cadence_sec) ||
        cadence_sec <= 0.0) {
        throw std::runtime_error("invalid fixed-duration scan inputs");
    }
    const Eigen::Index samples_per_chunk = round_half_up_positive(
        requested_duration_sec / cadence_sec, "fixed scan duration");

    ScanWindowPlan plan;
    plan.policy = "fixed_duration_round_half_up_v1";
    plan.requested_value = requested_duration_sec;
    plan.effective_duration_sec =
        static_cast<double>(samples_per_chunk) * cadence_sec;
    if (!std::isfinite(plan.effective_duration_sec)) {
        throw std::overflow_error(
            "effective fixed-duration scan support is nonfinite");
    }
    plan.observation_sample_count = sample_count;
    for (Eigen::Index start = support.start; start < support.stop;) {
        const Eigen::Index remaining = support.stop - start;
        const Eigen::Index step =
            std::min<Eigen::Index>(remaining, samples_per_chunk);
        const Eigen::Index stop = start + step;
        const bool has_full_requested_support =
            stop - start == samples_per_chunk;
        append_scan_record(
            plan, {start, stop}, context_samples,
            has_full_requested_support ? ScanStatus::usable
                                       : ScanStatus::partial_support,
            // OD5 retains the final partial as a compact identity, while the
            // governing existing-use adapter remains restricted to the full
            // chunks admitted by 9aae. A downstream consumer may separately
            // admit this partial under its own declared support contract.
            has_full_requested_support, std::nullopt,
            "requested_processing_chunk_under_continuous_observation_no_physical_scan_authority",
            support);
        start = stop;
    }
    validate_scan_window_plan(plan);
    return plan;
}

inline ScanWindowPlan make_number_scan_plan(
    Eigen::Index sample_count, HalfOpenInterval support,
    Eigen::Index requested_count, double cadence_sec,
    Eigen::Index context_samples = 0) {
    if (sample_count <= 0 || !support.valid() || support.empty() ||
        support.stop > sample_count || requested_count <= 0 ||
        requested_count > support.size() ||
        !std::isfinite(cadence_sec) || cadence_sec <= 0.0) {
        throw std::runtime_error("invalid number-based scan inputs");
    }

    ScanWindowPlan plan;
    plan.policy = "fixed_count_balanced_v1";
    plan.requested_value = static_cast<double>(requested_count);
    plan.observation_sample_count = sample_count;
    const Eigen::Index base = support.size() / requested_count;
    const Eigen::Index remainder = support.size() % requested_count;
    Eigen::Index start = support.start;
    for (Eigen::Index q = 0; q < requested_count; ++q) {
        const Eigen::Index size = base + (q < remainder ? 1 : 0);
        if (size <= 0 || size > support.stop - start) {
            throw std::overflow_error(
                "balanced scan chunk exceeds governing support");
        }
        const Eigen::Index stop = start + size;
        append_scan_record(plan, {start, stop}, context_samples,
                           ScanStatus::usable, true, std::nullopt,
                           "requested_processing_chunk_under_continuous_observation_no_physical_scan_authority",
                           support);
        auto &record = plan.records.back();
        const Eigen::Index legacy_start =
            support.start + q * base;
        const HalfOpenInterval legacy_science{
            legacy_start, legacy_start + base};
        record.compatibility_science = legacy_science;
        record.compatibility_context = clipped_context(
            legacy_science, context_samples, support, sample_count);
        start = stop;
    }
    plan.effective_duration_sec =
        static_cast<double>(base) * cadence_sec;
    if (!std::isfinite(plan.effective_duration_sec)) {
        throw std::overflow_error(
            "effective fixed-count scan duration is nonfinite");
    }
    validate_scan_window_plan(plan);
    return plan;
}

inline Eigen::Index checked_number_scan_count(
    double requested_count, Eigen::Index governing_support_size) {
    if (!std::isfinite(requested_count) || requested_count <= 0.0 ||
        std::floor(requested_count) != requested_count ||
        governing_support_size <= 0) {
        throw std::runtime_error("invalid number-based scan count");
    }

    // The conversion below is defined only inside the signed Eigen::Index
    // range.  Compare against its exact exclusive power-of-two bound before
    // narrowing; converting max() to double can round upward on 64-bit builds.
    const double index_exclusive_upper = std::ldexp(
        1.0, std::numeric_limits<Eigen::Index>::digits);
    if (requested_count >= index_exclusive_upper ||
        requested_count > static_cast<double>(governing_support_size)) {
        throw std::overflow_error(
            "number-based scan count exceeds governing support");
    }

    const auto narrowed = static_cast<Eigen::Index>(requested_count);
    if (narrowed <= 0 || narrowed > governing_support_size ||
        static_cast<double>(narrowed) != requested_count) {
        throw std::overflow_error(
            "number-based scan count is not exactly representable");
    }
    return narrowed;
}

inline ScanWindowPlan make_fixed_duration_scan_plan(
    Eigen::Index sample_count, double requested_duration_sec,
    double cadence_sec, Eigen::Index context_samples = 0) {
    return make_fixed_duration_scan_plan(
        sample_count, {0, sample_count}, requested_duration_sec,
        cadence_sec, context_samples);
}

inline ScanWindowPlan make_number_scan_plan(
    Eigen::Index sample_count, Eigen::Index requested_count,
    double cadence_sec, Eigen::Index context_samples = 0) {
    return make_number_scan_plan(
        sample_count, {0, sample_count}, requested_count,
        cadence_sec, context_samples);
}

using TelescopeHoldWord = std::uint64_t;

enum class TelescopeHoldReason : TelescopeHoldWord {
    pointing = 0x02,
    external = 0x04,
    obs_pgm = 0x08,
    m1 = 0x10,
    m2 = 0x20,
    m3 = 0x40,
};

struct TelescopeHoldReasonDefinition {
    TelescopeHoldReason reason;
    const char *producer_name;
    bool declared_never_implemented;
};

inline constexpr std::array<TelescopeHoldReasonDefinition, 6>
    telescope_hold_reason_definitions{{
        {TelescopeHoldReason::pointing, "Pointing", false},
        {TelescopeHoldReason::external, "External", true},
        {TelescopeHoldReason::obs_pgm, "ObsPgm", false},
        {TelescopeHoldReason::m1, "M1", false},
        {TelescopeHoldReason::m2, "M2", false},
        {TelescopeHoldReason::m3, "M3", false},
    }};

inline constexpr TelescopeHoldWord telescope_hold_defined_mask = 0x7e;

inline constexpr TelescopeHoldWord telescope_hold_reason_mask(
    TelescopeHoldReason reason) noexcept {
    return static_cast<TelescopeHoldWord>(reason);
}

inline constexpr bool native_hold_word_has_reason(
    TelescopeHoldWord word, TelescopeHoldReason reason) noexcept {
    return (word & telescope_hold_reason_mask(reason)) != 0;
}

inline constexpr TelescopeHoldWord native_hold_word_unknown_bits(
    TelescopeHoldWord word) noexcept {
    return word & ~telescope_hold_defined_mask;
}

// Producer authority defines zero as the only science-valid native word.
// This intentionally fails closed for both defined and unknown set bits. It
// records native-state meaning; it does not introduce a new all-profile mask.
inline constexpr bool native_hold_word_science_valid(
    TelescopeHoldWord word) noexcept {
    return word == 0;
}

inline constexpr const char *telescope_hold_transition_side_authority =
    "unresolved";

// This is the only admitted non-native Hold predicate.  It reproduces the
// released 4.x order of operations: linearly align the complete numeric word,
// then test that aligned value for nonzero.  Native bit meanings and native
// zero-only validity are producer-authoritative, but this compatibility
// alignment still assigns no left/right transition-event timing.
inline bool legacy_hold_linear_any_nonzero_state(double aligned_numeric_word) {
    if (!std::isfinite(aligned_numeric_word) || aligned_numeric_word < 0.0) {
        throw std::runtime_error(
            "invalid aligned numeric input to the legacy Hold compatibility view");
    }
    return aligned_numeric_word != 0.0;
}

// Routine TOD output keeps the historical variable name `Hold` solely as a
// compatibility alias.  Emitting the fractional linearly aligned word would
// falsely look like a raw state word, so the persisted view is explicitly the
// post-nonzero 0/1 result. Exact native words remain internal; no raw-word
// exporter is implemented by this bounded repair.
inline Eigen::VectorXd legacy_hold_emitted_compatibility_view(
    const Eigen::VectorXd &legacy_aligned_numeric_word) {
    Eigen::VectorXd result(legacy_aligned_numeric_word.size());
    for (Eigen::Index i = 0; i < legacy_aligned_numeric_word.size(); ++i) {
        const bool state = legacy_hold_linear_any_nonzero_state(
            legacy_aligned_numeric_word(i));
        result(i) = state ? 1.0 : 0.0;
    }
    return result;
}

inline std::vector<unsigned char> compose_legacy_hold_and_outside(
    const Eigen::VectorXd &legacy_aligned_numeric_word,
    const std::vector<unsigned char> &outside_map_box) {
    if (legacy_aligned_numeric_word.size() !=
        static_cast<Eigen::Index>(outside_map_box.size())) {
        throw std::runtime_error("Hold and outside-map support sizes differ");
    }
    std::vector<unsigned char> result(outside_map_box.size(), 0);
    for (Eigen::Index i = 0; i < legacy_aligned_numeric_word.size(); ++i) {
        result[static_cast<std::size_t>(i)] =
            static_cast<unsigned char>(
                legacy_hold_linear_any_nonzero_state(
                    legacy_aligned_numeric_word(i)) ||
                outside_map_box[static_cast<std::size_t>(i)] != 0);
    }
    return result;
}

inline ScanWindowPlan make_raster_compatibility_scan_plan(
    const std::vector<unsigned char> &composite_excluded,
    HalfOpenInterval support, double cadence_hz,
    Eigen::Index context_samples = 0,
    double legacy_minimum_duration_sec = 2.0,
    Eigen::Index legacy_inner_edge_trim_samples = 0) {
    const auto observation_sample_count = checked_scan_container_size(
        composite_excluded.size(), "raster observation sample count");
    if (composite_excluded.empty() || !std::isfinite(cadence_hz) ||
        cadence_hz <= 0.0 || !support.valid() || support.empty() ||
        support.stop > observation_sample_count ||
        !std::isfinite(legacy_minimum_duration_sec) ||
        legacy_minimum_duration_sec < 0.0 ||
        legacy_inner_edge_trim_samples < 0) {
        throw std::runtime_error("invalid raster scan inputs");
    }
    const double legacy_minimum_samples_value = std::ceil(
        legacy_minimum_duration_sec * cadence_hz);
    const double exclusive_index_upper =
        -static_cast<double>(std::numeric_limits<Eigen::Index>::min());
    if (!std::isfinite(legacy_minimum_samples_value) ||
        legacy_minimum_samples_value < 0.0 ||
        legacy_minimum_samples_value >= exclusive_index_upper) {
        throw std::overflow_error(
            "legacy raster minimum support exceeds Eigen index range");
    }
    const Eigen::Index legacy_minimum_samples =
        static_cast<Eigen::Index>(legacy_minimum_samples_value);

    ScanWindowPlan plan;
    plan.policy = "legacy_4x_linear_any_nonzero_plus_outside_v1";
    plan.requested_value = legacy_minimum_duration_sec;
    plan.observation_sample_count = observation_sample_count;

    Eigen::Index index = support.start;
    while (index < support.stop) {
        while (index < support.stop &&
               composite_excluded[static_cast<std::size_t>(index)] != 0) {
            ++index;
        }
        if (index == support.stop) {
            break;
        }
        const Eigen::Index start = index;
        while (index < support.stop &&
               composite_excluded[static_cast<std::size_t>(index)] == 0) {
            ++index;
        }
        const HalfOpenInterval interval{start, index};
        const Eigen::Index omitted_first =
            start == support.start ? start : start + 1;
        const HalfOpenInterval legacy_science{omitted_first, index};
        const bool admitted =
            !legacy_science.empty() &&
            legacy_science.size() >= legacy_minimum_samples;
        append_scan_record(plan, interval, context_samples,
                           admitted ? ScanStatus::usable
                                    : ScanStatus::short_support,
                           admitted, std::nullopt,
                           "legacy_inferred_raster_compatibility_segment_not_physical",
                           support);
        if (admitted) {
            auto &record = plan.records.back();
            record.compatibility_science = legacy_science;
            record.compatibility_context = clipped_context(
                legacy_science, context_samples, support,
                plan.observation_sample_count);
        }
    }
    if (plan.records.empty()) {
        throw std::runtime_error("raster scan policy found no usable support identity");
    }
    if (!plan.compatibility_to_stable_id.empty() &&
        legacy_inner_edge_trim_samples > 0) {
        auto &first = plan.records.at(static_cast<std::size_t>(
            plan.compatibility_to_stable_id.front()));
        auto &last = plan.records.at(static_cast<std::size_t>(
            plan.compatibility_to_stable_id.back()));
        auto first_science = *first.compatibility_science;
        const Eigen::Index first_removable =
            first_science.stop - first_science.start - 1;
        const Eigen::Index first_trim = std::min(
            legacy_inner_edge_trim_samples, first_removable);
        first_science.start += first_trim;
        first.compatibility_science = first_science;

        auto last_science = *last.compatibility_science;
        const Eigen::Index last_removable =
            last_science.stop - last_science.start - 1;
        const Eigen::Index last_trim = std::min(
            legacy_inner_edge_trim_samples, last_removable);
        last_science.stop -= last_trim;
        last.compatibility_science = last_science;
    }
    validate_scan_window_plan(plan);
    return plan;
}

inline ScanWindowPlan make_raster_compatibility_scan_plan(
    const std::vector<unsigned char> &composite_excluded,
    double cadence_hz, Eigen::Index context_samples = 0,
    double legacy_minimum_duration_sec = 2.0,
    Eigen::Index legacy_inner_edge_trim_samples = 0) {
    return make_raster_compatibility_scan_plan(
        composite_excluded,
        {0, checked_scan_container_size(
                composite_excluded.size(),
                "raster observation sample count")},
        cadence_hz, context_samples, legacy_minimum_duration_sec,
        legacy_inner_edge_trim_samples);
}

using ScanIndexMatrix =
    Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>;

inline ScanIndexMatrix compatibility_scan_indices(
    const ScanWindowPlan &plan) {
    validate_scan_window_plan(plan);
    ScanIndexMatrix result(
        4, checked_scan_container_size(
               plan.compatibility_to_stable_id.size(),
               "compatibility scan index count"));
    for (Eigen::Index ordinal = 0; ordinal < result.cols(); ++ordinal) {
        const auto stable_id = plan.compatibility_to_stable_id[
            static_cast<std::size_t>(ordinal)];
        const auto &record = plan.records[static_cast<std::size_t>(stable_id)];
        const auto &science = compatibility_science_window(record);
        const auto &context = compatibility_context_window(record);
        if (science.empty() || context.empty()) {
            throw std::runtime_error("empty scan cannot be adapted for processing");
        }
        result(0, ordinal) = science.start;
        result(1, ordinal) = science.stop - 1;
        result(2, ordinal) = context.start;
        result(3, ordinal) = context.stop - 1;
    }
    return result;
}

}  // namespace citlali::pipeline::sci_align
