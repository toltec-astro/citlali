#pragma once

#include <citlali/core/pipeline/timestream_native_alignment.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace citlali::pipeline {

enum class CalWvrCause : unsigned char {
    available, absent, unbracketed, gap_outside_source_validity,
    conflicting_duplicate, negative, nonfinite, time_mapping_unavailable
};
std::string_view cal_wvr_cause_name(CalWvrCause);
struct CalWvrRecord {
    std::string identity;
    double time_unix_sec, tau225;
    bool producer_valid;
    double valid_first_unix_sec, valid_last_unix_sec;
};
struct CalWvrSample {
    CalWvrCause cause = CalWvrCause::absent;
    std::optional<double> tau225;
    double mapped_time_unix_sec = 0, weight = 0;
    std::size_t first_record = 0, last_record = 0;
    bool exact_match = false;
};
enum class CalOpacityQuality {
    invalid_opacity_input, opacity_quality_unavailable, outside_supported_opacity,
    science_qualification_eligible, engineering_only
};
std::string_view cal_opacity_quality_name(CalOpacityQuality);
struct CalWvrQuality {
    struct Excursion { double first, last, duration, peak; };
    CalOpacityQuality classification = CalOpacityQuality::opacity_quality_unavailable;
    std::string cause;
    double first = 0, last = 0, duration = 0, area = 0, mean = 0, minimum = 0, maximum = 0;
    double excursion_duration = 0, longest_excursion = 0, excursion_fraction = 0, integrated_excess = 0;
    std::size_t breakpoint_count = 0;
    bool summary_available = false;
    std::vector<Excursion> excursions;
};

// Source records and validity are immutable; bracket indices reference this
// one observation-owned inventory. Times are Unix seconds in the admitted
// ALIGN reference basis, tau225 is dimensionless. No cadence-based gap rule.
class CalWvrEvidence {
public:
    static std::shared_ptr<const CalWvrEvidence> learn(
        NativeObservationScope, std::string source_identity,
        std::string align_time_mapping_identity, std::vector<CalWvrRecord>);
    CalWvrSample at(double mapped_time_unix_sec) const;
    CalWvrQuality quality(double first_detector_time, double last_detector_time) const;
    const auto &scope() const noexcept { return scope_; }
    const auto &source_identity() const noexcept { return source_; }
    const auto &time_mapping_identity() const noexcept { return mapping_; }
    const auto &records() const noexcept { return records_; }
    static constexpr std::string_view method = "cal_wvr_tau225_linear_detector_time_v1";
    static constexpr std::string_view quality_method = "cal_wvr_observation_quality_mean_peak_v1";
private:
    explicit CalWvrEvidence(NativeObservationScope scope):scope_{scope} {}
    NativeObservationScope scope_;
    std::string source_, mapping_;
    std::vector<CalWvrRecord> records_;
    struct Group { std::size_t first, last; bool conflict; };
    std::vector<Group> groups_;
    CalWvrCause support_cause(const Group &,const Group &,double first,double last) const;
};

} // namespace citlali::pipeline
