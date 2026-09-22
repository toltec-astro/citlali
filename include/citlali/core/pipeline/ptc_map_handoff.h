#pragma once
#include <citlali/core/pipeline/timestream_ptc_numerics.h>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace citlali::pipeline {
// Persisted PTC adapter boundary. The adapter verifies required components and
// supplies this exact, completed single-segment publication. Locators resolve
// through stable detector identities and the immutable RTC slot-generation.
struct PtcMapPublicationIdentity {
    std::string publication, application, cal_parent, rtc_grid, processing,
        observation, segment, support_reference;
    int network, array, rank;
    std::vector<std::string> detectors;
    std::vector<std::int64_t> slots;
};
class PtcMapPublication {
public:
    static std::shared_ptr<const PtcMapPublication> bind_verified_segment(
        PtcMapPublicationIdentity, PtcMatrix, PtcMask, bool converged);
    const auto &identity() const noexcept { return identity_; }
    const auto &values() const noexcept { return values_; }
    const auto &causes() const noexcept { return causes_; }
    auto retained() const noexcept { return retained_; }
    bool member(std::size_t time, std::size_t detector) const;
private:
    PtcMapPublicationIdentity identity_;
    PtcMatrix values_;
    PtcMask causes_;
    std::size_t retained_=0;
};

// VAL truth composition, restricted here to the PTC uniform handoff. These are
// separate facts; an F wins over independent U/C, but C is never split into F.
enum class UniformFact { true_value, false_value, unknown, conflict };
UniformFact uniform_all(std::span<const UniformFact>);
struct UniformSelection {
    std::string identity;
    std::optional<std::string> family;
    bool conflicting=false;
};
struct UniformDecision {
    bool requested=false, structurally_bound=false, applicable=false, realized=false;
    std::array<UniformFact,7> restrictions;
    bool pass() const noexcept;
private:
    friend class ValUniformHandoffEvaluator;
    friend class PtcUniformCoefficients;
    const void *coefficient_=nullptr;
    std::size_t time_=0, detector_=0;
    bool evaluated_pass_=false;
};
class PtcUniformCoefficients {
public:
    static std::shared_ptr<const PtcUniformCoefficients> publish(
        std::shared_ptr<const PtcMapPublication>, UniformSelection,
        std::string coefficient_identity, std::string stored_support_reference);
    std::optional<double> value(const UniformDecision &, std::size_t time, std::size_t detector) const;
    const auto &parent() const noexcept { return parent_; }
    const auto &identity() const noexcept { return identity_; }
    const auto &selection() const noexcept { return selection_; }
    const auto &shared_restrictions() const noexcept { return restrictions_; }
    static constexpr std::string_view registry="SCI-PTC:analysis_gridding_coefficients@draft-0.1";
    static constexpr std::string_view admission="PTC-UNIFORM-ADMISSION r0.4/2026-09-06";
    static constexpr std::string_view family="SCI-PTC:uniform_constant@draft-0.1";
    static constexpr std::string_view profile="SCI-PTC:uniform_coefficient_handoff@draft-0.1";
    static constexpr std::string_view source_sha256="8bc42deb69e23c757883834f7cdb82b94e89210015333f483d15c32377a56685";
private:
    friend class ValUniformHandoffEvaluator;
    std::shared_ptr<const PtcMapPublication> parent_;
    UniformSelection selection_;
    std::string identity_, stored_support_;
    std::array<UniformFact,7> restrictions_;
};
// VAL evaluates the PTC-authored uniform restrictions; PTC owns the payload.
class ValUniformHandoffEvaluator {
public:
    static UniformDecision evaluate(const PtcUniformCoefficients &,
        const std::shared_ptr<const PtcMapPublication> &exact,
        std::size_t time,std::size_t detector,bool requested);
};
} // namespace citlali::pipeline
