#pragma once

#include <citlali/core/pipeline/timestream_cal_rtc_source.h>
#include <citlali/core/pipeline/timestream_cal_atmosphere.h>
#include <citlali/core/pipeline/timestream_cal_wvr.h>
#include <citlali/core/pipeline/ast_rtc_coordinates.h>

namespace citlali::pipeline {

// Producer facts, resolved at the application boundary from the verified
// selected child APT. No source selection, rescaling or factor inference.
struct CalDetectorFactor {
    NativeReadoutDetectorBinding detector;
    std::string selected_row_identity;
    int array;
    bool uniquely_matched;
    std::optional<double> flxscale_mJy_beam_per_x;
};

class CalEvidence {
public:
    static std::shared_ptr<const CalEvidence> learn(CalRtcSource,
        std::shared_ptr<const AstRtcCoordinates>, std::shared_ptr<const CalWvrEvidence>,
        std::shared_ptr<const CalAtmosphereSurface>, std::string selected_apt_identity,
        std::vector<CalDetectorFactor>);
    const auto &source() const noexcept { return source_; }
    const auto &ast_handle() const noexcept { return ast_; }
    const auto &wvr_handle() const noexcept { return wvr_; }
    const auto &atmosphere_handle() const noexcept { return atmosphere_; }
    const auto &factors() const noexcept { return factors_; }
    const auto &apt_identity() const noexcept { return apt_identity_; }
    const auto &opacity_quality() const noexcept { return quality_; }
private:
    explicit CalEvidence(CalRtcSource source):source_{std::move(source)} {}
    CalRtcSource source_;
    std::shared_ptr<const AstRtcCoordinates> ast_;
    std::shared_ptr<const CalWvrEvidence> wvr_;
    std::shared_ptr<const CalAtmosphereSurface> atmosphere_;
    std::string apt_identity_;
    std::vector<CalDetectorFactor> factors_;
    CalWvrQuality quality_;
};

// Cause bits preserve overlapping limitations rather than choosing a single
// winner. WVR's detailed producer cause is retained separately by occurrence.
enum CalCause : std::uint16_t {
    cal_available = 0, cal_rtc_unavailable = 1, cal_direct_replacement_or_exclusion = 2,
    cal_invalid_factor = 4, cal_outside_supported_calibration = 8,
    cal_invalid_atmosphere = 16, cal_pointing_unavailable = 32, cal_numeric_failure = 64
};
class CalPlan {
public:
    struct Entry {
        std::uint16_t causes = 0;
        CalWvrCause wvr_cause = CalWvrCause::absent;
        std::optional<double> multiplier;
    };
    static std::shared_ptr<const CalPlan> consider(std::shared_ptr<const CalEvidence>,
        std::shared_ptr<const ValSnapshot>, std::uint64_t instance);
    const auto &evidence_handle() const noexcept { return evidence_; }
    const auto &snapshot_handle() const noexcept { return evidence_->source().val_snapshot_handle(); }
    const auto &entries() const noexcept { return entries_; }
    auto instance() const noexcept { return instance_; }
private:
    std::shared_ptr<const CalEvidence> evidence_;
    std::uint64_t instance_;
    std::vector<std::vector<Entry>> entries_;
};

class CalAppliedSignal {
public:
    static std::shared_ptr<const CalAppliedSignal> apply(std::shared_ptr<const CalPlan>,
        const CalRtcSource &, std::shared_ptr<const ValSnapshot>);
    const auto &plan_handle() const noexcept { return plan_; }
    std::optional<double> value(std::size_t detector,std::size_t slot) const;
    std::uint16_t causes(std::size_t detector,std::size_t slot) const;
    std::size_t available_count() const noexcept { return available_; }
    std::size_t logical_owned_cell_bytes() const noexcept {
        std::size_t n=0;for(const auto &c:cells_)n+=c.size()*sizeof(Cell);return n;
    }
    static constexpr std::string_view unit = "mJy/nominal-beam";
    static constexpr std::string_view observable = "ordinary-xs:delta_f/f_res:positive-optical-loading";
    static constexpr bool r_calibrated = false, literal_peak_response_qualified = false;
    static constexpr std::string_view uncertainty = "unavailable-no-admitted-conditional-covariance-or-complete-nuisance-ledger";
    static constexpr std::array<int,3> reference_frequency_GHz{272,214,150};
private:
    std::shared_ptr<const CalPlan> plan_;
    struct Cell { double value = 0; std::uint16_t causes = 0; };
    std::vector<std::vector<Cell>> cells_;
    std::size_t available_ = 0;
};

// CAL publishes its newly owned facts once, retaining the exact RTC facts,
// immutable plan and signal. It does not rewrite producer validity or make
// a MAP/PTC eligibility decision. The calibrated realization is distinct
// from the RTC grid even though their scheduled occurrence slots coincide.
class ValCalOutputFacts {
public:
    static std::shared_ptr<const ValCalOutputFacts> preserve(std::shared_ptr<const CalAppliedSignal>);
    const auto &signal_handle() const noexcept { return signal_; }
    std::uint16_t at(const std::shared_ptr<const CalAppliedSignal> &exact,
                     std::size_t detector,std::size_t slot) const;
private:
    explicit ValCalOutputFacts(std::shared_ptr<const CalAppliedSignal> s):signal_{std::move(s)} {}
    std::shared_ptr<const CalAppliedSignal> signal_;
};

} // namespace citlali::pipeline
