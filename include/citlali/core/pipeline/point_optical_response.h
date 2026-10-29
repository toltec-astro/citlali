#pragma once

#include <citlali/core/pipeline/timestream_ptc_numerics.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace citlali::pipeline {

// Verified saved-CAL numerical input, owned by the caller. Rows are scheduled
// CAL occurrences, never compact retained-row offsets. The adapter verifies
// hashes and exact occurrence/APT joins before constructing this value. Learning
// retains compact aggregates only; it does not retain the raw/RTC parent chain.
struct PointOpticalNetwork {
    std::string observation_identity, cal_identity, source_mask_identity;
    int array = -1, network = -1;
    std::vector<std::string> detector_ids;
    std::vector<double> times_unix_sec;
    // Nonnegative shared processing-science block identity; -1 is outside.
    std::vector<std::int64_t> science_blocks;
    PtcMatrix values;                  // mJy/nominal-beam, time x detector
    PtcMask eligible, source_free;    // binary, exact same scheduled domain
};

struct PointOpticalRequest {
    double min_relative_response = 0.10;
    double bin_seconds = 0.5;
    double confidence_level = 0.99;
    std::size_t bootstrap_samples = 2000;
};

struct PointOpticalInterval {
    bool available = false;
    double lower = NAN, upper = NAN;
    std::size_t informative_blocks = 0, resamples = 0;
    std::uint64_t seed = 0;
    std::string reason = "not_evaluated";
};

struct PointOpticalDetectorEvidence {
    std::string detector_id;
    std::size_t available_samples = 0, source_free_samples = 0, binned_samples = 0;
    double relative_response = NAN, correlation = NAN;
    double binned_residual_rms = NAN; // diagnostic only, NOT a noise weight
    PointOpticalInterval interval;
    // full, early, late x disjoint peer panels 0,1. These are conditional
    // resampled intervals, not a demonstrated repeated-sampling coverage law.
    std::array<PointOpticalInterval, 6> checks;
};

struct PointOpticalNetworkEvidence {
    std::string observation_identity, cal_identity, source_mask_identity;
    int array = -1, network = -1;
    std::vector<std::string> detector_ids;
    std::vector<int> reference_networks;
    bool reference_available = false;
    bool cross_array_required = false;
    bool cross_array_corroborated = false;
    double response_normalization = NAN;
    std::string reference_reason;
    std::vector<PointOpticalDetectorEvidence> detectors;
};

class PointOpticalEvidence {
public:
    static std::shared_ptr<const PointOpticalEvidence> learn(
        const std::vector<PointOpticalNetwork> &, PointOpticalRequest = {});
    // Adapter must verify the immutable producer publication and its method,
    // request and source bindings first. This validates internal consistency;
    // it does not turn arbitrary serialized numbers into scientific authority.
    static std::shared_ptr<const PointOpticalEvidence> restore_verified(
        PointOpticalRequest, std::vector<PointOpticalNetworkEvidence>);
    const auto &request() const noexcept { return request_; }
    const auto &networks() const noexcept { return networks_; }
    static constexpr std::string_view method =
        "point-optical-heldout-network-block-panel-bootstrap-v1";
    static constexpr std::string_view uncertainty =
        "conditional-percentile-interval; frozen robust-fit weights and normal-response scale; "
        "science-block exchangeability assumed; disjoint peer panels; "
        "coverage and observation-wide false-rejection rate unvalidated";
private:
    PointOpticalRequest request_;
    std::vector<PointOpticalNetworkEvidence> networks_;
};

enum class PointOpticalState {
    prior_unavailable, retained, weak_response, sign_mismatch, unresolved
};
std::string_view point_optical_state_name(PointOpticalState);

struct PointOpticalDecision {
    PointOpticalState state = PointOpticalState::unresolved;
    std::string reason;
    bool retained() const noexcept { return state == PointOpticalState::retained; }
};

// An observation-frozen membership. Threshold tuning changes Consider; it does
// not relearn the peer cohort or repeatedly trim a chosen fraction of detectors.
class PointOpticalPlan {
public:
    static std::shared_ptr<const PointOpticalPlan> consider(
        std::shared_ptr<const PointOpticalEvidence>, double min_relative_response);
    static std::shared_ptr<const PointOpticalPlan> consider(
        std::shared_ptr<const PointOpticalEvidence>);
    const auto &evidence_handle() const noexcept { return evidence_; }
    double min_relative_response() const noexcept { return threshold_; }
    const auto &decisions() const noexcept { return decisions_; }
    // Verifies exact scope/CAL/mask binding and detector ordering before exposing
    // membership to the weighted PTC consumer. No value-based identity guessing.
    const std::vector<PointOpticalDecision> &for_network(const PointOpticalNetwork &) const;
private:
    std::shared_ptr<const PointOpticalEvidence> evidence_;
    double threshold_ = 0.10;
    std::vector<std::vector<PointOpticalDecision>> decisions_;
};

} // namespace citlali::pipeline
