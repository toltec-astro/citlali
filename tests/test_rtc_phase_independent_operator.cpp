#include <kids/toltec/toltec.h>

#include <citlali/core/timestream/rtc/rtcproc.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using timestream::RTCInfluenceCause;
using timestream::RTCInfluenceLedger;
using timestream::RTCPhysicalEventSemantics;
using timestream::RTCResponseParity;
using timestream::RTCResponseStage;
using timestream::RTCResponseUnavailableCause;

bool has_cause(RTCInfluenceCause value, RTCInfluenceCause cause) {
    return timestream::rtc_has_influence_cause(value, cause);
}

std::uint32_t response_stage_bits(RTCResponseStage value) {
    return static_cast<std::uint32_t>(value);
}

std::uint32_t unavailable_cause_bits(
    RTCResponseUnavailableCause value) {
    return static_cast<std::uint32_t>(value);
}

TEST(RtcPhaseIndependentInfluence,
     ReplacementNonfiniteFirAndDecimationSupportStayIneligible) {
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> before(12, 2);
    before.setConstant(false);
    before(1, 1) = true;

    auto after = before;
    after(4, 0) = true;
    after(5, 0) = true;

    Eigen::MatrixXd signal = Eigen::MatrixXd::Zero(12, 2);
    signal(8, 1) = std::numeric_limits<double>::quiet_NaN();

    RTCInfluenceLedger ledger(12, 2);
    ledger.mark_flagged(before, RTCInfluenceCause::input_ineligible);
    ledger.mark_newly_flagged(before, after);
    ledger.mark_nonfinite(signal);

    EXPECT_TRUE(has_cause(
        ledger.causes_at(4, 0),
        RTCInfluenceCause::replacement_or_synthesis));
    EXPECT_TRUE(has_cause(
        ledger.causes_at(8, 1), RTCInfluenceCause::nonfinite_payload));

    ledger.propagate_fir(1);
    EXPECT_TRUE(has_cause(
        ledger.causes_at(3, 0),
        RTCInfluenceCause::replacement_or_synthesis));
    EXPECT_TRUE(has_cause(
        ledger.causes_at(3, 0), RTCInfluenceCause::fir_support));
    EXPECT_TRUE(has_cause(
        ledger.causes_at(0, 0),
        RTCInfluenceCause::incomplete_filter_edge));
    EXPECT_TRUE(has_cause(
        ledger.causes_at(11, 1),
        RTCInfluenceCause::incomplete_filter_edge));
    EXPECT_TRUE(ledger.scientifically_eligible(7, 0));

    const auto decimated = ledger.downsample_phase_zero(2);
    EXPECT_EQ(decimated.assigned_sample_count(), 6);
    EXPECT_TRUE(has_cause(
        decimated.causes_at(1, 0),
        RTCInfluenceCause::replacement_or_synthesis));
    EXPECT_TRUE(has_cause(
        decimated.causes_at(1, 0),
        RTCInfluenceCause::decimation_support));
    EXPECT_TRUE(has_cause(
        decimated.causes_at(4, 1),
        RTCInfluenceCause::nonfinite_payload));

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> output_flags(6, 2);
    output_flags.setConstant(false);
    decimated.enforce_ineligible_flags(output_flags);
    EXPECT_TRUE(output_flags(1, 0));
    EXPECT_TRUE(output_flags(4, 1));
    EXPECT_FALSE(output_flags(4, 0));
}

TEST(RtcPhaseIndependentInfluence,
     RecursiveSupportPreservesCauseForCausalAndZeroPhaseStages) {
    RTCInfluenceLedger causal(10, 1);
    causal.add_interval(
        0, 4, 5, RTCInfluenceCause::replacement_or_synthesis);
    causal.propagate_recursive_filter(false);

    EXPECT_TRUE(causal.scientifically_eligible(3, 0));
    EXPECT_FALSE(causal.scientifically_eligible(4, 0));
    EXPECT_FALSE(causal.scientifically_eligible(9, 0));
    EXPECT_TRUE(has_cause(
        causal.causes_at(9, 0),
        RTCInfluenceCause::replacement_or_synthesis));
    EXPECT_TRUE(has_cause(
        causal.causes_at(9, 0),
        RTCInfluenceCause::recursive_filter_support));

    RTCInfluenceLedger zero_phase(10, 1);
    zero_phase.add_interval(
        0, 4, 5, RTCInfluenceCause::replacement_or_synthesis);
    zero_phase.propagate_recursive_filter(true);

    EXPECT_FALSE(zero_phase.scientifically_eligible(0, 0));
    EXPECT_FALSE(zero_phase.scientifically_eligible(9, 0));
    EXPECT_TRUE(has_cause(
        zero_phase.causes_at(0, 0),
        RTCInfluenceCause::replacement_or_synthesis));
}

TEST(RtcPhaseIndependentInfluence,
     IntervalNormalizationAndRepeatedExecutionAreDeterministic) {
    auto construct = [] {
        RTCInfluenceLedger ledger(16, 2);
        ledger.add_interval(
            1, 6, 7, RTCInfluenceCause::replacement_or_synthesis);
        ledger.add_interval(
            1, 8, 9, RTCInfluenceCause::replacement_or_synthesis);
        ledger.propagate_fir(2);
        return ledger.downsample_phase_zero(4);
    };

    const auto first = construct();
    const auto second = construct();
    ASSERT_EQ(first.intervals().size(), second.intervals().size());
    for (std::size_t index = 0; index < first.intervals().size(); ++index) {
        const auto &lhs = first.intervals()[index];
        const auto &rhs = second.intervals()[index];
        EXPECT_EQ(lhs.detector, rhs.detector);
        EXPECT_EQ(lhs.first_assigned_sample, rhs.first_assigned_sample);
        EXPECT_EQ(lhs.last_assigned_sample, rhs.last_assigned_sample);
        EXPECT_EQ(static_cast<std::uint32_t>(lhs.causes),
                  static_cast<std::uint32_t>(rhs.causes));
    }
}

TEST(RtcPhaseIndependentResponse,
     CompleteSignalResponseParityCoversDeterministicBasisSignals) {
    constexpr Eigen::Index n_samples = 64;
    Eigen::MatrixXd signal(n_samples, 4);
    signal.col(0).setOnes();
    signal.col(1).setZero();
    signal(n_samples / 2, 1) = 1.0;
    signal.col(2) = Eigen::VectorXd::LinSpaced(n_samples, -1.0, 1.0);
    for (Eigen::Index sample = 0; sample < n_samples; ++sample) {
        signal(sample, 3) = std::sin(
            2.0 * std::acos(-1.0) * 4.0 *
            static_cast<double>(sample) /
            static_cast<double>(n_samples));
    }
    Eigen::MatrixXd response = signal;

    RTCResponseParity parity;

    timestream::Filter fir;
    fir.n_terms = 1;
    fir.filter.resize(3);
    fir.filter << 0.25, 0.5, 0.25;
    parity.apply_matched_in_place(
        RTCResponseStage::fir, signal, &response,
        [&](Eigen::MatrixXd &value) { fir.convolve(value); });

    timestream::Filter notch;
    notch.notch_zero_phase = true;
    notch.w0s = {4.0};
    notch.qs = {8.0};
    notch.make_notch_filter(64.0);
    parity.apply_matched_in_place(
        RTCResponseStage::notch, signal, &response,
        [&](Eigen::MatrixXd &value) { notch.iir(value); });

    timestream::Filter highpass;
    highpass.iir_highpass_freq_Hz = 0.5;
    highpass.iir_highpass_order = 1;
    highpass.iir_highpass_zero_phase = false;
    parity.apply_matched_in_place(
        RTCResponseStage::iir_highpass, signal, &response,
        [&](Eigen::MatrixXd &value) {
            highpass.iir_highpass(value, 64.0);
        });

    EXPECT_TRUE(signal.allFinite());
    EXPECT_DOUBLE_EQ((signal - response).squaredNorm(), 0.0);
    EXPECT_TRUE(parity.complete_response_available());
    EXPECT_EQ(response_stage_bits(parity.signal_stages()),
              response_stage_bits(parity.response_stages()));
    EXPECT_EQ(
        unavailable_cause_bits(parity.unavailable_causes()), 0u);
    EXPECT_EQ(parity.physical_event_semantics(),
              RTCPhysicalEventSemantics::unavailable);
    EXPECT_EQ(timestream::rtc_physical_event_semantics_name(
                  parity.physical_event_semantics()),
              "unavailable");
}

TEST(RtcPhaseIndependentResponse,
     PhaseZeroDownsampleAppliesTheSameAssignedGridTransform) {
    Eigen::MatrixXd signal_input(9, 2);
    signal_input.col(0) = Eigen::VectorXd::LinSpaced(9, -2.0, 2.0);
    signal_input.col(1).setZero();
    signal_input(4, 1) = 1.0;
    Eigen::MatrixXd response_input = signal_input;
    Eigen::MatrixXd signal_output;
    Eigen::MatrixXd response_output;

    timestream::Downsampler downsampler;
    downsampler.factor = 2;

    RTCResponseParity parity;
    parity.apply_matched_transform(
        RTCResponseStage::downsample,
        signal_input, signal_output,
        &response_input, &response_output,
        [&](Eigen::MatrixXd &input, Eigen::MatrixXd &output) {
            downsampler.downsample(input, output);
        });

    ASSERT_EQ(signal_output.rows(), 5);
    ASSERT_EQ(signal_output.cols(), 2);
    EXPECT_DOUBLE_EQ((signal_output - response_output).squaredNorm(), 0.0);
    for (Eigen::Index output_sample = 0;
         output_sample < signal_output.rows(); ++output_sample) {
        EXPECT_DOUBLE_EQ(signal_output(output_sample, 0),
                         signal_input(2 * output_sample, 0));
    }
    EXPECT_TRUE(parity.complete_response_available());
    EXPECT_EQ(parity.physical_event_semantics(),
              RTCPhysicalEventSemantics::unavailable);
}

TEST(RtcPhaseIndependentResponse,
     MissingOrUnrepresentedResponseFailsClosedWithoutTimingClaims) {
    Eigen::MatrixXd signal = Eigen::MatrixXd::Ones(8, 1);
    RTCResponseParity parity;
    parity.apply_matched_in_place(
        RTCResponseStage::fir, signal, nullptr,
        [](Eigen::MatrixXd &value) { value *= 2.0; });
    parity.mark_unrepresented_signal_stage(
        RTCResponseStage::replacement,
        RTCResponseUnavailableCause::
            replacement_donor_mixing_unrepresented);
    parity.mark_unrepresented_signal_stage(
        RTCResponseStage::altaz_projection,
        RTCResponseUnavailableCause::projection_unrepresented);

    EXPECT_FALSE(parity.complete_response_available());
    EXPECT_EQ(parity.physical_event_semantics(),
              RTCPhysicalEventSemantics::unavailable);
    EXPECT_NE(
        unavailable_cause_bits(parity.unavailable_causes()) &
            unavailable_cause_bits(
                RTCResponseUnavailableCause::response_missing),
        0u);
    EXPECT_NE(
        unavailable_cause_bits(parity.unavailable_causes()) &
            unavailable_cause_bits(
                RTCResponseUnavailableCause::
                    replacement_donor_mixing_unrepresented),
        0u);
    EXPECT_NE(
        unavailable_cause_bits(parity.unavailable_causes()) &
            unavailable_cause_bits(
                RTCResponseUnavailableCause::projection_unrepresented),
        0u);
}

TEST(RtcPhaseIndependentResponse,
     ShapeMismatchCannotProduceACompleteResponseClaim) {
    Eigen::MatrixXd signal = Eigen::MatrixXd::Ones(8, 1);
    Eigen::MatrixXd response = Eigen::MatrixXd::Ones(7, 1);
    RTCResponseParity parity;
    parity.apply_matched_in_place(
        RTCResponseStage::fir, signal, &response,
        [](Eigen::MatrixXd &value) { value *= 2.0; });

    EXPECT_FALSE(parity.complete_response_available());
    EXPECT_NE(
        unavailable_cause_bits(parity.unavailable_causes()) &
            unavailable_cause_bits(
                RTCResponseUnavailableCause::response_shape_mismatch),
        0u);
    EXPECT_EQ(response_stage_bits(parity.response_stages()), 0u);
}

}  // namespace
