#include "timestream_rtc_reassessment_test_support.h"
#include <bit>
#include <chrono>
#include <citlali/core/pipeline/timestream_rtc_line_transfer.h>
#include <citlali/core/timestream/rtc/filter.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>

namespace {
using namespace citlali::pipeline;
using Input = citlali::test::rtc_reassessment::Input;
constexpr auto X = NativeReadoutCoordinate::x, R = NativeReadoutCoordinate::r;
struct Trial {
  std::shared_ptr<const NativePairedReadoutObservation> parent;
  std::shared_ptr<const ValSnapshot> val;
  std::shared_ptr<const RtcNativeSpectralEvidence> spectral;
  std::shared_ptr<const RtcLinePowerEvidence> lines;
  std::shared_ptr<const RtcLinePowerConsideration> joint;
  explicit Trial(
      Input in, RtcSpikeProtection protection = RtcSpikeProtection::unavailable,
      double tolerance = 1e-8) {
    parent = in.freeze();
    val = ValSnapshot::initial(parent);
    auto spikes = learn_rtc_spike_candidates(
        NativePairedReadoutView::full(parent), val,
        RtcSpikeSourceProtection::admit(parent, "transfer-fixture-source",
                                        protection),
        1);
    auto native =
        ValNativeRealization::create(parent, {ValProducer::align, 1}, 1,
                                     ValNativeProductRole::original_input, 0);
    auto identity = RtcSpectralInputIdentity::bind(
        native, val, spikes->input_handle()->span(0),
        RtcSpectralInputStage::original_reference, "transfer-native", 1);
    spectral = RtcNativeSpectralEvidence::learn_initial(
        spikes, {identity},
        {{0, "transfer-cadence", 2 * in.integration_half, tolerance}}, 2);
    lines = RtcLinePowerEvidence::learn(spectral, val,
                                        RtcLinePowerProfile::initial_2_hz, 3);
    std::vector<RtcEventPeerEligibility> peers;
    for (std::uint32_t d = 0; d < 3; ++d)
      peers.push_back(
          {0, d, parent->network(0).detector(d).detector_occurrence_id, true});
    auto events = RtcEventAssessmentDecision::consider(
        learn_rtc_event_assessment(
            spikes,
            RtcEventPeerPopulation::admit(spikes, "transfer-peers", peers), 4),
        val, 5);
    joint = RtcLinePowerConsideration::rank(
        lines,
        RtcSpectralTransientConsideration::consider(spectral, val, events, val,
                                                    6),
        7);
  }
  auto candidate(RtcLineTransferSpecification s) const {
    return RtcLineTransferCandidate::bind(lines, 0, 0, std::move(s));
  }
  auto assess(RtcLineTransferSpecification s) const {
    return RtcLineTransferAssessment::consider(candidate(std::move(s)), joint,
                                               val, 8);
  }
};
Input input(std::size_t n = 2048) {
  Input in(n, 1. / 128);
  in.x.setZero();
  in.r.setZero();
  for (std::size_t i = 0; i < n; ++i) {
    const double t = i / 128.;
    in.x(i, 0) = 2 * std::sin(2 * std::numbers::pi * 11 * t + .3) +
                 std::sin(2 * std::numbers::pi * 29 * t + .7) +
                 .1 * std::cos(2 * std::numbers::pi * 3.3 * t);
    in.r(i, 0) = 4 * std::sin(2 * std::numbers::pi * 48 * t + .5);
  }
  return in;
}
RtcLineTransferSpecification spec(unsigned factor = 4) {
  RtcLineTransferSpecification s;
  s.identity = "explicit-trial";
  s.lowpass_identity = "three-tap-binomial-fixture";
  s.state_support_identity = "steady-state-only;finite-record-unqualified";
  s.input_interval_seconds = 1. / 128;
  s.factor = factor;
  s.centered_lowpass = {.25, .5, .25};
  return s;
}
RtcNotchResponseSection notch(double hz = 11, double width = .5,
                              bool zero_phase = true, double rate = 128) {
  // Reuse the mature coefficient constructor as a controlled trial, not a
  // scientific choice of family/width or a runtime filter-design operation.
  timestream::Filter f;
  f.w0s = {hz};
  f.qs = {hz / width};
  f.make_notch_filter(rate);
  RtcNotchResponseSection n;
  n.identity = "mature-notch-fixture-" + std::to_string(hz);
  for (std::size_t j = 0; j < 3; ++j) {
    n.a[j] = f.notch_a[0](j);
    n.b[j] = f.notch_b[0](j);
  }
  n.direction = zero_phase ? RtcNotchResponseDirection::forward_reverse
                           : RtcNotchResponseDirection::causal;
  return n;
}
TEST(rtc_line_transfer,
     identity_preserves_all_native_bin_power_and_exact_context) {
  Trial t(input());
  auto s = spec(1);
  s.centered_lowpass = {1};
  auto result = t.assess(s);
  EXPECT_EQ(result->snapshot_handle(), t.val);
  EXPECT_EQ(result->joint_handle(), t.joint);
  EXPECT_EQ(result->candidate_handle()->line_handle(), t.lines);
  for (auto &c : result->coordinates()) {
    ASSERT_TRUE(c.available());
    EXPECT_DOUBLE_EQ(c.incoherent_folded_power_proxy, 0);
    for (auto &b : c.bins) {
      EXPECT_DOUBLE_EQ(b.input_power, b.combined_power);
      EXPECT_DOUBLE_EQ(b.input_hz, b.folded_output_hz);
      EXPECT_FALSE(b.above_output_nyquist);
    }
    const auto a = t.lines->measure_band(0, 0, c.coordinate, "full", 0, 64);
    auto b = result->measure_band(c.coordinate, "full", 0, 64);
    EXPECT_NEAR(b.combined_stored_power, a.stored_psd_power,
                1e-13 * a.stored_psd_power);
    EXPECT_NEAR(b.combined_signed_residual_power, a.signed_residual_power,
                1e-13 * a.stored_psd_power);
  }
  EXPECT_FALSE(result->apply_authorized);
  EXPECT_FALSE(result->interference_admitted);
  EXPECT_FALSE(result->finite_record_response_qualified);
  EXPECT_FALSE(result->coherent_alias_cross_terms_available);
  EXPECT_FALSE(result->candidate_handle()->filter_bank_certified);
}
TEST(rtc_line_transfer,
     lowpass_and_notch_compose_with_forward_reverse_power_convention) {
  Trial t(input());
  auto s = spec();
  s.notches = {notch()};
  auto c = t.candidate(s);
  for (double f : {0., 11., 11.25, 29., 48., 64.}) {
    const double w = 2 * std::numbers::pi * f / 128;
    const auto z = std::exp(std::complex<double>{0, -w});
    const auto &n = s.notches[0];
    const auto h = (n.b[0] + n.b[1] * z + n.b[2] * z * z) /
                   (n.a[0] + n.a[1] * z + n.a[2] * z * z);
    const double lp = .5 + .5 * std::cos(w);
    EXPECT_NEAR(std::abs(c->combined_response(f) -
                         std::complex<double>{std::norm(h) * lp, 0}),
                0, 2e-13);
  }
  EXPECT_LT(std::abs(c->combined_response(11)), 1e-20);
  auto a = t.assess(s);
  ASSERT_TRUE(a->coordinates()[0].available());
  bool suppressed = false;
  for (const auto &r : a->coordinates()[0].regions)
    if (t.lines->coordinates()[0].regions[r.source_region].peak_bin == 44) {
      EXPECT_LT(r.combined_positive_excess_power,
                r.lowpass_only_positive_excess_power);
      suppressed = true;
    }
  EXPECT_TRUE(suppressed);
}
TEST(rtc_line_transfer,
     response_agrees_with_independent_causal_impulse_recurrence) {
  Trial t(input());
  auto s = spec(1);
  s.notches = {notch(11, .5, false)};
  auto c = t.candidate(s);
  std::vector<double> h(16384);
  const auto &n = s.notches[0];
  for (std::size_t i = 0; i < h.size(); ++i) {
    h[i] = (i < 3 ? n.b[i] : 0) - (i > 0 ? n.a[1] * h[i - 1] : 0) -
           (i > 1 ? n.a[2] * h[i - 2] : 0);
  }
  for (double f : {0., 10.5, 11., 11.4, 29., 48.}) {
    std::complex<long double> measured{0, 0};
    for (std::size_t i = 0; i < h.size(); ++i) {
      const long double w = -2 * std::numbers::pi_v<long double> * f * i / 128;
      measured += static_cast<long double>(h[i]) *
                  std::complex<long double>{std::cos(w), std::sin(w)};
    }
    const double lp = .5 + .5 * std::cos(2 * std::numbers::pi * f / 128);
    EXPECT_NEAR(
        std::abs(std::complex<double>{static_cast<double>(measured.real()),
                                      static_cast<double>(measured.imag())} *
                     lp -
                 c->combined_response(f)),
        0, 2e-12);
  }
}
TEST(rtc_line_transfer,
     signed_mirrored_folding_uses_actual_factor_and_keeps_native_cells) {
  Trial t(input());
  auto s = spec(8);
  auto r = t.assess(s);
  const auto &b = r->coordinates()[0].bins;
  EXPECT_DOUBLE_EQ(b[44].input_hz, 11);
  EXPECT_DOUBLE_EQ(b[44].folded_output_hz, 5);
  EXPECT_DOUBLE_EQ(b[116].folded_output_hz, 3);
  EXPECT_DOUBLE_EQ(b[192].folded_output_hz, 0);
  EXPECT_FALSE(b[32].above_output_nyquist);
  EXPECT_TRUE(b[33].above_output_nyquist);
  long double expected = 0;
  for (auto &x : b)
    if (x.input_hz > 8)
      expected += x.combined_power;
  EXPECT_DOUBLE_EQ(r->coordinates()[0].incoherent_folded_power_proxy,
                   static_cast<double>(expected));
  EXPECT_EQ(b.size(), t.spectral->network(0).frequency_hz.size());
  EXPECT_DOUBLE_EQ(rtc_line_transfer_detail::fold(-29, 16), 3);
  auto other = t.assess(spec(2));
  EXPECT_DOUBLE_EQ(other->coordinates()[0].bins[192].folded_output_hz, 16);
}
TEST(rtc_line_transfer,
     extra_notch_benefit_accounts_for_already_supplied_lowpass) {
  Trial t(input());
  auto s = spec(4);
  s.notches = {notch(48)};
  auto a = t.assess(s);
  const auto &r = a->coordinates()[1];
  ASSERT_TRUE(r.available());
  const auto &b = r.bins[192];
  EXPECT_LT(b.lowpass_only_power, b.input_power * .03);
  EXPECT_LT(b.combined_power, b.lowpass_only_power * 1e-12);
  auto identity = spec(4);
  identity.centered_lowpass = {1};
  identity.notches = s.notches;
  auto c = t.assess(identity);
  EXPECT_LT(b.combined_power, c->coordinates()[1].bins[192].combined_power);
  EXPECT_FALSE(a->apply_authorized);
}
TEST(rtc_line_transfer,
     overlapping_notches_are_one_complete_ordered_candidate) {
  Trial t(input());
  auto a = spec(4);
  a.notches = {notch(11), notch(11.4)};
  auto candidate = t.candidate(a);
  const double f = 11.2;
  auto first = a;
  first.notches.resize(1);
  first.centered_lowpass = {1};
  auto second = first;
  second.notches = {a.notches[1]};
  const auto lp = t.candidate(spec(4))->combined_response(f);
  EXPECT_NEAR(std::abs(candidate->combined_response(f) -
                       t.candidate(first)->combined_response(f) *
                           t.candidate(second)->combined_response(f) * lp),
              0, 1e-13);
  EXPECT_EQ(candidate->specification().notches[1].identity,
            a.notches[1].identity);
}
TEST(rtc_line_transfer,
     candidate_copies_coefficients_and_preserves_original_xr_and_psd) {
  auto in = input();
  Trial t(in);
  const auto saved = t.spectral->spectrum(0, 0, X).psd;
  auto s = spec();
  auto c = t.candidate(s);
  s.centered_lowpass[1] = 99;
  auto r = RtcLineTransferAssessment::consider(c, t.joint, t.val, 8);
  EXPECT_DOUBLE_EQ(c->specification().centered_lowpass[1], .5);
  EXPECT_EQ(saved, t.spectral->spectrum(0, 0, X).psd);
  for (std::size_t i = 0; i < in.times.size(); ++i)
    for (std::size_t d = 0; d < 3; ++d)
      for (auto coordinate : {X, R})
        EXPECT_EQ(std::bit_cast<std::uint64_t>(
                      t.parent->network(0).value(coordinate, 100 + i, d)),
                  std::bit_cast<std::uint64_t>(
                      (coordinate == X ? in.x : in.r)(i, d)));
}
TEST(rtc_line_transfer, exact_evidence_snapshot_and_attempt_are_required) {
  Trial t(input()), other(input());
  auto c = t.candidate(spec());
  EXPECT_THROW(RtcLineTransferAssessment::consider(c, other.joint, t.val, 8),
               std::invalid_argument);
  EXPECT_THROW(RtcLineTransferAssessment::consider(
                   c, t.joint, ValSnapshot::initial(t.parent), 8),
               std::invalid_argument);
  EXPECT_THROW(RtcLineTransferAssessment::consider(c, t.joint, t.val, 0),
               std::invalid_argument);
  EXPECT_THROW(RtcLineTransferAssessment::consider(nullptr, t.joint, t.val, 8),
               std::invalid_argument);
  auto s = spec();
  s.input_interval_seconds = 1. / 127;
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
  EXPECT_THROW(RtcLineTransferCandidate::bind(t.lines, 0, 9999, s),
               std::out_of_range);
}
TEST(rtc_line_transfer,
     malformed_and_unstable_trials_are_rejected_without_normalization) {
  Trial t(input());
  auto s = spec();
  s.factor = 0;
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
  s = spec();
  s.factor = 257;
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
  s = spec();
  s.centered_lowpass = {.5, .5};
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
  s = spec();
  s.centered_lowpass = {.2, .5, .3};
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
  s = spec();
  s.centered_lowpass[0] = NAN;
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
  s = spec();
  s.identity.clear();
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
  s = spec();
  s.notches = {notch()};
  s.notches[0].a = {1, 0, 1};
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
  s.notches[0].a = {2, 0, 0};
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
  s.notches[0].a = {1, 0, 0};
  s.notches[0].b[0] = INFINITY;
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
  s = spec();
  s.notches = {notch(), notch()};
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
  s = spec();
  s.input_interval_seconds = std::numeric_limits<double>::max();
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
}
TEST(rtc_line_transfer,
     unavailable_inputs_and_overflow_are_not_available_zero_residuals) {
  auto in = input();
  for (std::size_t i = 0; i < in.times.size(); ++i) {
    in.x(i, 0) = NAN;
    in.xs[i * 3] =
        NativeReadoutCoordinateState::measured(true, false, true, false);
  }
  Trial t(in);
  auto r = t.assess(spec());
  EXPECT_EQ(r->coordinates()[0].cause,
            RtcLineTransferCause::spectral_unavailable);
  EXPECT_TRUE(r->coordinates()[1].available());
  EXPECT_THROW(r->measure_band(X, "missing", 10, 12), std::invalid_argument);
  Trial good(input());
  auto s = spec();
  s.centered_lowpass = {std::numeric_limits<double>::max()};
  r = good.assess(s);
  EXPECT_EQ(r->coordinates()[0].cause,
            RtcLineTransferCause::arithmetic_unavailable);
  EXPECT_TRUE(r->coordinates()[0].bins.empty());
  EXPECT_TRUE(std::isnan(r->coordinates()[0].incoherent_folded_power_proxy));
}
TEST(rtc_line_transfer,
     source_protection_is_retained_and_r_does_not_authorize_shared_action) {
  for (auto protection :
       {RtcSpikeProtection::protected_source,
        RtcSpikeProtection::outside_source, RtcSpikeProtection::unavailable}) {
    Trial t(input(), protection);
    auto r = t.assess(spec());
    EXPECT_EQ(r->joint_handle(), t.joint);
    EXPECT_TRUE(r->coordinates()[0].available());
    EXPECT_TRUE(r->coordinates()[1].available());
    EXPECT_FALSE(r->interference_admitted);
    EXPECT_FALSE(r->apply_authorized);
  }
}
TEST(rtc_line_transfer,
     signed_named_band_retains_negative_residual_and_declared_selection) {
  Trial t(input());
  auto s = spec();
  s.centered_lowpass = {.5};
  auto r = t.assess(s);
  auto base = t.lines->measure_band(0, 0, X, "named", 0, 64);
  auto band = r->measure_band(X, "named", 0, 64);
  EXPECT_NEAR(band.combined_signed_residual_power,
              base.signed_residual_power * .25, 1e-14 * base.stored_psd_power);
  EXPECT_NEAR(band.combined_background_power, base.background_power * .25,
              1e-14 * base.stored_psd_power);
  EXPECT_THROW(r->measure_band(X, "", 0, 64), std::invalid_argument);
  EXPECT_THROW(r->measure_band(X, "outside", 0, 65), std::invalid_argument);
}
TEST(rtc_line_transfer,
     science_domain_reuses_accepted_airy_scale_and_remains_conditional) {
  Trial t(input());
  auto s = spec();
  auto no = t.assess(s);
  EXPECT_FALSE(no->coordinates()[0].science);
  s.science_domain = RtcTransferScienceDomain{"trial-a1100-100arcsec",
                                              RtcOpticalArray::a1100, 100};
  auto r = t.assess(s);
  auto q = *r->coordinates()[0].science;
  const double radians = std::numbers::pi / (180 * 3600),
               lambda = 299792458. / 272e9;
  EXPECT_DOUBLE_EQ(q.full_temporal_support_hz, 100 * radians * 50 / lambda);
  EXPECT_DOUBLE_EQ(q.airy_fwhm_arcsec,
                   1.028993969962188 * lambda / 50 / radians);
  EXPECT_GT(q.sampled_bins, 1);
  EXPECT_GT(q.maximum_sampled_magnitude_error, 0);
  EXPECT_FALSE(q.exceeds_native_nyquist);
  EXPECT_TRUE(q.exceeds_output_nyquist);
  EXPECT_FALSE(r->coordinates()[1].science);
  s.science_domain->trial_speed_arcsec_per_sec = 1000;
  r = t.assess(s);
  EXPECT_TRUE(r->coordinates()[0].science->exceeds_native_nyquist);
  EXPECT_FALSE(r->apply_authorized);
  s.science_domain->trial_speed_arcsec_per_sec = 0;
  r = t.assess(s);
  EXPECT_EQ(r->coordinates()[0].science->sampled_bins, 1);
  s.science_domain->trial_speed_arcsec_per_sec = -1;
  EXPECT_THROW(t.candidate(s), std::invalid_argument);
}
TEST(rtc_line_transfer,
     causal_phase_is_preserved_and_positive_negative_response_conjugates) {
  Trial t(input());
  auto s = spec();
  s.notches = {notch(11, .5, false)};
  auto c = t.candidate(s);
  EXPECT_NEAR(std::abs(c->combined_response(-11.25) -
                       std::conj(c->combined_response(11.25))),
              0, 1e-14);
  EXPECT_GT(std::abs(c->combined_response(11.25).imag()), .01);
  EXPECT_THROW(c->combined_response(65), std::invalid_argument);
  EXPECT_THROW(c->combined_response(NAN), std::invalid_argument);
}
} // namespace

namespace {
TEST(rtc_line_transfer,
     incoherent_power_is_not_a_bound_for_phase_locked_aliases) {
  long double combined = 0, independent = 0;
  for (int i = 0; i < 1024; ++i) {
    const double a = std::cos(2 * std::numbers::pi * 3 * i / 16),
                 b = std::cos(2 * std::numbers::pi * 29 * i / 16);
    combined += (a + b) * (a + b);
    independent += a * a + b * b;
  }
  EXPECT_NEAR(static_cast<double>(combined / independent), 2, 1e-12);
  EXPECT_FALSE(RtcLineTransferAssessment::coherent_alias_cross_terms_available);
}
TEST(rtc_line_transfer,
     controlled_and_retained_real_fixture_export_exact_trials) {
  const auto fixture =
      std::filesystem::path{__FILE__}.parent_path() /
      "fixtures/timestream_rtc_event_assessment/case_e_original.txt";
  std::ifstream f(fixture);
  ASSERT_TRUE(f);
  std::array<double, 4> row;
  std::vector<std::array<double, 4>> rows;
  while (f >> row[0] >> row[1] >> row[2] >> row[3])
    rows.push_back(row);
  ASSERT_EQ(rows.size(), 550);
  Input real(rows.size(), .008192);
  real.x.setZero();
  real.r.setZero();
  for (std::size_t i = 0; i < rows.size(); ++i) {
    real.times[i] = 1000 + rows[i][1] - rows[0][1];
    real.x(i, 0) = rows[i][2];
    real.r(i, 0) = rows[i][3];
  }
  // Retained original values in the already accepted reindexed test parent;
  // this does not establish operational source/AST/cadence authority.
  Trial controlled(input()),
      retained(real, RtcSpikeProtection::unavailable, 1e-4);
  std::ofstream out, coefficients, support;
  if (const auto *path = std::getenv("CITLALI_TRANSFER_REPORT")) {
    out.open(path);
    coefficients.open(std::string{path} + ".coefficients.csv");
    support.open(std::string{path} + ".support.csv");
    ASSERT_TRUE(support);
    ASSERT_TRUE(out);
    ASSERT_TRUE(coefficients);
  }
  if (coefficients)
    coefficients
        << std::setprecision(17)
        << "dataset,trial,identity,input_interval_seconds,factor,state_support_"
           "identity,section,section_identity,direction,index,b,a\n";
  if (out)
    out << std::setprecision(17)
        << "dataset,trial,coordinate,input_hz,folded_hz,notch_real,notch_imag,"
           "lowpass_real,combined_real,combined_imag,input_power,lowpass_only_"
           "power,combined_power,science_support_hz\n";
  if (support)
    support << std::setprecision(17)
            << "dataset,coordinate,run_index,first_row,past_last_row,begin_"
               "unix_sec,end_unix_sec,padded_samples,outside_samples,protected_"
               "samples,unknown_samples\n";
  for (const auto &[name, t] :
       std::array<std::pair<const char *, const Trial *>, 2>{
           {{"controlled", &controlled}, {"retained_case_e", &retained}}}) {
    if (support)
      for (auto coordinate : {X, R})
        for (const auto &w : t->spectral->spectrum(0, 0, coordinate).windows)
          support << name << ',' << (coordinate == X ? 'x' : 'r') << ','
                  << w.run_index << ',' << w.rows.first << ','
                  << w.rows.past_last << ',' << w.support_begin_unix_sec << ','
                  << w.support_end_unix_sec << ',' << w.padded_samples << ','
                  << w.source_counts[0] << ',' << w.source_counts[1] << ','
                  << w.source_counts[2] << '\n';
    for (double center : {0., 11., 29., 48.}) {
      auto s = spec(4);
      s.input_interval_seconds = t->spectral->network(0).interval_seconds;
      s.identity = std::string{name} + "-trial-" + std::to_string(center);
      if (center)
        s.notches = {notch(center, .5, true, 1 / s.input_interval_seconds)};
      s.science_domain = RtcTransferScienceDomain{
          "conditional-a1100-100arcsec-per-sec", RtcOpticalArray::a1100, 100};
      if (coefficients) {
        auto prefix = [&]() -> std::ostream & {
          return coefficients << name << ',' << center << ',' << s.identity
                              << ',' << s.input_interval_seconds << ','
                              << s.factor << ',' << s.state_support_identity
                              << ',';
        };
        for (std::size_t j = 0; j < s.centered_lowpass.size(); ++j)
          prefix() << "fir," << s.lowpass_identity << ",centered," << j << ','
                   << s.centered_lowpass[j] << ",1\n";
        for (const auto &n : s.notches)
          for (std::size_t j = 0; j < 3; ++j)
            prefix() << "notch," << n.identity << ",forward_reverse," << j
                     << ',' << n.b[j] << ',' << n.a[j] << '\n';
      }
      auto a = t->assess(s);
      for (const auto &c : a->coordinates()) {
        ASSERT_TRUE(c.available());
        for (const auto &b : c.bins)
          if (out)
            out << name << ',' << center << ','
                << (c.coordinate == X ? 'x' : 'r') << ',' << b.input_hz << ','
                << b.folded_output_hz << ',' << b.notch_response.real() << ','
                << b.notch_response.imag() << ',' << b.lowpass_response.real()
                << ',' << b.combined_response.real() << ','
                << b.combined_response.imag() << ',' << b.input_power << ','
                << b.lowpass_only_power << ',' << b.combined_power << ','
                << a->coordinates()[0].science->full_temporal_support_hz
                << '\n';
      }
    }
  }
  if (out.is_open()) {
    out.flush();
    coefficients.flush();
    support.flush();
    ASSERT_TRUE(support);
    ASSERT_TRUE(out);
    ASSERT_TRUE(coefficients);
  }
  for (std::size_t i = 0; i < rows.size(); ++i) {
    EXPECT_DOUBLE_EQ(retained.parent->network(0).value(X, i + 100, 0),
                     rows[i][2]);
    EXPECT_DOUBLE_EQ(retained.parent->network(0).value(R, i + 100, 0),
                     rows[i][3]);
  }
}
TEST(rtc_line_transfer, arithmetic_cost_is_reported_separately_from_learn) {
  Trial t(input());
  auto s = spec();
  s.notches = {notch(11), notch(29)};
  auto c = t.candidate(s);
  const auto start = std::chrono::steady_clock::now();
  std::size_t bins = 0;
  for (std::uint64_t i = 0; i < 100; ++i)
    bins += RtcLineTransferAssessment::consider(c, t.joint, t.val, 100 + i)
                ->coordinates()[0]
                .bins.size();
  const auto seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  EXPECT_EQ(bins, 25700);
  std::cout << "TRANSFER_TIMING calls=100 paired_coordinates=2 "
               "bins_per_coordinate=257 notches=2 fir_taps=3 seconds="
            << seconds << "\n";
}
TEST(rtc_line_transfer,
     odd_native_fft_and_insufficient_window_support_remain_explicit) {
  Input odd(2048, 1. / 127.75);
  Trial t(odd);
  auto s = spec(1);
  s.input_interval_seconds = t.spectral->network(0).interval_seconds;
  s.centered_lowpass = {1};
  auto a = t.assess(s);
  const auto &frequency = t.spectral->network(0).frequency_hz;
  ASSERT_TRUE(a->coordinates()[0].available());
  EXPECT_LT(frequency.back(), .5 / s.input_interval_seconds);
  EXPECT_EQ(a->coordinates()[0].bins.size(), frequency.size());
  for (const auto &b : a->coordinates()[0].bins) {
    EXPECT_DOUBLE_EQ(b.input_power, b.combined_power);
    EXPECT_DOUBLE_EQ(b.input_hz, b.folded_output_hz);
  }
  Trial short_record(input(512));
  auto unavailable = short_record.assess(spec());
  EXPECT_EQ(unavailable->coordinates()[0].cause,
            RtcLineTransferCause::spectral_unavailable);
  EXPECT_TRUE(unavailable->coordinates()[0].bins.empty());
  EXPECT_TRUE(
      std::isnan(unavailable->coordinates()[0].incoherent_folded_power_proxy));
}
} // namespace
