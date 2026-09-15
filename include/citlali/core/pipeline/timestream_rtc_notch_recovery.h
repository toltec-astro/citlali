#pragma once

#include <citlali/core/pipeline/ast_scan_motion_alignment.h>
#include <citlali/core/pipeline/timestream_rtc_line_transfer.h>
#include <citlali/core/pipeline/timestream_rtc_transient_exclusion.h>
#include <citlali/core/pipeline/timestream_rtc_donor_fill.h>
#include <citlali/core/timestream/rtc/filter.h>

namespace citlali::pipeline {

// Explicit experiment domain, not a production filter-bank entry or an
// automatic admission policy. The AST handle supplies actual native support.
struct RtcNotchRecoveryDomain {
  std::string identity, detector_array_association;
  std::shared_ptr<const AstScanMotionNetworkView> motion;
  RtcOpticalArray array = RtcOpticalArray::a2000;
  double speed_ceiling_arcsec_per_sec = NAN;
  double speed_margin_fraction = .05, cadence_margin_fraction = .0001;
  double nominal_interval_seconds = NAN;
  std::size_t notch_guard_samples = 0;
  bool reject = false;
};

enum class RtcNotchRecoveryCause : std::uint8_t {
  // Pair-domain disposition, separate from coordinate availability and donor
  // influence. Retained does not imply two available independent values.
  retained,
  producer_invalid,
  transient_excluded,
  motion_unavailable,
  motion_outside_domain,
  boundary_guard,
  rejection,
  below_minimum_speed,
  insufficient_output_sampling
};

// Consider freezes one complete *experimental* sequence: existing transient
// exclusions, optional explicit notch, centered FIR, fixed-phase decimation.
// No event promotion, donor invention, coefficient design or route activation.
class RtcNotchRecoveryPlan {
public:
  static std::shared_ptr<const RtcNotchRecoveryPlan>
  consider(std::shared_ptr<const RtcLineTransferAssessment> assessment,
           std::shared_ptr<const RtcTransientExclusionPlan> transients,
           std::shared_ptr<const ValSnapshot> snapshot,
           RtcNotchRecoveryDomain domain, std::uint64_t id,
           std::vector<std::shared_ptr<const RtcDonorFillPlan>> donors = {}) {
    if (!assessment || !transients || !snapshot || !id ||
        assessment->snapshot_handle().get() != snapshot.get() ||
        transients->val_snapshot_handle().get() != snapshot.get())
      throw std::invalid_argument(
          "RTC recovery requires exact assessment/transient/VAL bindings");
    const auto &candidate = *assessment->candidate_handle();
    const auto &spikes =
        candidate.line_handle()->spectral_handle()->original_spike_handle();
    if (spikes.get() != transients->screening_handle()->evidence_handle().get())
      throw std::invalid_argument(
          "RTC recovery requires the same original Learn evidence");
    // This increment implements only the explicitly approved initial VAL
    // use. A later generation needs its own bound use policy.
    if (snapshot->generation().value != 0)
      throw StaleRtcValGeneration("RTC recovery trial requires initial VAL");
    const auto &net = transients->input_handle()->network(candidate.network());
    const auto &axis = net.occurrence_axis();
    const auto &s = candidate.specification();
    std::sort(donors.begin(), donors.end(), [](const auto &a, const auto &b) {
      if (!a || !b) throw std::invalid_argument("RTC donor plan is absent");
      return a->selection().affected.first < b->selection().affected.first;
    });
    TimestreamNativeRow donor_end = axis.first_native_row();
    for (const auto &donor : donors) {
      if (!donor || donor->exclusions_handle().get() != transients.get() ||
          donor->event().network != candidate.network() ||
          donor->event().detector != candidate.detector() ||
          donor->selection().affected.first < donor_end ||
          !s.notches.empty())
        throw std::invalid_argument("RTC finite chain requires exact, disjoint donor plans for its detector");
      donor_end = donor->selection().affected.past_last;
    }
    if (domain.identity.empty() || !domain.motion ||
        domain.motion->network_timing_handle().get() !=
            axis.native_timing_handle().get() ||
        domain.motion->raw_product_handle()
                ->source_handle()
                ->admitted_detector_scope() != snapshot->scope() ||
        domain.detector_array_association !=
            net.detector(candidate.detector()).detector_association_record_id ||
        !(domain.speed_ceiling_arcsec_per_sec > 0) ||
        !std::isfinite(domain.speed_ceiling_arcsec_per_sec) ||
        !std::isfinite(domain.speed_margin_fraction) ||
        domain.speed_margin_fraction < 0 ||
        !std::isfinite(domain.cadence_margin_fraction) ||
        domain.cadence_margin_fraction < 0 ||
        !(domain.nominal_interval_seconds > 0) ||
        !std::isfinite(domain.nominal_interval_seconds))
      throw std::invalid_argument(
          "RTC recovery requires exact native AST/array/cadence domain");
    if (s.notches.size() > 1 ||
        (!s.notches.empty() &&
         (s.notches[0].direction !=
              RtcNotchResponseDirection::forward_reverse ||
          !domain.notch_guard_samples)) ||
        (s.notches.empty() && domain.notch_guard_samples))
      throw std::invalid_argument("RTC recovery trial supports one explicit "
                                  "forward/reverse notch and guard");
    if (domain.notch_guard_samples >
        static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()) -
            s.centered_lowpass.size() / 2)
      throw std::invalid_argument(
          "RTC recovery guard exceeds representable native support");
    const auto finite_half = s.centered_notch.size() / 2;
    if (finite_half >
        static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()) -
            s.centered_lowpass.size() / 2)
      throw std::invalid_argument(
          "RTC finite notch support exceeds index range");
    const auto optical =
        rtc_optical_scale(domain.array, domain.speed_ceiling_arcsec_per_sec);
    if (optical.temporal_support_hz >=
        .5 / (s.input_interval_seconds * s.factor) *
            (1 - domain.cadence_margin_fraction))
      throw std::invalid_argument(
          "RTC recovery output cadence cannot contain its optical domain");
    double dc = 0;
    for (double h : s.centered_lowpass)
      dc += h;
    if (std::abs(dc - 1) > 1e-12)
      throw std::invalid_argument("RTC recovery centered FIR must preserve DC");
    if (!s.centered_notch.empty()) {
      double notch_dc = 0;
      for (double h : s.centered_notch)
        notch_dc += h;
      if (std::abs(notch_dc - 1) > 1e-12)
        throw std::invalid_argument("RTC finite notch must preserve DC");
      if ((finite_half + s.centered_lowpass.size() / 2) *
              s.input_interval_seconds >
          5.)
        throw std::invalid_argument(
            "RTC finite chain exceeds five-second half-support");
    }
    auto out = std::shared_ptr<RtcNotchRecoveryPlan>(new RtcNotchRecoveryPlan);
    out->assessment_ = std::move(assessment);
    out->transients_ = std::move(transients);
    out->donors_ = std::move(donors);
    out->domain_ = std::move(domain);
    out->id_ = id;
    out->sampling_speed_limit_ =
        optical.airy_fwhm_arcsec *
        ((1 - out->domain_.cadence_margin_fraction) /
         out->domain_.nominal_interval_seconds) /
        (4 * s.factor * (1 + out->domain_.speed_margin_fraction));
    out->first_ = axis.first_native_row();
    out->causes_.resize(axis.occurrence_count(),
                        RtcNotchRecoveryCause::retained);
    const auto &d = out->domain_;
    for (const auto &run : axis.contiguous_runs()) {
      TimestreamNativeRow start = run.first_native_row;
      for (auto row = run.first_native_row; row < run.past_last_native_row;
           ++row) {
        auto &cause = out->causes_[row - out->first_];
        if (!net.state(NativeReadoutCoordinate::x, row, candidate.detector())
                 .valid() ||
            !net.state(NativeReadoutCoordinate::r, row, candidate.detector())
                 .valid())
          cause = RtcNotchRecoveryCause::producer_invalid;
        else {
          for (auto c :
               {NativeReadoutCoordinate::x, NativeReadoutCoordinate::r})
            if (!std::isfinite(net.value(c, row, candidate.detector())))
              throw std::invalid_argument("RTC recovery unexpected nonfinite "
                                          "in admitted original support");
          if (out->transients_->excludes(candidate.network(), row,
                                         candidate.detector())) {
            cause = RtcNotchRecoveryCause::transient_excluded;
          }
          const auto v = d.motion->scalar_speed_arcsec_per_sec(row);
          if (cause !=
              RtcNotchRecoveryCause::retained) { /* retain the upstream cause */
          } else if (!v)
            cause = RtcNotchRecoveryCause::motion_unavailable;
          else if (!ast_scan_motion_speed_admitted(*v))
            cause = RtcNotchRecoveryCause::below_minimum_speed;
          else if (*v > out->sampling_speed_limit_)
            cause = RtcNotchRecoveryCause::insufficient_output_sampling;
          else if (*v * (1 + d.speed_margin_fraction) >
                   d.speed_ceiling_arcsec_per_sec)
            cause = RtcNotchRecoveryCause::motion_outside_domain;
        }
        if (row > run.first_native_row) {
          const auto t =
              axis.native_identity(row).reconstructed_time_unix_sec();
          const auto prev =
              axis.native_identity(row - 1).reconstructed_time_unix_sec();
          const auto roundoff = 4 * (std::nextafter(t, INFINITY) - t);
          if (std::abs((t - prev) - d.nominal_interval_seconds) >
              d.nominal_interval_seconds * d.cadence_margin_fraction + roundoff)
            throw std::invalid_argument(
                "RTC recovery native cadence outside explicit domain");
        }
        if (cause != RtcNotchRecoveryCause::retained) {
          if (start < row)
            out->runs_.push_back({start, row});
          start = row + 1;
        }
      }
      if (start < run.past_last_native_row)
        out->runs_.push_back({start, run.past_last_native_row});
    }
    return out;
  }
  const auto &assessment_handle() const noexcept { return assessment_; }
  const auto &transient_handle() const noexcept { return transients_; }
  const auto &input_handle() const noexcept {
    return transients_->input_handle();
  }
  const auto &snapshot_handle() const noexcept {
    return transients_->val_snapshot_handle();
  }
  const auto &domain() const noexcept { return domain_; }
  const auto &donor_plans() const noexcept { return donors_; }
  const auto &runs() const noexcept { return runs_; }
  const auto &input_causes() const noexcept { return causes_; }
  auto first_native_row() const noexcept { return first_; }
  auto consideration() const noexcept { return id_; }
  std::size_t full_support_half_samples() const noexcept {
    const auto &s = assessment_->candidate_handle()->specification();
    return s.centered_lowpass.size() / 2 + s.centered_notch.size() / 2 +
           domain_.notch_guard_samples;
  }
  // Inexpensive Consider accounting on the exact already resolved domain.
  // An IIR guard is not finite impulse support and cannot use this product.
  std::vector<RtcEventRange> finite_retained_runs() const {
    if (!assessment_->candidate_handle()->specification().notches.empty())
      throw std::invalid_argument("RTC IIR has no finite support accounting");
    std::vector<RtcEventRange> out;
    if (domain_.reject)
      return out;
    const auto half =
        static_cast<TimestreamNativeRow>(full_support_half_samples());
    for (const auto &run : runs_)
      if (run.past_last - run.first > 2 * half)
        out.push_back({run.first + half, run.past_last - half});
    return out;
  }
  double sampling_speed_limit_arcsec_per_sec() const noexcept {
    return sampling_speed_limit_;
  }
  // The unchanged experimental IIR has whole-run dependence. A finite
  // endpoint guard is not proof of a five-second finite source footprint.
  bool finite_five_second_footprint() const noexcept {
    const auto &s = assessment_->candidate_handle()->specification();
    const auto half = static_cast<TimestreamNativeRow>(s.centered_lowpass.size()/2+s.centered_notch.size()/2);
    if (!s.notches.empty() || half*s.input_interval_seconds > 5.) return false;
    // Frozen donor values retain numerical dependence on their original local
    // background and donor samples. Short filter taps do not erase that reach.
    const auto &axis=input_handle()->network(assessment_->candidate_handle()->network()).occurrence_axis();
    for(const auto &d:donors_) if(d->cause()==RtcDonorFillCause::ready) {
      auto first=d->donor_support().first,last=d->donor_support().past_last-1;
      for(const auto &side:d->event().background[0].support) if(side.usable) {
        first=std::min(first,side.first_used);last=std::max(last,side.last_used);
      }
      const auto affected=d->selection().affected;
      for(const auto &run:runs_) {
        const auto begin=std::max(run.first+half,affected.first-half);
        const auto end=std::min(run.past_last-half-1,affected.past_last-1+half);
        if(begin>end)continue;
        const auto time=[&](auto row){return axis.native_identity(row).reconstructed_time_unix_sec();};
        if(time(last)-time(begin)>5. || time(end)-time(first)>5.)return false;
      }
    }
    return true;
  }
  static constexpr bool production_authorized = false;

private:
  RtcNotchRecoveryPlan() = default;
  std::shared_ptr<const RtcLineTransferAssessment> assessment_;
  std::shared_ptr<const RtcTransientExclusionPlan> transients_;
  std::vector<std::shared_ptr<const RtcDonorFillPlan>> donors_;
  RtcNotchRecoveryDomain domain_;
  std::vector<RtcNotchRecoveryCause> causes_;
  std::vector<RtcEventRange> runs_;
  TimestreamNativeRow first_ = 0;
  std::uint64_t id_ = 0;
  double sampling_speed_limit_ = NAN;
};

// A named diagnostic overlay; it never masquerades as original measurements.
// Binding to the plan keeps coefficients, masks, timing and boundaries fixed
// between each injected/uninjected pair.
struct RtcRecoveryInjection {
  std::shared_ptr<const RtcNotchRecoveryPlan> plan;
  std::string identity;
  Eigen::Matrix<double, Eigen::Dynamic, 2> delta;
};

class RtcNotchRecoveryResult {
public:
  static std::shared_ptr<const RtcNotchRecoveryResult>
  apply(std::shared_ptr<const RtcNotchRecoveryPlan> plan,
        std::shared_ptr<const NativePairedReadoutView> original,
        std::shared_ptr<const ValSnapshot> snapshot,
        std::span<const std::shared_ptr<const NativePairedReadoutView>>
            partitions,
        const RtcRecoveryInjection *injection = nullptr) {
    if (!plan || !original || plan->input_handle().get() != original.get())
      throw std::invalid_argument("RTC recovery Apply requires exact original "
                                  "input; no cumulative replay");
    auto retained = RtcTransientExclusionResult::apply(
        plan->transient_handle(), original, snapshot, partitions);
    const auto n = plan->input_causes().size();
    if (injection &&
        (injection->plan.get() != plan.get() || injection->identity.empty() ||
         injection->delta.rows() != static_cast<Eigen::Index>(n) ||
         !injection->delta.allFinite()))
      throw std::invalid_argument("RTC recovery injection requires exact "
                                  "frozen plan and finite paired shape");
    auto out =
        std::shared_ptr<RtcNotchRecoveryResult>(new RtcNotchRecoveryResult);
    out->plan_ = std::move(plan);
    out->retained_ = std::move(retained);
    out->injection_identity_ = injection ? injection->identity : "none";
    // A donor plan freezes its original-data fit/median. Injecting into that
    // already selected support would require an explicit donor-response model.
    if (injection && !out->plan_->donor_plans().empty())
      throw std::invalid_argument("RTC paired injection with donors requires separately bound donor response");
    for (const auto &donor : out->plan_->donor_plans())
      out->donors_.push_back(RtcDonorFillResult::apply(donor, original, snapshot, partitions));
    out->causes_ = out->plan_->input_causes();
    out->native_state_.assign(n, 0);
    out->conditioned_.resize(n, 2);
    out->conditioned_.setConstant(NAN);
    out->filtered_.resize(n, 2);
    out->filtered_.setConstant(NAN);
    const auto &candidate =
        *out->plan_->assessment_handle()->candidate_handle();
    const auto &s = candidate.specification();
    const auto first = out->plan_->first_native_row();
    const auto &net = original->network(candidate.network());
    const auto half = s.centered_lowpass.size() / 2;
    const auto notch_half = s.centered_notch.size() / 2;
    const auto guard = out->plan_->full_support_half_samples();
    for (auto run : out->plan_->runs()) {
      const auto size = run.past_last - run.first;
      Eigen::MatrixXd values(size, 2);
      std::vector<std::uint8_t> influence(size, 0);
      for (Eigen::Index i = 0; i < size; ++i)
        for (int c = 0; c < 2; ++c) {
          const auto row = run.first + i;
          values(i, c) = net.value(static_cast<NativeReadoutCoordinate>(c), row,
                                   candidate.detector());
          if (injection)
            values(i, c) += injection->delta(row - first, c);
        }
      if (!values.allFinite())
        throw std::overflow_error(
            "RTC recovery original plus injection overflow");
      for (const auto &donor : out->donors_) {
        const auto affected = donor->plan_handle()->selection().affected;
        for (auto row = std::max(run.first, affected.first);
             row < std::min(run.past_last, affected.past_last); ++row) {
          values(row - run.first, 0) = donor->value_for_conditioning(
              NativeReadoutCoordinate::x, candidate.network(), row,
              candidate.detector()).value_or(NAN);
          values(row - run.first, 1) = NAN; // accepted x-only donor exception
          influence[row-run.first] = donor->filled() ? 1 : 2;
        }
      }
      if (!s.notches.empty()) {
        // Reuse the mature finite kernel unchanged: odd reflection of
        // min(9,N-1), constant endpoint initialization, then reverse.
        timestream::Filter filter;
        filter.notch_zero_phase = true;
        const auto &section = s.notches[0];
        filter.notch_a.emplace_back(
            Eigen::Map<const Eigen::Vector3d>(section.a.data()));
        filter.notch_b.emplace_back(
            Eigen::Map<const Eigen::Vector3d>(section.b.data()));
        filter.iir(values);
      }
      if (out->donors_.empty() && !values.allFinite())
        throw std::overflow_error(
            "RTC recovery finite notch arithmetic failed");
      if (!s.centered_notch.empty()) {
        // Both centered stages use only complete real input support. The
        // first-stage edges are unavailable, never padded or renormalized.
        Eigen::MatrixXd finite = Eigen::MatrixXd::Constant(size, 2, NAN);
        std::vector<std::uint8_t> finite_influence(size, 0);
        for (Eigen::Index i = notch_half;
             i + static_cast<Eigen::Index>(notch_half) < size; ++i)
          for (int c = 0; c < 2; ++c) {
            double v = 0;
            bool unavailable = false;
            for (std::size_t j = 0; j < s.centered_notch.size(); ++j) {
              if (!out->donors_.empty() && s.centered_notch[j] == 0) continue;
              unavailable |= !std::isfinite(values(i+j-notch_half,c));
              v = std::fma(s.centered_notch[j], values(i + j - notch_half, c),
                           v);
              if (s.centered_notch[j] != 0) finite_influence[i] |= influence[i+j-notch_half];
            }
            if (!unavailable && !std::isfinite(v))
              throw std::overflow_error("RTC finite notch arithmetic failed");
            finite(i, c) = std::isfinite(v) ? v : NAN;
          }
        values = std::move(finite);
        influence = std::move(finite_influence);
      }
      out->conditioned_.middleRows(run.first - first, size) = values;
      for (Eigen::Index i = 0; i < size; ++i) {
        const auto row = run.first + i, local = row - first;
        const auto native_guard = notch_half + out->plan_->domain().notch_guard_samples;
        if (!out->plan_->domain().reject && i >= static_cast<Eigen::Index>(native_guard) &&
            i+static_cast<Eigen::Index>(native_guard) < size) {
          for (int c = 0; c < 2; ++c)
            if (std::isfinite(values(i,c))) out->native_state_[local] |= 1U << c;
          if (influence[i] & 1U) out->native_state_[local] |= 4U;
          if (influence[i] & 2U) out->native_state_[local] |= 64U;
        }
        if (out->plan_->domain().reject)
          out->causes_[local] = RtcNotchRecoveryCause::rejection;
        else if (i < static_cast<Eigen::Index>(guard) ||
                 i + static_cast<Eigen::Index>(guard) >= size)
          out->causes_[local] = RtcNotchRecoveryCause::boundary_guard;
        else {
          for (int c = 0; c < 2; ++c) {
            double v = 0;
            bool unavailable = false;
            std::uint8_t influenced = 0;
            for (std::size_t j = 0; j < s.centered_lowpass.size(); ++j) {
              if (!out->donors_.empty() && s.centered_lowpass[j] == 0) continue;
              unavailable |= !std::isfinite(values(i+j-half,c));
              v = std::fma(s.centered_lowpass[j], values(i + j - half, c), v);
              if (s.centered_lowpass[j] != 0) influenced |= influence[i+j-half];
            }
            out->filtered_(local, c) = std::isfinite(v) ? v : NAN;
            if (!unavailable && !std::isfinite(v))
              throw std::overflow_error("RTC recovery FIR arithmetic failed");
            if (!unavailable) out->native_state_[local] |= 8U << c;
            if (influenced & 1U) out->native_state_[local] |= 32U;
            if (influenced & 2U) out->native_state_[local] |= 128U;
          }
          if (local % s.factor == 0)
            out->output_rows_.push_back(row);
        }
      }
    }
    return out;
  }
  const auto &plan_handle() const noexcept { return plan_; }
  const auto &donor_results() const noexcept { return donors_; }
  // Support facts, not downstream eligibility. Nonrepresentative donor
  // influence never silently becomes universal rejection (SCI-RTC-REQ-020).
  bool representative_replaced(TimestreamNativeRow row) const {
    (void)native_state_.at(row-plan_->first_native_row());
    for (const auto &d : donors_) {
      const auto r = d->plan_handle()->selection().affected;
      if (d->filled() && row >= r.first && row < r.past_last) return true;
    }
    return false;
  }
  bool requires_representative_exclusion(TimestreamNativeRow row) const {
    (void)native_state_.at(row-plan_->first_native_row());
    for (const auto &d : donors_) {
      const auto r=d->plan_handle()->selection().affected;
      if (row >= r.first && row < r.past_last) return true;
    }
    return false;
  }
  bool replacement_influence(TimestreamNativeRow row, bool after_lowpass) const {
    return (native_state_.at(row-plan_->first_native_row()) & (after_lowpass ? 32U : 4U)) != 0;
  }
  bool unrepaired_influence(TimestreamNativeRow row, bool after_lowpass) const {
    return (native_state_.at(row-plan_->first_native_row()) & (after_lowpass ? 128U : 64U)) != 0;
  }
  bool native_stage_admitted(TimestreamNativeRow row, bool after_lowpass) const {
    const auto local = row - plan_->first_native_row();
    if (local < 0 || static_cast<std::size_t>(local) >= causes_.size())
      throw std::out_of_range("RTC native stage row outside original support");
    if (after_lowpass) return causes_[local] == RtcNotchRecoveryCause::retained;
    if (plan_->domain().reject) return false;
    const auto &s = plan_->assessment_handle()->candidate_handle()->specification();
    const auto half = static_cast<TimestreamNativeRow>(s.centered_notch.size()/2 + plan_->domain().notch_guard_samples);
    auto run = std::upper_bound(plan_->runs().begin(),plan_->runs().end(),row,
        [](auto r, const auto &span) { return r < span.first; });
    if (run != plan_->runs().begin()) {
      --run;
      if (row >= run->first+half && row < run->past_last-half) return true;
    }
    return false;
  }
  bool coordinate_stage_available(NativeReadoutCoordinate c, TimestreamNativeRow row,
                                  bool after_lowpass) const {
    if (c != NativeReadoutCoordinate::x && c != NativeReadoutCoordinate::r)
      throw std::invalid_argument("RTC conditioned stage requires x or r");
    const auto shift = static_cast<unsigned>(c) + (after_lowpass ? 3 : 0);
    return (native_state_.at(row-plan_->first_native_row()) & (1U << shift)) != 0;
  }
  const auto &injection_identity() const noexcept {
    return injection_identity_;
  }
  const auto &conditioned_native_pair() const noexcept { return conditioned_; }
  const auto &filtered_native_pair() const noexcept { return filtered_; }
  const auto &causes() const noexcept { return causes_; }
  const auto &output_native_rows() const noexcept { return output_rows_; }
  static constexpr bool independent_measurements = false,
                        map_input_authorized = false;

private:
  RtcNotchRecoveryResult() = default;
  std::shared_ptr<const RtcNotchRecoveryPlan> plan_;
  std::shared_ptr<const RtcTransientExclusionResult> retained_;
  std::vector<std::shared_ptr<const RtcDonorFillResult>> donors_;
  Eigen::Matrix<double, Eigen::Dynamic, 2> conditioned_, filtered_;
  std::vector<RtcNotchRecoveryCause> causes_;
  std::vector<std::uint8_t> native_state_;
  std::vector<TimestreamNativeRow> output_rows_;
  std::string injection_identity_;
};

} // namespace citlali::pipeline
