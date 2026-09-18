#pragma once

#include <citlali/core/pipeline/timestream_rtc_native_spectral_learn.h>
#include <optional>

namespace citlali::pipeline {

enum class RtcTreatmentOutcomeCause {
  available,
  input_evidence_unavailable,
  matched_evidence_unavailable,
  unequal_realized_support,
  arithmetic_nonfinite
};

// Coordinate-local, descriptive powers in native-coordinate squared units.
// These are integrals of the stored D2 estimator, including its inherited DC,
// odd-grid and padding conventions; not excess-line power or independent noise.
struct RtcTreatmentOutcomePower {
  double original = NAN, conditioned = NAN;
  std::optional<double> conditioned_over_original;
};
struct RtcTreatmentOutcomeRecord {
  TimestreamNetworkId network;
  std::uint32_t detector;
  NativeReadoutCoordinate coordinate;
  RtcTreatmentOutcomeCause cause =
      RtcTreatmentOutcomeCause::input_evidence_unavailable;
  std::size_t original_eligible_samples = 0, conditioned_eligible_samples = 0;
  std::size_t common_eligible_samples = 0, window_union_samples = 0;
  double window_union_seconds = 0;
  // Original and conditioned populations remain separately retained in the
  // parents. This intersection is a comparison use, not a new processing mask.
  std::vector<RtcEventRange> common_support, window_union;
  RtcNativeSpectrum original_matched, conditioned_matched;
  RtcTreatmentOutcomePower power;
  bool available() const noexcept {
    return cause == RtcTreatmentOutcomeCause::available;
  }
};

class RtcTreatmentOutcomeEvidence {
public:
  static std::shared_ptr<const RtcTreatmentOutcomeEvidence>
  learn(std::shared_ptr<const RtcNativeSpectralEvidence> original,
        std::shared_ptr<const RtcNativeSpectralEvidence> conditioned,
        std::uint64_t attempt) {
    if (!original || !conditioned || !attempt ||
        original->conditioned_handle() || !conditioned->conditioned_handle() ||
        original->original_spike_handle().get() !=
            conditioned->original_spike_handle().get())
      throw std::invalid_argument(
          "RTC treatment outcome requires exact original and numerical "
          "conditioned evidence");
    auto out = std::shared_ptr<RtcTreatmentOutcomeEvidence>(
        new RtcTreatmentOutcomeEvidence);
    out->original_ = std::move(original);
    out->conditioned_ = std::move(conditioned);
    out->attempt_ = attempt;
    // Reuse the numerical owner. These private workers cannot escape as a new
    // initial or conditioned reference under its broader use policy.
    RtcNativeSpectralEvidence before{out->original_->original_spike_handle(),
                                     attempt};
    RtcNativeSpectralEvidence after{out->original_->original_spike_handle(),
                                    attempt};
    after.conditioned_ = out->conditioned_->conditioned_handle();
    for (const auto &n : out->original_->networks()) {
      const auto id = n.input->support().network_id;
      const auto &m = out->conditioned_->network(id);
      if (n.input->support() != m.input->support() ||
          n.cadence_domain.authority != m.cadence_domain.authority ||
          n.cadence_domain.nominal_interval_seconds !=
              m.cadence_domain.nominal_interval_seconds ||
          n.cadence_domain.maximum_fractional_deviation !=
              m.cadence_domain.maximum_fractional_deviation ||
          n.frequency_hz != m.frequency_hz || n.fft_samples != m.fft_samples)
        throw std::invalid_argument("RTC treatment comparison cannot mix "
                                    "cadence/profile/support bindings");
      std::vector<double> window(n.frequency_hz.empty() ? 0 : n.fft_samples);
      for (std::size_t i = 0; i < window.size(); ++i)
        window[i] =
            .5 - .5 * std::cos(2 * std::numbers::pi * static_cast<double>(i) /
                               (window.size() - 1));
      const auto &net =
          out->original_->original_spike_handle()->input_handle()->network(id);
      for (std::uint32_t d = 0;
           d < static_cast<std::uint32_t>(net.detector_count()); ++d)
        for (auto c :
             {NativeReadoutCoordinate::x, NativeReadoutCoordinate::r}) {
          const auto &a = out->original_->spectrum(id, d, c),
                     &b = out->conditioned_->spectrum(id, d, c);
          RtcTreatmentOutcomeRecord record;
          record.network = id;
          record.detector = d;
          record.coordinate = c;
          record.original_eligible_samples = count(a.centering_support);
          record.conditioned_eligible_samples = count(b.centering_support);
          record.common_support =
              intersect(a.centering_support, b.centering_support);
          record.common_eligible_samples = count(record.common_support);
          const auto same_population=[&](const auto &support) {
            return support.size()==record.common_support.size() &&
                std::equal(support.begin(),support.end(),record.common_support.begin(),
                    [](auto x,auto y){return x.first==y.first && x.past_last==y.past_last;});
          };
          // Exact same input/profile/VAL and complete centering population:
          // reuse its already measured spectrum, including nonfinite/run
          // dispositions. No evidence from a different support is substituted.
          record.original_matched = same_population(a.centering_support) ? a :
              before.measure(n, window, d, c, &record.common_support);
          record.conditioned_matched = same_population(b.centering_support) ? b :
              after.measure(m, window, d, c, &record.common_support);
          if (a.available() && b.available()) {
            const auto &x = record.original_matched,
                       &y = record.conditioned_matched;
            if (!x.available() || !y.available())
              record.cause =
                  RtcTreatmentOutcomeCause::matched_evidence_unavailable;
            else if (!same_support(x, y))
              record.cause = RtcTreatmentOutcomeCause::unequal_realized_support;
            else {
              record.cause = RtcTreatmentOutcomeCause::available;
              std::size_t previous_run = x.windows.front().run_index;
              for (const auto &w : x.windows) {
                if (w.run_index == previous_run && !record.window_union.empty() &&
                    w.rows.first <= record.window_union.back().past_last)
                  record.window_union.back().past_last = std::max(
                      record.window_union.back().past_last, w.rows.past_last);
                else
                  record.window_union.push_back(w.rows);
                previous_run = w.run_index;
              }
              record.window_union_samples = count(record.window_union);
              // Sum actual native integration support once per occurrence;
              // neither overlapping Welch windows nor gaps count as exposure.
              for (auto span : record.window_union)
                for (auto row = span.first; row < span.past_last; ++row) {
                  const auto &support =
                      net.occurrence_axis().occurrence(row).integration_support;
                  record.window_union_seconds +=
                      support.end_unix_sec - support.begin_unix_sec;
                }
              record.power = integrate(x, y, n, 0, n.frequency_hz.size());
              if (!std::isfinite(record.power.original) ||
                  !std::isfinite(record.power.conditioned))
                record.cause = RtcTreatmentOutcomeCause::arithmetic_nonfinite;
            }
          }
          out->records_.push_back(std::move(record));
        }
    }
    out->peak_scratch_ =
        std::max(before.peak_scratch_samples(), after.peak_scratch_samples());
    return out;
  }
  const auto &original_handle() const noexcept { return original_; }
  const auto &conditioned_handle() const noexcept { return conditioned_; }
  const auto &records() const noexcept { return records_; }
  auto attempt() const noexcept { return attempt_; }
  const RtcTreatmentOutcomeRecord &record(TimestreamNetworkId n,
                                          std::uint32_t d,
                                          NativeReadoutCoordinate c) const {
    auto it =
        std::find_if(records_.begin(), records_.end(), [=](const auto &r) {
          return r.network == n && r.detector == d && r.coordinate == c;
        });
    if (it == records_.end())
      throw std::out_of_range("RTC treatment outcome absent");
    return *it;
  }
  // Explicit half-open stored-bin interval; no automatic band, line ranking,
  // background estimator or admission threshold. Absent ratios are never zero.
  RtcTreatmentOutcomePower band(TimestreamNetworkId n, std::uint32_t d,
                                NativeReadoutCoordinate c, std::size_t first,
                                std::size_t past_last) const {
    const auto &r = record(n, d, c);
    const auto &grid = original_->network(n);
    if (first >= past_last || past_last > grid.frequency_hz.size())
      throw std::invalid_argument("invalid outcome bin range");
    return r.available() ? integrate(r.original_matched, r.conditioned_matched,
                                     grid, first, past_last)
                         : RtcTreatmentOutcomePower{};
  }
  std::size_t logical_owned_bytes() const noexcept {
    std::size_t bytes = records_.size() * sizeof(RtcTreatmentOutcomeRecord);
    for (const auto &r : records_) {
      bytes += (r.common_support.size() + r.window_union.size()) *
               sizeof(RtcEventRange);
      for (const auto *s : {&r.original_matched, &r.conditioned_matched})
        bytes += s->psd.size() * sizeof(double) +
                 s->windows.size() * sizeof(RtcSpectralWindow) +
                 s->runs.size() * sizeof(RtcSpectralRun) +
                 s->centering_support.size() * sizeof(RtcEventRange);
    }
    return bytes;
  }
  auto peak_scratch_samples() const noexcept { return peak_scratch_; }
  static constexpr const char *use_policy =
      "rtc-matched-native-treatment-outcome-v1";
  static constexpr bool classification_authorized = false,
                        stopping_rule_selected = false,
                        independent_noise_estimate = false;

private:
  RtcTreatmentOutcomeEvidence() = default;
  static std::size_t count(const std::vector<RtcEventRange> &spans) {
    std::size_t n = 0;
    for (auto r : spans)
      n += r.past_last - r.first;
    return n;
  }
  static std::vector<RtcEventRange>
  intersect(const std::vector<RtcEventRange> &a,
            const std::vector<RtcEventRange> &b) {
    std::vector<RtcEventRange> out;
    std::size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
      const auto first = std::max(a[i].first, b[j].first),
                 last = std::min(a[i].past_last, b[j].past_last);
      if (first < last)
        out.push_back({first, last});
      if (a[i].past_last < b[j].past_last)
        ++i;
      else
        ++j;
    }
    return out;
  }
  static bool same_support(const RtcNativeSpectrum &a,
                           const RtcNativeSpectrum &b) {
    if (a.centering_support.size() != b.centering_support.size() ||
        a.windows.size() != b.windows.size())
      return false;
    for (std::size_t i = 0; i < a.centering_support.size(); ++i)
      if (a.centering_support[i].first != b.centering_support[i].first ||
          a.centering_support[i].past_last != b.centering_support[i].past_last)
        return false;
    for (std::size_t i = 0; i < a.windows.size(); ++i) {
      const auto &x = a.windows[i], &y = b.windows[i];
      if (x.rows.first != y.rows.first ||
          x.rows.past_last != y.rows.past_last || x.run_index != y.run_index ||
          x.padded_samples != y.padded_samples ||
          x.support_begin_unix_sec != y.support_begin_unix_sec ||
          x.support_end_unix_sec != y.support_end_unix_sec)
        return false;
    }
    return true;
  }
  static RtcTreatmentOutcomePower
  integrate(const RtcNativeSpectrum &a, const RtcNativeSpectrum &b,
            const RtcSpectralNetwork &n, std::size_t first, std::size_t last) {
    RtcTreatmentOutcomePower p;
    p.original = 0;
    p.conditioned = 0;
    const double df = 1 / (n.fft_samples * n.interval_seconds);
    for (auto k = first; k < last; ++k) {
      p.original += a.psd.at(k) * df;
      p.conditioned += b.psd.at(k) * df;
    }
    if (std::isfinite(p.original) && std::isfinite(p.conditioned) &&
        p.original > 0) {
      const double ratio = p.conditioned / p.original;
      if (std::isfinite(ratio))
        p.conditioned_over_original = ratio;
    }
    return p;
  }
  std::shared_ptr<const RtcNativeSpectralEvidence> original_, conditioned_;
  std::vector<RtcTreatmentOutcomeRecord> records_;
  std::uint64_t attempt_ = 0;
  std::size_t peak_scratch_ = 0;
};

} // namespace citlali::pipeline
