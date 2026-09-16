#include <citlali/core/pipeline/timestream_rtc_common_mode.h>
#include <set>

namespace citlali::pipeline {
namespace {
double median(std::vector<double> v) {
  return v.empty() ? NAN : rtc_spike_detail::median(v);
}
double mad(const std::vector<double> &v) {
  const double center = median(v);
  std::vector<double> deviations;
  deviations.reserve(v.size());
  for (double x : v)
    deviations.push_back(std::abs(x - center));
  return RtcCommonModePolicy::mad_scale * median(std::move(deviations));
}
RtcCommonModeFit fit(const NativePairedReadoutNetwork &net,
                     std::span<const double> reference, RtcCommonModeFit out) {
  const auto first = net.occurrence_axis().first_native_row();
  const auto n = out.rows.past_last - out.rows.first;
  if (n <
      static_cast<TimestreamNativeRow>(RtcCommonModePolicy::minimum_samples))
    return out;
  std::vector<double> c, y, w(n, 1), residual(n);
  c.reserve(n);
  y.reserve(n);
  for (auto row = out.rows.first; row < out.rows.past_last; ++row) {
    c.push_back(reference[row - first]);
    y.push_back(net.value(NativeReadoutCoordinate::x, row, out.detector));
    if (!std::isfinite(c.back()))
      return out;
    if (!std::isfinite(y.back())) {
      out.cause = RtcCommonModeCause::input_nonfinite;
      return out;
    }
  }
  const double cm = median(c), ym = median(y), cs = mad(c), ys = mad(y);
  out.reference_scatter = cs;
  if (!(cs > 0) || !std::isfinite(cs)) {
    out.cause = RtcCommonModeCause::weak_reference;
    return out;
  }
  // Center/scale reference for stable two-parameter weighted affine solves.
  for (double &x : c)
    x = (x - cm) / cs;
  for (double &x : y)
    x -= ym;
  double a = 0, b = 0;
  bool converged = false;
  for (std::size_t iter = 0; iter < RtcCommonModePolicy::maximum_iterations;
       ++iter) {
    long double sw = 0, sx = 0, sy = 0;
    for (std::size_t i = 0; i < c.size(); ++i) {
      sw += w[i];
      sx += w[i] * c[i];
      sy += w[i] * y[i];
    }
    const double cx = sx / sw, cy = sy / sw;
    long double xx = 0, xy = 0;
    for (std::size_t i = 0; i < c.size(); ++i) {
      xx += w[i] * (c[i] - cx) * (c[i] - cx);
      xy += w[i] * (c[i] - cx) * (y[i] - cy);
    }
    if (!(xx > 0) || !std::isfinite(static_cast<double>(xx)))
      break;
    const double next_a = xy / xx, next_b = cy - next_a * cx;
    const double change = std::max(std::abs(next_a - a), std::abs(next_b - b));
    a = next_a;
    b = next_b;
    out.iterations = iter + 1;
    for (std::size_t i = 0; i < c.size(); ++i)
      residual[i] = y[i] - a * c[i] - b;
    const double scale = mad(residual);
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(scale))
      break;
    const double roundoff = 64 * std::numeric_limits<double>::epsilon() *
                            std::max({std::abs(ym), ys, std::abs(a),
                                      std::numeric_limits<double>::min()});
    if (scale <= roundoff ||
        (iter &&
         change <= RtcCommonModePolicy::tolerance *
                       std::max({ys, std::abs(a), std::abs(b), roundoff}))) {
      converged = true;
      break;
    }
    for (std::size_t i = 0; i < c.size(); ++i)
      w[i] = std::min(1., RtcCommonModePolicy::huber * scale /
                              std::max(std::abs(residual[i]), roundoff));
  }
  if (!converged) {
    out.cause = RtcCommonModeCause::fit_failed;
    return out;
  }
  out.gain = a / cs;
  out.offset = ym + b - out.gain * cm;
  out.residual_scatter = mad(residual);
  // Descriptive Pearson correlation, without independent-sample errors.
  const double cx = std::accumulate(c.begin(), c.end(), 0.) / n,
               cy = std::accumulate(y.begin(), y.end(), 0.) / n;
  long double xx = 0, yy = 0, xy = 0;
  for (std::size_t i = 0; i < c.size(); ++i) {
    xx += (c[i] - cx) * (c[i] - cx);
    yy += (y[i] - cy) * (y[i] - cy);
    xy += (c[i] - cx) * (y[i] - cy);
  }
  if (xx > 0 && yy > 0)
    out.correlation =
        std::clamp(static_cast<double>(xy / std::sqrt(xx * yy)), -1., 1.);
  if (!std::isfinite(out.gain) || !std::isfinite(out.offset) ||
      !std::isfinite(out.residual_scatter)) {
    out.cause = RtcCommonModeCause::fit_failed;
    return out;
  }
  out.cause = RtcCommonModeCause::available;
  return out;
}
} // namespace

std::shared_ptr<const RtcCommonModeEvidence>
RtcCommonModeEvidence::learn(std::shared_ptr<const RtcSpikeEvidence> original,
                             RtcCommonModeDomain domain,
                             std::uint64_t attempt) {
  if (!original || !attempt || !domain.scans || !domain.motion ||
      domain.population_authority.empty())
    throw std::invalid_argument("RTC common mode requires original screening "
                                "and exact minimum boundary authorities");
  const auto parent = original->input_handle()->parent_handle();
  const auto &net = parent->network(domain.network);
  const auto &axis = net.occurrence_axis();
  if (original->val_snapshot_handle()->generation().value != 0 ||
      domain.scans->parent_handle().get() != parent.get() ||
      domain.motion->network_timing_handle().get() !=
          axis.native_timing_handle().get() ||
      domain.motion->raw_product_handle()
              ->source_handle()
              ->admitted_detector_scope() != parent->scope() ||
      domain.members.size() != static_cast<std::size_t>(net.detector_count()))
    throw std::invalid_argument(
        "RTC common mode requires exact original parent, initial VAL, timing "
        "and detector population");
  for (std::size_t d = 0; d < domain.members.size(); ++d)
    if (domain.members[d].detector_occurrence !=
        net.detectors()[d].detector_occurrence_id)
      throw std::invalid_argument(
          "RTC common mode population occurrence mismatch");
  if (!std::isfinite(domain.nominal_interval_seconds) ||
      domain.nominal_interval_seconds <= 0 || !domain.output_factor ||
      !std::isfinite(domain.speed_ceiling_arcsec_per_sec) ||
      domain.speed_ceiling_arcsec_per_sec <= 0 ||
      !std::isfinite(domain.speed_margin_fraction) ||
      domain.speed_margin_fraction < 0 ||
      !std::isfinite(domain.cadence_margin_fraction) ||
      domain.cadence_margin_fraction < 0 || domain.cadence_margin_fraction >= 1)
    throw std::invalid_argument(
        "RTC common mode requires existing finite optical/cadence domain");
  auto out = std::shared_ptr<RtcCommonModeEvidence>(new RtcCommonModeEvidence);
  out->original_ = std::move(original);
  out->domain_ = std::move(domain);
  out->attempt_ = attempt;
  const auto &dom = out->domain_;
  const auto n = axis.occurrence_count(), nd = dom.members.size();
  const auto first = axis.first_native_row();
  out->speed_limit_ =
      rtc_optical_scale(dom.array, dom.speed_ceiling_arcsec_per_sec)
          .airy_fwhm_arcsec *
      ((1 - dom.cadence_margin_fraction) / dom.nominal_interval_seconds) /
      (4 * dom.output_factor * (1 + dom.speed_margin_fraction));
  out->reference_.assign(n, NAN);
  out->speed_admitted_.resize(n);
  for (auto row = first; row < axis.past_last_native_row(); ++row) {
    const auto v = dom.motion->scalar_speed_arcsec_per_sec(row);
    out->speed_admitted_[row - first] =
        v && ast_scan_motion_speed_admitted(*v) && *v <= out->speed_limit_ &&
        *v * (1 + dom.speed_margin_fraction) <=
            dom.speed_ceiling_arcsec_per_sec;
  }
  std::vector<std::vector<TimestreamNativeRow>> edges(nd);
  std::vector<std::vector<RtcEventRange>> noise_missing(nd);
  for (const auto &c : out->original_->candidates()) {
    const auto &b = out->original_->blocks()[c.noise_block_index];
    if (b.network_id == dom.network)
      edges[b.detector_index].push_back(c.later_row);
  }
  for (auto &e : edges) {
    std::sort(e.begin(), e.end());
    e.erase(std::unique(e.begin(), e.end()), e.end());
  }
  for (const auto &b : out->original_->blocks())
    if (b.network_id == dom.network && !b.pair_screening_available())
      noise_missing[b.detector_index].push_back({b.first, b.past_last});
  // Diagnostic support is computed once per interval, not a full duplicated
  // validity plane.
  std::vector<bool> seen(n, false);
  for (const auto &s : dom.scans->supports())
    if (s.native.network_id == dom.network) {
      for (const auto &run : axis.contiguous_runs()) {
        const RtcEventRange rows{
            std::max(s.native.first_native_row, run.first_native_row),
            std::min(s.native.past_last_native_row, run.past_last_native_row)};
        if (!rows.present())
          continue;
        for (auto row = rows.first; row < rows.past_last; ++row) {
          if (seen[row - first])
            throw std::invalid_argument(
                "RTC common mode processing intervals overlap");
          seen[row - first] = true;
        }
        RtcCommonModeInterval interval;
        interval.scan = s.scan;
        interval.rows = rows;
        interval.reference_reasons.resize(nd);
        interval.baseline.assign(nd, NAN);
        interval.paired_rows.resize(nd);
        interval.target_eligible_rows.resize(nd);
        for (auto row = rows.first; row < rows.past_last; ++row)
          interval.speed_eligible_rows += out->speed_admitted_[row - first];
        std::vector<std::vector<std::uint8_t>> eligible(
            nd, std::vector<std::uint8_t>(rows.past_last - rows.first));
        for (std::size_t d = 0; d < nd; ++d) {
          auto &reason = interval.reference_reasons[d];
          if (!dom.members[d].reference_eligible)
            reason |= 1;
          const auto edge =
              std::lower_bound(edges[d].begin(), edges[d].end(), rows.first);
          if (edge != edges[d].end() && *edge < rows.past_last)
            reason |= 4;
          std::vector<double> values;
          for (auto row = rows.first; row < rows.past_last; ++row) {
            if (net.state(NativeReadoutCoordinate::x, row, d).valid() &&
                net.state(NativeReadoutCoordinate::r, row, d).valid())
              ++interval.paired_rows[d];
            if (!out->speed_admitted_[row - first])
              continue;
            if (!net.state(NativeReadoutCoordinate::x, row, d).valid() ||
                !net.state(NativeReadoutCoordinate::r, row, d).valid()) {
              reason |= 2;
              continue;
            }
            if (!std::isfinite(net.value(NativeReadoutCoordinate::x, row, d)) ||
                !std::isfinite(net.value(NativeReadoutCoordinate::r, row, d)))
              throw std::invalid_argument(
                  "RTC common mode unexpected nonfinite original paired "
                  "measurement");
            if (rtc_event_assessment_detail::contains(noise_missing[d], row)) {
              reason |= 8;
              continue;
            }
            eligible[d][row - rows.first] = 1;
            ++interval.target_eligible_rows[d];
            values.push_back(net.value(NativeReadoutCoordinate::x, row, d));
          }
          if (values.size() < RtcCommonModePolicy::minimum_samples)
            reason |= 16;
          if (!reason) {
            interval.baseline[d] = median(std::move(values));
            ++interval.contributors;
          }
        }
        if (interval.contributors >=
            RtcCommonModePolicy::minimum_contributors) {
          std::vector<double> values, reference_values;
          values.reserve(nd);
          for (auto row = rows.first; row < rows.past_last; ++row)
            if (out->speed_admitted_[row - first]) {
              values.clear();
              for (std::size_t d = 0; d < nd; ++d)
                if (!interval.reference_reasons[d])
                  values.push_back(
                      net.value(NativeReadoutCoordinate::x, row, d) -
                      interval.baseline[d]);
              const double v = rtc_spike_detail::median(values);
              out->reference_[row - first] = v;
              reference_values.push_back(v);
            }
          interval.reference_scatter = mad(reference_values);
        }
        const auto index = out->intervals_.size();
        out->intervals_.push_back(std::move(interval));
        for (std::uint32_t d = 0; d < nd; ++d) {
          auto begin = rows.first;
          auto emit = [&](auto end) {
            if (begin < end)
              out->fits_.push_back(
                  fit(net, out->reference_, {d, index, {begin, end}}));
          };
          for (auto row = rows.first; row < rows.past_last; ++row) {
            const bool usable = eligible[d][row - rows.first] &&
                                std::isfinite(out->reference_[row - first]);
            if (!usable) {
              emit(row);
              begin = row + 1;
            } else if (std::binary_search(edges[d].begin(), edges[d].end(),
                                          row)) {
              emit(row);
              begin = row;
            }
          }
          emit(rows.past_last);
        }
        // Signed post-fit comparison, giving each eligible detector equal
        // weight.
        std::vector<std::vector<double>> slopes(nd);
        for (auto it = out->fits_.rbegin();
             it != out->fits_.rend() && it->interval == index; ++it)
          if (it->available())
            slopes[it->detector].push_back(it->gain);
        std::vector<double> calibrated;
        for (std::size_t d = 0; d < nd; ++d)
          if (!out->intervals_.back().reference_reasons[d] &&
              std::isfinite(dom.members[d].flxscale) &&
              dom.members[d].flxscale != 0) {
            const auto value = dom.members[d].flxscale * median(slopes[d]);
            if (std::isfinite(value))
              calibrated.push_back(value);
          }
        const double center = median(calibrated);
        if (calibrated.size() >= RtcCommonModePolicy::minimum_contributors &&
            std::isfinite(center) && center != 0)
          for (auto it = out->fits_.rbegin();
               it != out->fits_.rend() && it->interval == index; ++it) {
            const double phi = dom.members[it->detector].flxscale;
            if (it->available() && std::isfinite(phi) && phi != 0)
              it->calibrated_relative_gain = phi * it->gain / center;
          }
      }
    }
  return out;
}
std::size_t RtcCommonModeEvidence::logical_owned_bytes() const {
  std::size_t bytes = reference_.size() * sizeof(double) +
                      speed_admitted_.size() / 8 +
                      fits_.size() * sizeof(RtcCommonModeFit);
  for (const auto &s : intervals_)
    bytes += sizeof(s) + s.reference_reasons.size() +
             s.baseline.size() * sizeof(double) +
             (s.paired_rows.size() + s.target_eligible_rows.size()) *
                 sizeof(std::size_t);
  return bytes;
}
double RtcCommonModeEvidence::residual(std::size_t index,
                                       TimestreamNativeRow row) const {
  const auto &f = fits_.at(index);
  if (row < f.rows.first || row >= f.rows.past_last)
    throw std::out_of_range("RTC residual outside exact fit support");
  if (!f.available())
    return NAN;
  const auto &net = original_->input_handle()->network(domain_.network);
  return net.value(NativeReadoutCoordinate::x, row, f.detector) - f.offset -
         f.gain * reference_[row - net.occurrence_axis().first_native_row()];
}
std::vector<RtcCommonModeSelfCheck> RtcCommonModeEvidence::self_excluded_checks(
    std::span<const std::uint32_t> targets) const {
  if (targets.size() > RtcCommonModePolicy::maximum_self_checks)
    throw std::invalid_argument("RTC bounded self-exclusion budget exceeded");
  std::set<std::uint32_t> seen;
  std::vector<RtcCommonModeSelfCheck> checks;
  const auto &net = original_->input_handle()->network(domain_.network);
  const auto first = net.occurrence_axis().first_native_row();
  for (auto d : targets) {
    if (d >= domain_.members.size() || !seen.insert(d).second)
      throw std::invalid_argument("RTC self-check target invalid/repeated");
    RtcCommonModeSelfCheck check;
    check.evidence = shared_from_this();
    check.detector = d;
    check.reference.assign(reference_.size(), NAN);
    std::vector<double> values;
    values.reserve(domain_.members.size());
    for (const auto &s : intervals_)
      if (s.contributors - (!s.reference_reasons[d]) >=
          RtcCommonModePolicy::minimum_contributors)
        for (auto row = s.rows.first; row < s.rows.past_last; ++row)
          if (speed_admitted_[row - first]) {
            values.clear();
            for (std::size_t j = 0; j < domain_.members.size(); ++j)
              if (j != d && !s.reference_reasons[j])
                values.push_back(net.value(NativeReadoutCoordinate::x, row, j) -
                                 s.baseline[j]);
            check.reference[row - first] = rtc_spike_detail::median(values);
          }
    for (const auto &f : fits_)
      if (f.detector == d) {
        auto updated = fit(net, check.reference, {d, f.interval, f.rows});
        // Only the target is refitted. Do not relabel its gain against peer
        // fits made with another reference; calibrated relative LOO gain is
        // unavailable.
        check.fits.push_back(updated);
      }
    checks.push_back(std::move(check));
  }
  return checks;
}
RtcCommonModeConsideration RtcCommonModeConsideration::compare(
    std::shared_ptr<const RtcCommonModeEvidence> evidence,
    std::shared_ptr<const RtcLinePowerEvidence> lines) {
  if (!evidence || !lines ||
      lines->spectral_handle()->original_spike_handle().get() !=
          evidence->original_handle().get() ||
      lines->snapshot_handle().get() !=
          evidence->original_handle()->val_snapshot_handle().get())
    throw std::invalid_argument("RTC health comparison requires exact original "
                                "transient/spectral/VAL evidence");
  RtcCommonModeConsideration out;
  out.evidence_ = std::move(evidence);
  out.lines_ = std::move(lines);
  const auto &ev = *out.evidence_;
  const auto nw = ev.domain().network;
  for (std::uint32_t d = 0; d < ev.domain().members.size(); ++d) {
    RtcCommonModeDetectorSummary s;
    s.detector = d;
    std::vector<double> gains, relative, corr, residual;
    std::size_t negative = 0;
    for (const auto &f : ev.fits())
      if (f.detector == d) {
        if (!f.available()) {
          ++s.unavailable_segments;
          continue;
        }
        ++s.available_segments;
        s.fitted_rows += f.rows.past_last - f.rows.first;
        gains.push_back(f.gain);
        if (std::isfinite(f.calibrated_relative_gain)) {
          relative.push_back(f.calibrated_relative_gain);
          negative += f.calibrated_relative_gain < 0;
        }
        if (std::isfinite(f.correlation))
          corr.push_back(f.correlation);
        residual.push_back(f.residual_scatter);
      }
    s.gain = median(gains);
    if (gains.size() > 1)
      s.gain_mad = mad(gains);
    s.relative_gain = median(relative);
    if (!relative.empty())
      s.negative_relative_fraction = double(negative) / relative.size();
    s.correlation = median(corr);
    s.residual_scatter = median(residual);
    for (const auto &c : ev.original_handle()->candidates()) {
      const auto &b = ev.original_handle()->blocks()[c.noise_block_index];
      if (b.network_id == nw && b.detector_index == d)
        ++s.original_candidates;
    }
    for (const auto &b : ev.original_handle()->blocks())
      if (b.network_id == nw && b.detector_index == d &&
          !b.pair_screening_available())
        ++s.unavailable_noise_blocks;
    const auto &spectrum =
        out.lines_->coordinate(nw, d, NativeReadoutCoordinate::x);
    if (spectrum.available() && !spectrum.regions.empty()) {
      const auto &peak = *std::max_element(
          spectrum.regions.begin(), spectrum.regions.end(),
          [](const auto &a, const auto &b) {
            return a.positive_excess_power < b.positive_excess_power;
          });
      s.spectral_peak_hz = out.lines_->spectral_handle()
                               ->network(nw)
                               .frequency_hz[peak.peak_bin];
      s.spectral_excess_fraction = peak.stored_psd_power_fraction;
    }
    out.detectors_.push_back(s);
  }
  // Bounded inspection selection, not scientific admission. Include contrasting
  // signed/calibrated response, large residual, weak correlation, and ordinary
  // middle-of-distribution controls; support/LOO determine interpretation.
  std::vector<std::uint32_t> order;
  for (const auto &s : out.detectors_)
    if (s.available_segments)
      order.push_back(s.detector);
  auto add = [&](auto d) {
    if (std::find(out.targets_.begin(), out.targets_.end(), d) ==
        out.targets_.end())
      out.targets_.push_back(d);
  };
  auto select = [&](auto metric, bool both) {
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
      return metric(out.detectors_[a]) < metric(out.detectors_[b]);
    });
    if (!order.empty()) {
      add(order.front());
      if (both)
        add(order.back());
    }
  };
  select(
      [](const auto &s) {
        return std::isfinite(s.relative_gain) ? s.relative_gain : 1.;
      },
      true);
  select([](const auto &s) { return -s.residual_scatter; }, false);
  std::erase_if(order, [&](auto d) {
    return !ev.domain().members[d].reference_eligible;
  });
  select([](const auto &s) { return -s.residual_scatter; }, false);
  select(
      [](const auto &s) {
        return std::isfinite(s.correlation) ? std::abs(s.correlation) : 0.;
      },
      false);
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
    return out.detectors_[a].residual_scatter <
           out.detectors_[b].residual_scatter;
  });
  for (auto numerator : {2, 4, 6})
    if (!order.empty())
      add(order[(order.size() - 1) * numerator / 8]);
  return out;
}
} // namespace citlali::pipeline
