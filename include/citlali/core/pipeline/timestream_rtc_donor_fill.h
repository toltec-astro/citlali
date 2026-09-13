#pragma once

#include <citlali/core/pipeline/timestream_rtc_transient_exclusion.h>

namespace citlali::pipeline {

// Native x continuity only. See the owner decisions and bounded work order in
// TIMESTREAM_SUCCESSOR_RTC_EVENT_BACKGROUND_001_2026-09-08.md.
struct RtcDonorFillPolicy {
    static constexpr std::string_view identity = "rtc-native-donor-median-cubic-quartic-v1";
    static double taper(double u) {
        if (!std::isfinite(u) || u < 0 || u > 1)
            throw std::invalid_argument("RTC donor taper requires normalized in-gap time");
        const double v = u * (1 - u);
        return 16 * v * v;
    }
};

enum class RtcDonorSelectionState : std::uint8_t { unavailable, accepted_isolated_event };

// Input from the RTC event-admission owner, NOT an automatic classifier.
// A complete explicit selection is needed even for a recovered candidate.
struct RtcDonorSelectedEvent {
    std::shared_ptr<const RtcEventAssessmentEvidence> evidence;
    std::string admission_authority;
    RtcDonorSelectionState state = RtcDonorSelectionState::unavailable;
    std::size_t event = 0;
    RtcEventRange affected;
};

// Facts supplied by the static-factor and segmentation owners, bound below to
// the exact original Learn/VAL generation. No conversion from run-averaged FCF.
struct RtcDonorDetectorFacts {
    TimestreamNetworkId network = -1;
    std::uint32_t detector = 0;
    std::string detector_occurrence_id;
    std::string factor_identity;
    std::string factor_convention;
    std::optional<double> prior_flxscale;
    RtcEventRange factor_support;
    // Omission means unavailable; these are explicitly resolved stable ranges,
    // not inferred simply from an absence of detected candidates.
    std::vector<RtcEventRange> stable_segments;
    std::vector<RtcEventRange> contaminated;
};

class RtcDonorFillFacts {
public:
    static std::shared_ptr<const RtcDonorFillFacts> bind(
        std::shared_ptr<const RtcEventAssessmentEvidence> evidence,
        std::string prior_factor_authority, std::string compatible_convention,
        std::string segmentation_authority, std::string contamination_authority,
        std::vector<RtcDonorDetectorFacts> detectors) {
        if (!evidence || prior_factor_authority.empty() || compatible_convention.empty() ||
            segmentation_authority.empty() || contamination_authority.empty())
            throw std::invalid_argument("RTC donor facts require original evidence and named producer authorities");
        std::sort(detectors.begin(), detectors.end(), [](const auto &a, const auto &b) {
            return std::tie(a.network, a.detector) < std::tie(b.network, b.detector);
        });
        std::optional<std::pair<TimestreamNetworkId, std::uint32_t>> previous;
        for (const auto &d : detectors) {
            const auto &net = evidence->spike_handle()->input_handle()->network(d.network);
            if (d.detector >= net.detectors().size() ||
                net.detector(d.detector).detector_occurrence_id != d.detector_occurrence_id ||
                previous == std::pair{d.network, d.detector})
                throw std::invalid_argument("RTC donor facts have stale or repeated detector identity");
            previous = std::pair{d.network, d.detector};
            const auto &axis = net.occurrence_axis();
            auto check = [&](RtcEventRange r) {
                if (!r.present() || r.first < axis.first_native_row() || r.past_last > axis.past_last_native_row())
                    throw std::invalid_argument("RTC donor fact support is outside its exact parent");
            };
            if (d.prior_flxscale) {
                if (d.factor_identity.empty() || d.factor_convention.empty())
                    throw std::invalid_argument("RTC donor factor lacks identity or convention");
                check(d.factor_support);
            }
            TimestreamNativeRow end = -1;
            for (auto s : d.stable_segments) {
                check(s);
                if (s.first < end || rtc_event_assessment_detail::run_for(axis, s.first).past_last < s.past_last)
                    throw std::invalid_argument("RTC stable segments overlap, are unordered or cross a physical gap");
                end = s.past_last;
            }
            end = -1;
            for (auto s : d.contaminated) {
                check(s);
                if (s.first < end) throw std::invalid_argument("RTC donor contamination is unordered or overlaps");
                end = s.past_last;
            }
        }
        return std::shared_ptr<const RtcDonorFillFacts>(new RtcDonorFillFacts{
            std::move(evidence), std::move(prior_factor_authority), std::move(compatible_convention),
            std::move(segmentation_authority), std::move(contamination_authority), std::move(detectors)});
    }
    const auto &evidence_handle() const noexcept { return evidence_; }
    const auto &factor_authority() const noexcept { return factor_authority_; }
    const auto &convention() const noexcept { return convention_; }
    const auto &segmentation_authority() const noexcept { return segmentation_authority_; }
    const auto &contamination_authority() const noexcept { return contamination_authority_; }
    const auto &detectors() const noexcept { return detectors_; }
    const RtcDonorDetectorFacts *find(TimestreamNetworkId n, std::uint32_t d) const {
        const auto it = std::lower_bound(detectors_.begin(), detectors_.end(), std::pair{n, d},
            [](const auto &r, auto k) { return std::pair{r.network, r.detector} < k; });
        return it != detectors_.end() && it->network == n && it->detector == d ? &*it : nullptr;
    }
private:
    RtcDonorFillFacts(std::shared_ptr<const RtcEventAssessmentEvidence> e, std::string f,
        std::string c, std::string s, std::string x, std::vector<RtcDonorDetectorFacts> d)
        : evidence_{std::move(e)}, factor_authority_{std::move(f)}, convention_{std::move(c)},
          segmentation_authority_{std::move(s)}, contamination_authority_{std::move(x)}, detectors_{std::move(d)} {}
    std::shared_ptr<const RtcEventAssessmentEvidence> evidence_;
    std::string factor_authority_, convention_, segmentation_authority_, contamination_authority_;
    std::vector<RtcDonorDetectorFacts> detectors_;
};

enum class RtcDonorFillCause : std::uint8_t {
    ready, background_unavailable, boundary_unavailable, target_excluded,
    target_transfer_unavailable, no_usable_donor, arithmetic_nonfinite
};

struct RtcDonorMedianOccurrence {
    TimestreamNativeRow row = -1;
    double value = NAN;
    // Original detector identities and factors are recovered through the facts
    // handle. Keep the whole admitted population for data-dependent selection;
    // only the central one/two donors supply the fixed-selection median value.
    std::vector<std::uint32_t> eligible;
    std::array<std::uint32_t, 2> central{};
    std::size_t central_count = 0;
};

class RtcDonorFillPlan {
public:
    static std::shared_ptr<const RtcDonorFillPlan> consider(
        RtcDonorSelectedEvent selection, std::shared_ptr<const RtcDonorFillFacts> facts,
        std::shared_ptr<const RtcTransientExclusionPlan> exclusions,
        std::shared_ptr<const ValSnapshot> snapshot, std::uint64_t id) {
        if (!selection.evidence || !facts || !exclusions || !id || selection.admission_authority.empty() ||
            selection.state != RtcDonorSelectionState::accepted_isolated_event ||
            selection.evidence.get() != facts->evidence_handle().get() ||
            selection.event >= selection.evidence->events().size() || !selection.affected.present() ||
            selection.evidence->spike_handle().get() != exclusions->screening_handle()->evidence_handle().get() ||
            selection.evidence.get() != &exclusions->jump_plan_handle()->admission_handle()->assessment())
            throw std::invalid_argument("RTC donor plan requires explicit selected event and exact common Learn evidence");
        if (!snapshot || snapshot.get() != exclusions->val_snapshot_handle().get())
            throw StaleRtcValGeneration("RTC donor plan requires exact original VAL snapshot");
        const auto &e = selection.evidence->events()[selection.event];
        const auto &spikes = *selection.evidence->spike_handle();
        const auto &net = spikes.input_handle()->network(e.network);
        const auto &axis = net.occurrence_axis();
        const auto r = selection.affected;
        const auto run = rtc_event_assessment_detail::run_for(axis, r.first);
        const auto &seed = spikes.candidates()[e.seed];
        if (r.past_last > run.past_last || r.first > seed.later_row || r.past_last <= seed.earlier_row)
            throw std::invalid_argument("RTC donor selected support misses event or crosses a physical gap");
        // No protected-event admission implementation exists in this increment.
        // Check the entire event/background context, not only the central cell.
        auto first = r.first, last = r.past_last;
        for (const auto &side : e.background[0].support) if (side.usable) {
            first = std::min(first, side.first_used); last = std::max(last, side.last_used + 1);
        }
        for (auto row = first; row < last; ++row)
            if (spikes.protection_handle()->state(e.network, e.detector, row) != RtcSpikeProtection::outside_source)
                throw std::invalid_argument("RTC donor selection lacks outside-source authority; optical admission is separate");
        auto out = std::shared_ptr<RtcDonorFillPlan>(new RtcDonorFillPlan);
        out->selection_ = std::move(selection); out->facts_ = std::move(facts);
        out->exclusions_ = std::move(exclusions); out->id_ = id;
        auto fail = [&](RtcDonorFillCause c) { out->cause_ = c; out->medians_.clear(); return out; };
        if (e.refinement_limited || !e.background[0].available() || !std::isfinite(e.origin) ||
            !std::isfinite(e.time_scale) || e.time_scale <= 0)
            return fail(RtcDonorFillCause::background_unavailable);
        if (r.first == run.first || r.past_last == run.past_last)
            return fail(RtcDonorFillCause::boundary_unavailable);
        // The supplied background must have been learned outside this selected
        // gap, with clean context on both sides. Do not reuse a fit that consumed
        // any newly selected sample, or extrapolate beyond its learned flanks.
        if (e.background[0].support[0].last_used >= r.first ||
            e.background[0].support[1].first_used < r.past_last)
            return fail(RtcDonorFillCause::background_unavailable);
        out->support_ = {r.first - 1, r.past_last + 1};
        const auto *target = out->facts_->find(e.network, e.detector);
        if (!target || !out->transfer_available(*target, out->support_) || *target->prior_flxscale == 0)
            return fail(RtcDonorFillCause::target_transfer_unavailable);
        if (!stable(*target, {first, last}) || !stable(*target, out->support_))
            return fail(RtcDonorFillCause::boundary_unavailable);
        for (auto row = out->support_.first; row < out->support_.past_last; ++row) {
            if (out->exclusions_->excludes(e.network, row, e.detector))
                return fail(RtcDonorFillCause::target_excluded);
            if ((row < r.first || row >= r.past_last) && !pair_usable(net, row, e.detector))
                return fail(RtcDonorFillCause::boundary_unavailable);
            if ((row < r.first || row >= r.past_last) &&
                rtc_event_assessment_detail::contains(target->contaminated, row))
                return fail(RtcDonorFillCause::boundary_unavailable);
        }
        out->begin_ = rtc_event_assessment_detail::time(axis, out->support_.first);
        out->end_ = rtc_event_assessment_detail::time(axis, out->support_.past_last - 1);
        if (!std::isfinite(out->end_ - out->begin_) || out->end_ <= out->begin_)
            return fail(RtcDonorFillCause::boundary_unavailable);
        // Candidate guards are exclusions for donor eligibility, not new event
        // labels. Caller-supplied contamination additionally includes unresolved
        // or longer affected support from the segmentation/treatment owner.
        std::map<std::uint32_t, std::vector<RtcEventRange>> guards;
        for (const auto &candidate : spikes.candidates()) {
            const auto &b = spikes.blocks()[candidate.noise_block_index];
            if (b.network_id != e.network) continue;
            auto g = rtc_event_assessment_detail::trial(axis,
                rtc_event_assessment_detail::run_for(axis, candidate.earlier_row), candidate);
            if (g.first < out->support_.past_last && g.past_last > out->support_.first)
                guards[b.detector_index].push_back(g);
        }
        for (auto &[d, ranges] : guards) ranges = rtc_event_assessment_detail::merge(std::move(ranges));
        for (auto row = out->support_.first; row < out->support_.past_last; ++row) {
            std::vector<std::pair<double, std::uint32_t>> values;
            for (const auto &d : out->facts_->detectors()) {
                if (d.network != e.network || d.detector == e.detector ||
                    !out->selection_.evidence->population_handle()->eligible(d.network, d.detector) ||
                    !out->transfer_available(d, out->support_) || !stable(d, out->support_) ||
                    rtc_event_assessment_detail::contains(d.contaminated, row) ||
                    (guards.contains(d.detector) && rtc_event_assessment_detail::contains(guards.at(d.detector), row)) ||
                    out->exclusions_->excludes(d.network, row, d.detector) || !pair_usable(net, row, d.detector) ||
                    spikes.protection_handle()->state(d.network, d.detector, row) != RtcSpikeProtection::outside_source) continue;
                const double scale = *d.prior_flxscale / *target->prior_flxscale;
                const double value = scale * net.value(NativeReadoutCoordinate::x, row, d.detector);
                if (std::isfinite(scale) && std::isfinite(value)) values.emplace_back(value, d.detector);
            }
            if (values.empty()) return fail(RtcDonorFillCause::no_usable_donor);
            std::sort(values.begin(), values.end());
            RtcDonorMedianOccurrence m; m.row = row;
            for (auto [value, d] : values) m.eligible.push_back(d);
            const auto mid = values.size() / 2;
            m.central[0] = values[mid].second; m.central_count = 1; m.value = values[mid].first;
            if (values.size() % 2 == 0) {
                m.central[1] = values[mid - 1].second; m.central_count = 2;
                m.value = std::midpoint(values[mid].first, values[mid - 1].first);
            }
            out->medians_.push_back(std::move(m));
        }
        return out;
    }
    const auto &selection() const noexcept { return selection_; }
    const auto &facts_handle() const noexcept { return facts_; }
    const auto &exclusions_handle() const noexcept { return exclusions_; }
    const auto &input_handle() const noexcept { return exclusions_->input_handle(); }
    const auto &val_snapshot_handle() const noexcept { return exclusions_->val_snapshot_handle(); }
    const auto &medians() const noexcept { return medians_; }
    RtcEventRange donor_support() const noexcept { return support_; }
    RtcDonorFillCause cause() const noexcept { return cause_; }
    std::uint64_t consideration() const noexcept { return id_; }
    const auto &event() const { return selection_.evidence->events()[selection_.event]; }
    double begin_time() const noexcept { return begin_; }
    double end_time() const noexcept { return end_; }
    std::size_t logical_owned_bytes() const noexcept {
        std::size_t n = sizeof(*this) + selection_.admission_authority.size() + medians_.size() * sizeof(RtcDonorMedianOccurrence);
        for (const auto &m : medians_) n += m.eligible.size() * sizeof(std::uint32_t);
        return n;
    }
private:
    RtcDonorFillPlan() = default;
    static bool covers(RtcEventRange a, RtcEventRange b) { return a.first <= b.first && a.past_last >= b.past_last; }
    static bool stable(const RtcDonorDetectorFacts &d, RtcEventRange s) {
        return std::any_of(d.stable_segments.begin(), d.stable_segments.end(), [&](auto r) { return covers(r, s); });
    }
    bool transfer_available(const RtcDonorDetectorFacts &d, RtcEventRange r) const {
        return d.prior_flxscale && std::isfinite(*d.prior_flxscale) &&
            d.factor_convention == facts_->convention() && covers(d.factor_support, r);
    }
    static bool pair_usable(const NativePairedReadoutNetwork &n, TimestreamNativeRow r, std::uint32_t d) {
        for (auto c : {NativeReadoutCoordinate::x, NativeReadoutCoordinate::r})
            if (!n.state(c, r, d).valid() || !std::isfinite(n.value(c, r, d))) return false;
        return true;
    }
    RtcDonorSelectedEvent selection_;
    std::shared_ptr<const RtcDonorFillFacts> facts_;
    std::shared_ptr<const RtcTransientExclusionPlan> exclusions_;
    std::vector<RtcDonorMedianOccurrence> medians_;
    RtcEventRange support_;
    double begin_ = NAN, end_ = NAN;
    RtcDonorFillCause cause_ = RtcDonorFillCause::ready;
    std::uint64_t id_ = 0;
};

class RtcDonorFillResult {
public:
    static std::shared_ptr<const RtcDonorFillResult> apply(
        std::shared_ptr<const RtcDonorFillPlan> plan,
        std::shared_ptr<const NativePairedReadoutView> input,
        std::shared_ptr<const ValSnapshot> snapshot,
        std::span<const std::shared_ptr<const NativePairedReadoutView>> partitions) {
        if (!plan || !input || plan->input_handle().get() != input.get())
            throw std::invalid_argument("RTC donor Apply requires exact plan-bound input");
        if (!snapshot || snapshot.get() != plan->val_snapshot_handle().get())
            throw StaleRtcValGeneration("RTC donor Apply requires exact plan-bound VAL");
        auto out = std::shared_ptr<RtcDonorFillResult>(new RtcDonorFillResult);
        out->retained_ = RtcTransientExclusionResult::apply(plan->exclusions_handle(), input, snapshot, partitions);
        out->plan_ = std::move(plan); out->cause_ = out->plan_->cause();
        if (out->cause_ != RtcDonorFillCause::ready) return out;
        const auto &p = *out->plan_; const auto &e = p.event();
        const auto &axis = input->network(e.network).occurrence_axis();
        for (const auto &m : p.medians()) {
            if (m.row < p.selection().affected.first || m.row >= p.selection().affected.past_last) continue;
            const double time = rtc_event_assessment_detail::time(axis, m.row);
            const double u = (time - p.begin_time()) / (p.end_time() - p.begin_time());
            const double donor_line = std::lerp(p.medians().front().value, p.medians().back().value, u);
            const double background = rtc_event_assessment_detail::polynomial(e.background[0].cubic, (time - e.origin) / e.time_scale);
            const double value = background + RtcDonorFillPolicy::taper(u) * (m.value - donor_line);
            if (!std::isfinite(value)) {
                out->values_.clear(); out->cause_ = RtcDonorFillCause::arithmetic_nonfinite; return out;
            }
            out->values_.push_back(value);
        }
        return out;
    }
    const auto &plan_handle() const noexcept { return plan_; }
    RtcDonorFillCause cause() const noexcept { return cause_; }
    bool filled() const noexcept { return cause_ == RtcDonorFillCause::ready; }
    std::size_t owned_numeric_bytes() const noexcept { return values_.size() * sizeof(double); }
    bool selected(TimestreamNetworkId n, TimestreamNativeRow r, std::uint32_t d) const {
        const auto &e = plan_->event(); const auto s = plan_->selection().affected;
        return n == e.network && d == e.detector && r >= s.first && r < s.past_last;
    }
    bool requires_map_exclusion(TimestreamNetworkId n, TimestreamNativeRow r, std::uint32_t d) const {
        // Validate the address even when selected; no synthetic independent cell.
        const bool prior = plan_->exclusions_handle()->excludes(n, r, d);
        return prior || selected(n, r, d);
    }
    std::optional<double> value_for_conditioning(NativeReadoutCoordinate c,
        TimestreamNetworkId n, TimestreamNativeRow r, std::uint32_t d) const {
        if (c != NativeReadoutCoordinate::x && c != NativeReadoutCoordinate::r)
            throw std::invalid_argument("RTC donor result requires x or r");
        if (selected(n, r, d)) {
            if (c == NativeReadoutCoordinate::r || !filled()) return std::nullopt;
            return values_.at(static_cast<std::size_t>(r - plan_->selection().affected.first));
        }
        return retained_->value_if_retained(c, n, r, d);
    }
    std::optional<double> value_if_independent(NativeReadoutCoordinate c,
        TimestreamNetworkId n, TimestreamNativeRow r, std::uint32_t d) const {
        if (c != NativeReadoutCoordinate::x && c != NativeReadoutCoordinate::r)
            throw std::invalid_argument("RTC donor result requires x or r");
        if (requires_map_exclusion(n, r, d)) return std::nullopt;
        return retained_->value_if_retained(c, n, r, d);
    }
private:
    RtcDonorFillResult() = default;
    std::shared_ptr<const RtcDonorFillPlan> plan_;
    std::shared_ptr<const RtcTransientExclusionResult> retained_;
    std::vector<double> values_;
    RtcDonorFillCause cause_ = RtcDonorFillCause::no_usable_donor;
};

} // namespace citlali::pipeline
