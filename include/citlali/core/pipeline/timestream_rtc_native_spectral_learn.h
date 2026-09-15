#pragma once

#include <citlali/core/pipeline/timestream_rtc_event_assessment.h>
#include <unsupported/Eigen/FFT>
#include <complex>
#include <numbers>

namespace citlali::pipeline {

// Initial measurement policy, not notch/event admission. The implementation
// preserves the D2 fixed-grid wrapper and its actual masked-Welch conventions.
struct RtcInitialSpectralPolicy {
    static constexpr std::string_view identity = "rtc-original-native-spectral-use-v1";
    static constexpr std::string_view estimator = "d2-fixed-grid-masked-welch-4s-v1";
    static constexpr double segment_seconds = 4., minimum_segment_seconds = 2., overlap = .5;
    static constexpr std::size_t minimum_windows = 2, minimum_samples = 16;
    static constexpr std::string_view conventions =
        "network-median-then-chunk-median;symmetric-Hann;fs-sum-w2;interior-rfft-bins-doubled;"
        "arithmetic-window-mean;end-anchor;short-chunk-zero-pad;fixed-grid;nearest-even-rounding";
};

enum class RtcSpectralInputStage : std::uint8_t { original_reference, native_intermediate, native_conditioned };

// Identity only: this descriptor neither supplies a numerical product nor
// authorizes a use. Its snapshot is explicit and need not be generation zero.
// Intermediate/conditioned producers must additionally retain their
// actual numerical parent and replacement/support history in their evidence;
// a descriptor alone cannot publish such a spectrum through learn_initial().
class RtcSpectralInputIdentity {
public:
    static std::shared_ptr<const RtcSpectralInputIdentity> bind(
        std::shared_ptr<const ValNativeRealization> subject,
        std::shared_ptr<const ValSnapshot> snapshot, NativeOccurrenceSpan support,
        RtcSpectralInputStage stage, std::string processing_stage, std::uint64_t producer_attempt) {
        if (!subject || !snapshot || !producer_attempt || processing_stage.empty() ||
            subject->paired_handle().get() != snapshot->paired_handle().get() ||
            subject->network_id() != support.network_id || !support.occurrence_count())
            throw std::invalid_argument("RTC spectral identity requires exact subject, VAL, support, stage and attempt");
        const auto &a = subject->paired_handle()->network(support.network_id).occurrence_axis();
        if (support.first_native_row < a.first_native_row() || support.past_last_native_row > a.past_last_native_row() ||
            (stage != RtcSpectralInputStage::original_reference && stage != RtcSpectralInputStage::native_intermediate &&
             stage != RtcSpectralInputStage::native_conditioned) ||
            (stage == RtcSpectralInputStage::original_reference) != (subject->role() == ValNativeProductRole::original_input))
            throw std::invalid_argument("RTC spectral stage or support contradicts its exact subject");
        return std::shared_ptr<const RtcSpectralInputIdentity>(new RtcSpectralInputIdentity{
            std::move(subject), std::move(snapshot), support, stage, std::move(processing_stage), producer_attempt});
    }
    const auto &subject_handle() const noexcept { return subject_; }
    const auto &snapshot_handle() const noexcept { return snapshot_; }
    const auto &support() const noexcept { return support_; }
    auto stage() const noexcept { return stage_; }
    const auto &processing_stage() const noexcept { return processing_stage_; }
    auto producer_attempt() const noexcept { return producer_attempt_; }
private:
    RtcSpectralInputIdentity(std::shared_ptr<const ValNativeRealization> s, std::shared_ptr<const ValSnapshot> v,
        NativeOccurrenceSpan r, RtcSpectralInputStage stage, std::string p, std::uint64_t a)
        : subject_{std::move(s)}, snapshot_{std::move(v)}, support_{r}, stage_{stage}, processing_stage_{std::move(p)}, producer_attempt_{a} {}
    std::shared_ptr<const ValNativeRealization> subject_;
    std::shared_ptr<const ValSnapshot> snapshot_;
    NativeOccurrenceSpan support_;
    RtcSpectralInputStage stage_;
    std::string processing_stage_;
    std::uint64_t producer_attempt_;
};

class RtcNotchRecoveryResult;
class RtcPipelineResult;

// A concrete RTC product, created only by completed frozen-plan Apply. Numeric
// planes alias their immutable result owners; no copied original or replacement
// masquerades as an independent measurement. Native rows are never compacted.
class RtcConditionedNativeProduct {
public:
    struct Column {
        TimestreamNetworkId network;
        std::uint32_t detector;
        TimestreamNativeRow first;
        std::shared_ptr<const Eigen::Matrix<double, Eigen::Dynamic, 2>> values;
        // Coordinate availability (bits 0/1), representative replacement (2),
        // nonlocal/representative replacement influence (3), unrepaired donor
        // influence (4), representative independent-use exclusion (5). Eligibility is a
        // separate downstream decision; finite x does not invent available r.
        std::vector<std::uint8_t> state;
        std::shared_ptr<const RtcNotchRecoveryResult> source;
    };
    const auto &original_spike_handle() const noexcept { return original_; }
    const auto &snapshot_handle() const noexcept { return snapshot_; }
    const auto &identities() const noexcept { return identities_; }
    const auto &columns() const noexcept { return columns_; }
    auto producer_attempt() const noexcept { return attempt_; }
    bool after_lowpass() const noexcept { return after_lowpass_; }
    const Column &column(TimestreamNetworkId n, std::uint32_t d) const {
        const auto it = std::lower_bound(columns_.begin(), columns_.end(), std::pair{n,d},
            [](const auto &c, auto key) { return std::pair{c.network,c.detector} < key; });
        if (it == columns_.end() || it->network != n || it->detector != d)
            throw std::out_of_range("RTC conditioned detector absent");
        return *it;
    }
    static constexpr std::string_view use_policy = "rtc-conditioned-native-spectral-review-v1";
    static constexpr bool independent_measurements = false, classification_authorized = false;
private:
    friend class RtcPipelineResult;
    RtcConditionedNativeProduct() = default;
    std::shared_ptr<const RtcSpikeEvidence> original_;
    std::shared_ptr<const ValSnapshot> snapshot_;
    std::vector<std::shared_ptr<const RtcSpectralInputIdentity>> identities_;
    std::vector<Column> columns_;
    std::uint64_t attempt_ = 0;
    bool after_lowpass_ = false;
};

// Same explicit cadence-domain prerequisite as D2: no locally invented jitter
// tolerance. This owner-supplied bound is checked against original native time.
struct RtcSpectralCadenceDomain {
    TimestreamNetworkId network = -1;
    std::string authority;
    double nominal_interval_seconds = NAN, maximum_fractional_deviation = NAN;
};
enum class RtcSpectralRunCause : std::uint8_t {
    contributing, insufficient_support, input_consistency_failure, arithmetic_nonfinite
};
enum class RtcSpectralCause : std::uint8_t {
    available, available_with_unavailable_runs, cadence_unavailable, insufficient_windows,
    fixed_grid_unavailable, input_consistency_failure, arithmetic_nonfinite
};
struct RtcSpectralWindow {
    std::size_t run_index = 0;
    RtcEventRange rows;
    double support_begin_unix_sec = NAN, support_end_unix_sec = NAN;
    std::size_t padded_samples = 0;
    double centered_chunk_median = NAN;
    // Counts annotate original input; source status never excludes this profile.
    std::array<std::size_t, 3> source_counts{}; // outside, protected, unknown
    std::size_t representative_replacements = 0, replacement_influenced_samples = 0;
    std::size_t unrepaired_influenced_samples = 0, representative_exclusions = 0;
};
struct RtcSpectralRun {
    RtcEventRange rows;
    RtcSpectralRunCause cause = RtcSpectralRunCause::insufficient_support;
    std::size_t admitted_samples = 0, declared_invalid_samples = 0, unexpected_nonfinite_samples = 0;
    TimestreamNativeRow first_unexpected_nonfinite = -1;
    std::size_t first_window = 0, past_last_window = 0;
};
struct RtcNativeSpectrum {
    TimestreamNetworkId network = -1;
    std::uint32_t detector = 0;
    NativeReadoutCoordinate coordinate = NativeReadoutCoordinate::x;
    RtcSpectralCause cause = RtcSpectralCause::insufficient_windows;
    // All actually admitted finite stretches used in global centering. This
    // includes short tails that may not themselves supply an FFT window.
    std::vector<RtcEventRange> centering_support;
    double population_median = NAN;
    std::vector<RtcSpectralRun> runs;
    std::vector<RtcSpectralWindow> windows;
    std::vector<double> psd; // original coordinate units squared / Hz
    bool available() const noexcept {
        return cause == RtcSpectralCause::available || cause == RtcSpectralCause::available_with_unavailable_runs;
    }
};
struct RtcSpectralNetwork {
    std::shared_ptr<const RtcSpectralInputIdentity> input;
    RtcSpectralCadenceDomain cadence_domain;
    double interval_seconds = NAN, minimum_interval_seconds = NAN, maximum_interval_seconds = NAN;
    double maximum_fractional_interval_deviation = NAN;
    std::size_t fft_samples = 0, minimum_chunk_samples = 0, hop_samples = 0;
    double window_norm = NAN, equivalent_noise_bandwidth_hz = NAN;
    std::vector<double> frequency_hz;
    bool cadence_available = false;
};

namespace rtc_native_spectral_detail {
inline std::size_t rounded(double x) {
    if (!std::isfinite(x) || x < 0 || x >= static_cast<double>(std::numeric_limits<int>::max()))
        throw std::length_error("RTC spectral FFT length exceeds supported finite range");
    const auto n = static_cast<std::size_t>(std::floor(x));
    return n + (x - n > .5 || (x - n == .5 && n % 2));
}
inline double median(std::vector<double> values) {
    if (values.empty()) return NAN;
    std::sort(values.begin(), values.end());
    const auto m = values.size() / 2;
    // Preserve NumPy's even median arithmetic, including unavailable overflow.
    return values.size() % 2 ? values[m] : (values[m-1] + values[m]) / 2.;
}
inline std::size_t source_index(RtcSpikeProtection p) {
    return p == RtcSpikeProtection::outside_source ? 0 : p == RtcSpikeProtection::protected_source ? 1 : 2;
}
} // namespace rtc_native_spectral_detail

class RtcNativeSpectralEvidence {
public:
    static std::shared_ptr<const RtcNativeSpectralEvidence> learn_initial(
        std::shared_ptr<const RtcSpikeEvidence> original,
        std::vector<std::shared_ptr<const RtcSpectralInputIdentity>> inputs,
        std::vector<RtcSpectralCadenceDomain> cadence, std::uint64_t attempt) {
        return learn(std::move(original), std::move(inputs), std::move(cadence), attempt, nullptr);
    }
    static std::shared_ptr<const RtcNativeSpectralEvidence> learn_conditioned(
        std::shared_ptr<const RtcConditionedNativeProduct> product,
        std::shared_ptr<const ValSnapshot> snapshot,
        std::vector<RtcSpectralCadenceDomain> cadence, std::uint64_t attempt) {
        if (!product || !snapshot || product->snapshot_handle().get() != snapshot.get())
            throw std::invalid_argument("RTC conditioned Learn requires exact numerical product and VAL snapshot");
        return learn(product->original_spike_handle(), product->identities(), std::move(cadence), attempt, product);
    }
private:
    static std::shared_ptr<const RtcNativeSpectralEvidence> learn(
        std::shared_ptr<const RtcSpikeEvidence> original,
        std::vector<std::shared_ptr<const RtcSpectralInputIdentity>> inputs,
        std::vector<RtcSpectralCadenceDomain> cadence, std::uint64_t attempt,
        std::shared_ptr<const RtcConditionedNativeProduct> conditioned) {
        if (!original || !attempt || inputs.size() != original->input_handle()->network_count() || cadence.size() != inputs.size())
            throw std::invalid_argument("RTC spectral Learn requires complete original references and attempt");
        auto out = std::shared_ptr<RtcNativeSpectralEvidence>(new RtcNativeSpectralEvidence{original, attempt});
        out->conditioned_ = std::move(conditioned);
        for (std::size_t ni = 0; ni < inputs.size(); ++ni) {
            const auto &binding = inputs[ni]; const auto span = original->input_handle()->spans()[ni];
            if (!out->conditioned_ && (!binding || binding->subject_handle()->paired_handle().get() != original->input_handle()->parent_handle().get() ||
                binding->snapshot_handle().get() != original->val_snapshot_handle().get() ||
                binding->snapshot_handle()->generation().value != 0 || binding->support() != span ||
                binding->stage() != RtcSpectralInputStage::original_reference ||
                binding->producer_attempt() != original->attempt()))
                throw std::invalid_argument("RTC initial spectrum requires exact original stage, initial VAL and learning attempt");
            const auto &domain = cadence[ni];
            if (domain.network != span.network_id || domain.authority.empty() ||
                !std::isfinite(domain.nominal_interval_seconds) || domain.nominal_interval_seconds <= 0 ||
                !std::isfinite(domain.maximum_fractional_deviation) || domain.maximum_fractional_deviation < 0)
                throw std::invalid_argument("RTC spectral cadence domain is absent or malformed");
            const auto &net = original->input_handle()->network(span.network_id); const auto &axis = net.occurrence_axis();
            RtcSpectralNetwork n; n.input = binding; n.cadence_domain = domain;
            {
            std::vector<double> intervals;
            for (const auto &run : axis.contiguous_runs())
                for (auto row = run.first_native_row + 1; row < run.past_last_native_row; ++row)
                    intervals.push_back(axis.native_identity(row).reconstructed_time_unix_sec() - axis.native_identity(row-1).reconstructed_time_unix_sec());
            // median takes a copy. Account for both buffers, then release
            // cadence scratch before measuring individual coordinates.
            out->peak_scratch_ = std::max(out->peak_scratch_, 2*intervals.size());
            if (!intervals.empty()) {
                n.interval_seconds = rtc_native_spectral_detail::median(intervals);
                n.minimum_interval_seconds = *std::min_element(intervals.begin(), intervals.end());
                n.maximum_interval_seconds = *std::max_element(intervals.begin(), intervals.end());
                n.cadence_available = std::isfinite(n.interval_seconds) && n.interval_seconds > 0;
                n.maximum_fractional_interval_deviation = 0;
                for (double dt : intervals) {
                    n.cadence_available &= std::isfinite(dt) && dt > 0 &&
                        std::abs(dt-domain.nominal_interval_seconds)/domain.nominal_interval_seconds <= domain.maximum_fractional_deviation;
                    n.maximum_fractional_interval_deviation = std::max(n.maximum_fractional_interval_deviation, std::abs(dt-n.interval_seconds)/n.interval_seconds);
                }
            }
            }
            std::vector<double> window;
            if (n.cadence_available) {
                n.fft_samples = std::max(RtcInitialSpectralPolicy::minimum_samples, rtc_native_spectral_detail::rounded(4./n.interval_seconds));
                n.minimum_chunk_samples = std::max(RtcInitialSpectralPolicy::minimum_samples, rtc_native_spectral_detail::rounded(2./n.interval_seconds));
                n.hop_samples = std::max(std::size_t{1}, rtc_native_spectral_detail::rounded(n.fft_samples*.5));
                if (n.fft_samples <= span.occurrence_count()) {
                window.resize(n.fft_samples); double sum = 0, sum2 = 0;
                for (std::size_t i = 0; i < window.size(); ++i) {
                    window[i] = .5 - .5*std::cos(2*std::numbers::pi*static_cast<double>(i)/(window.size()-1));
                    sum += window[i]; sum2 += window[i]*window[i];
                }
                n.window_norm = sum2/n.interval_seconds;
                n.equivalent_noise_bandwidth_hz = n.window_norm/(sum*sum);
                for (std::size_t k = 0; k <= n.fft_samples/2; ++k) n.frequency_hz.push_back(k/(n.fft_samples*n.interval_seconds));
                }
            }
            out->networks_.push_back(n);
            for (std::uint32_t d = 0; d < static_cast<std::uint32_t>(net.detector_count()); ++d)
                for (auto c : {NativeReadoutCoordinate::x, NativeReadoutCoordinate::r})
                    out->spectra_.push_back(out->measure(n, window, d, c));
        }
        return out;
    }
public:
    const auto &original_spike_handle() const noexcept { return original_; }
    const auto &conditioned_handle() const noexcept { return conditioned_; }
    std::uint64_t attempt() const noexcept { return attempt_; }
    auto use_policy() const noexcept { return conditioned_ ? RtcConditionedNativeProduct::use_policy : RtcInitialSpectralPolicy::identity; }
    const auto &networks() const noexcept { return networks_; }
    const auto &spectra() const noexcept { return spectra_; }
    // Conservative bound on visible scalar scratch; FFT internals/allocator are excluded.
    std::size_t peak_scratch_samples() const noexcept { return peak_scratch_; }
    const RtcSpectralNetwork &network(TimestreamNetworkId id) const {
        auto it = std::find_if(networks_.begin(), networks_.end(), [=](const auto &n) { return n.input->support().network_id == id; });
        if (it == networks_.end()) throw std::out_of_range("RTC spectral network absent");
        return *it;
    }
    const RtcNativeSpectrum &spectrum(TimestreamNetworkId n, std::uint32_t d, NativeReadoutCoordinate c) const {
        auto it = std::find_if(spectra_.begin(), spectra_.end(), [=](const auto &s) { return s.network == n && s.detector == d && s.coordinate == c; });
        if (it == spectra_.end()) throw std::out_of_range("RTC spectrum absent");
        return *it;
    }
    std::size_t logical_owned_bytes() const noexcept {
        std::size_t bytes = networks_.size()*sizeof(RtcSpectralNetwork) + spectra_.size()*sizeof(RtcNativeSpectrum);
        for (const auto &n : networks_) bytes += n.frequency_hz.size()*sizeof(double);
        for (const auto &s : spectra_) bytes += s.psd.size()*sizeof(double) + s.windows.size()*sizeof(RtcSpectralWindow) +
            s.runs.size()*sizeof(RtcSpectralRun) + s.centering_support.size()*sizeof(RtcEventRange);
        return bytes;
    }
private:
    RtcNativeSpectralEvidence(std::shared_ptr<const RtcSpikeEvidence> o, std::uint64_t a) : original_{std::move(o)}, attempt_{a} {}
    RtcNativeSpectrum measure(const RtcSpectralNetwork &n, const std::vector<double> &window,
        std::uint32_t detector, NativeReadoutCoordinate coordinate) {
        using namespace rtc_native_spectral_detail;
        RtcNativeSpectrum s; s.network = n.input->support().network_id; s.detector = detector; s.coordinate = coordinate;
        const auto &net = original_->input_handle()->network(s.network); const auto &axis = net.occurrence_axis();
        const auto *derived = conditioned_ ? &conditioned_->column(s.network,detector) : nullptr;
        const auto admitted = [&](auto row) {
            return derived ? (derived->state.at(row-derived->first) & (1U << static_cast<unsigned>(coordinate))) != 0
                           : net.state(coordinate,row,detector).valid();
        };
        const auto value = [&](auto row) {
            return derived ? (*derived->values)(row-derived->first,static_cast<unsigned>(coordinate))
                           : net.value(coordinate,row,detector);
        };
        std::vector<double> population;
        std::size_t longest = 0; bool input_failure = false;
        for (const auto &run : axis.contiguous_runs()) {
            RtcSpectralRun r; r.rows = {run.first_native_row, run.past_last_native_row};
            for (auto row = r.rows.first; row < r.rows.past_last; ++row) {
                if (!admitted(row)) { ++r.declared_invalid_samples; continue; }
                ++r.admitted_samples;
                if (!std::isfinite(value(row))) {
                    ++r.unexpected_nonfinite_samples;
                    if (r.first_unexpected_nonfinite < 0) r.first_unexpected_nonfinite = row;
                }
            }
            if (r.unexpected_nonfinite_samples) { r.cause = RtcSpectralRunCause::input_consistency_failure; input_failure = true; }
            else {
                auto row = r.rows.first;
                while (row < r.rows.past_last) {
                    if (!admitted(row)) { ++row; continue; }
                    auto first = row;
                    while (row < r.rows.past_last && admitted(row)) {
                        population.push_back(value(row)); ++row;
                    }
                    s.centering_support.push_back({first, row}); longest = std::max(longest, static_cast<std::size_t>(row-first));
                }
            }
            s.runs.push_back(r);
        }
        peak_scratch_ = std::max(peak_scratch_, 2*population.size() + 4*n.fft_samples);
        if (!n.cadence_available) { s.cause = RtcSpectralCause::cadence_unavailable; return s; }
        if (population.empty()) { s.cause = input_failure ? RtcSpectralCause::input_consistency_failure : RtcSpectralCause::insufficient_windows; return s; }
        s.population_median = median(std::move(population));
        if (!std::isfinite(s.population_median)) {
            s.cause = RtcSpectralCause::arithmetic_nonfinite;
            for (auto &r : s.runs) if (r.cause != RtcSpectralRunCause::input_consistency_failure) r.cause = RtcSpectralRunCause::arithmetic_nonfinite;
            return s;
        }
        if (longest < n.fft_samples) { s.cause = RtcSpectralCause::fixed_grid_unavailable; return s; }
        std::vector<double> total(n.frequency_hz.size(), 0.);
        Eigen::FFT<double> fft; fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
        for (std::size_t ri = 0; ri < s.runs.size(); ++ri) {
            auto &run = s.runs[ri]; run.first_window = s.windows.size(); run.past_last_window = s.windows.size();
            if (run.cause == RtcSpectralRunCause::input_consistency_failure) continue;
            std::vector<double> sum = total; bool arithmetic_failure = false;
            for (auto stretch : s.centering_support) {
                if (stretch.first < run.rows.first || stretch.past_last > run.rows.past_last ||
                    static_cast<std::size_t>(stretch.past_last-stretch.first) < n.minimum_chunk_samples) continue;
                std::vector<TimestreamNativeRow> starts;
                if (static_cast<std::size_t>(stretch.past_last-stretch.first) < n.fft_samples) starts.push_back(stretch.first);
                else {
                    for (auto first = stretch.first; first <= stretch.past_last-static_cast<TimestreamNativeRow>(n.fft_samples); first += n.hop_samples) starts.push_back(first);
                    if (starts.back() != stretch.past_last-static_cast<TimestreamNativeRow>(n.fft_samples)) starts.push_back(stretch.past_last-n.fft_samples);
                }
                for (auto first : starts) {
                    const auto last = std::min(stretch.past_last, first+static_cast<TimestreamNativeRow>(n.fft_samples));
                    std::vector<double> samples;
                    for (auto row = first; row < last; ++row) samples.push_back(value(row)-s.population_median);
                    const double center = median(samples);
                    RtcSpectralWindow info{ri, {first,last}, axis.occurrence(first).integration_support.begin_unix_sec,
                        axis.occurrence(last-1).integration_support.end_unix_sec, n.fft_samples-samples.size(), center, {}};
                    for (auto row = first; row < last; ++row) ++info.source_counts[source_index(original_->protection_handle()->state(s.network, detector, row))];
                    if (derived) for (auto row = first; row < last; ++row) {
                        info.representative_replacements += (derived->state[row-derived->first] & 4U) != 0;
                        info.replacement_influenced_samples += (derived->state[row-derived->first] & 8U) != 0;
                        info.unrepaired_influenced_samples += (derived->state[row-derived->first] & 16U) != 0;
                        info.representative_exclusions += (derived->state[row-derived->first] & 32U) != 0;
                    }
                    for (auto &value : samples) value -= center;
                    samples.resize(n.fft_samples, 0.);
                    for (std::size_t i = 0; i < samples.size(); ++i) { samples[i] *= window[i]; arithmetic_failure |= !std::isfinite(samples[i]); }
                    if (arithmetic_failure) break;
                    std::vector<std::complex<double>> transformed; fft.fwd(transformed, samples);
                    for (std::size_t k = 0; k < sum.size(); ++k) {
                        double value = std::norm(transformed[k])/n.window_norm;
                        // Exact inherited convention: only [1:-1] doubled, also
                        // for odd N. Do not silently "fix" the accepted estimator.
                        if (sum.size() > 2 && k > 0 && k+1 < sum.size()) value *= 2;
                        sum[k] += value; arithmetic_failure |= !std::isfinite(sum[k]);
                    }
                    if (arithmetic_failure) break;
                    s.windows.push_back(info);
                }
                if (arithmetic_failure) break;
            }
            if (arithmetic_failure) {
                s.windows.resize(run.first_window); run.cause = RtcSpectralRunCause::arithmetic_nonfinite; continue;
            }
            run.past_last_window = s.windows.size();
            if (run.past_last_window > run.first_window) {
                run.cause = RtcSpectralRunCause::contributing;
                total = std::move(sum);
            }
        }
        if (s.windows.size() < RtcInitialSpectralPolicy::minimum_windows) { s.cause = RtcSpectralCause::insufficient_windows; return s; }
        for (auto &v : total) { v /= s.windows.size(); if (!std::isfinite(v)) { s.cause = RtcSpectralCause::arithmetic_nonfinite; return s; } }
        s.psd = std::move(total);
        s.cause = std::any_of(s.runs.begin(), s.runs.end(), [](auto r) { return r.cause != RtcSpectralRunCause::contributing; }) ?
            RtcSpectralCause::available_with_unavailable_runs : RtcSpectralCause::available;
        return s;
    }
    std::shared_ptr<const RtcSpikeEvidence> original_;
    std::shared_ptr<const RtcConditionedNativeProduct> conditioned_;
    std::uint64_t attempt_;
    std::vector<RtcSpectralNetwork> networks_;
    std::vector<RtcNativeSpectrum> spectra_;
    std::size_t peak_scratch_ = 0;
};

// A concrete RTC Consider input product. Retains both evidence generations;
// does not rewrite the older review's spectral-unavailable field or promote a
// candidate. Conditioned spectra retain their numerical product and review use;
// their exact snapshot checks never rewrite the initial evidence.
class RtcSpectralTransientConsideration {
public:
    static std::shared_ptr<const RtcSpectralTransientConsideration> consider(
        std::shared_ptr<const RtcNativeSpectralEvidence> spectra, std::shared_ptr<const ValSnapshot> spectral_snapshot,
        std::shared_ptr<const RtcEventAssessmentDecision> transients, std::shared_ptr<const ValSnapshot> transient_snapshot,
        std::uint64_t attempt) {
        if (!spectra || !transients || !attempt ||
            spectra->original_spike_handle().get() != transients->evidence_handle()->spike_handle().get() ||
            transient_snapshot.get() != transients->evidence_handle()->spike_handle()->val_snapshot_handle().get())
            throw std::invalid_argument("RTC spectral consideration requires exact original transient evidence and VAL");
        for (const auto &n : spectra->networks())
            if (n.input->snapshot_handle().get() != spectral_snapshot.get())
                throw std::invalid_argument("RTC spectral consideration cannot rebind spectral VAL");
        return std::shared_ptr<const RtcSpectralTransientConsideration>(new RtcSpectralTransientConsideration{
            std::move(spectra), std::move(transients), attempt});
    }
    const auto &spectral_handle() const noexcept { return spectral_; }
    const auto &transient_handle() const noexcept { return transient_; }
    std::uint64_t attempt() const noexcept { return attempt_; }
    const RtcNativeSpectrum &event_spectrum(std::size_t event, NativeReadoutCoordinate c) const {
        const auto &e = transient_->evidence_handle()->events().at(event);
        return spectral_->spectrum(e.network, e.detector, c);
    }
    bool event_run_contributes(std::size_t event, NativeReadoutCoordinate c) const {
        const auto &e = transient_->evidence_handle()->events().at(event); const auto &s = event_spectrum(event,c);
        if (!s.available()) return false;
        const auto row = transient_->evidence_handle()->spike_handle()->candidates()[e.seed].later_row;
        return std::any_of(s.runs.begin(), s.runs.end(), [=](auto r) {
            return r.rows.first <= row && row < r.rows.past_last && r.cause == RtcSpectralRunCause::contributing;
        });
    }
    static constexpr bool notch_admitted = false, spike_admitted = false, apply_authorized = false;
private:
    RtcSpectralTransientConsideration(std::shared_ptr<const RtcNativeSpectralEvidence> s,
        std::shared_ptr<const RtcEventAssessmentDecision> t, std::uint64_t a) : spectral_{std::move(s)}, transient_{std::move(t)}, attempt_{a} {}
    std::shared_ptr<const RtcNativeSpectralEvidence> spectral_;
    std::shared_ptr<const RtcEventAssessmentDecision> transient_;
    std::uint64_t attempt_;
};

} // namespace citlali::pipeline
