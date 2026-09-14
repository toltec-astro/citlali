#pragma once
#include <citlali/core/pipeline/timestream_rtc_native_spectral_learn.h>
#include <numeric>
#include <span>

namespace citlali::pipeline {

// Approved descriptive profiles. None is a line-significance or notch policy.
enum class RtcLinePowerProfile { initial_2_hz, sensitivity_1_hz, sensitivity_4_hz };
enum class RtcLinePowerCause { available, spectral_unavailable, arithmetic_unavailable };
struct RtcLineBackgroundNeighborhood {
    std::size_t first_bin = 0, past_last_bin = 0;
    bool clipped_low = false, clipped_high = false;
};
struct RtcPositiveSpectralRegion {
    std::size_t first_bin = 0, past_last_bin = 0, peak_bin = 0;
    double first_frequency_hz = NAN, last_frequency_hz = NAN, bin_span_hz = NAN;
    double positive_excess_power = NAN, stored_psd_power_fraction = NAN, peak_contrast = NAN;
    bool extent_incomplete = false, background_neighborhood_truncated = false;
};
struct RtcLineBandMeasurement {
    std::string identity;
    double requested_low_hz = NAN, requested_high_hz = NAN;
    std::size_t first_bin = 0, past_last_bin = 0;
    double stored_psd_power = NAN, background_power = NAN, signed_residual_power = NAN;
};
struct RtcLinePowerCoordinate {
    TimestreamNetworkId network = -1;
    std::uint32_t detector = 0;
    NativeReadoutCoordinate coordinate = NativeReadoutCoordinate::x;
    RtcLinePowerCause cause = RtcLinePowerCause::spectral_unavailable;
    double bin_increment_hz = NAN, total_stored_psd_power = NAN;
    std::vector<double> background;
    std::vector<RtcLineBackgroundNeighborhood> neighborhoods;
    std::vector<RtcPositiveSpectralRegion> regions;
    // A pooled PSD and overlapping window count do not establish persistence.
    static constexpr bool persistence_measured = false;
    bool available() const noexcept { return cause == RtcLinePowerCause::available; }
};

namespace rtc_line_power_detail {
inline double radius(RtcLinePowerProfile p) {
    switch (p) {
        case RtcLinePowerProfile::initial_2_hz: return 2.;
        case RtcLinePowerProfile::sensitivity_1_hz: return 1.;
        case RtcLinePowerProfile::sensitivity_4_hz: return 4.;
    }
    throw std::invalid_argument("RTC line-power profile is not approved");
}
// Internal arithmetic seam, not an unbound scientific-evidence constructor.
inline RtcLinePowerCoordinate measure(std::span<const double> frequency,
    std::span<const double> psd, double radius_hz) {
    if (frequency.size() < 2 || frequency.size() != psd.size() ||
        !std::isfinite(radius_hz) || radius_hz <= 0)
        throw std::invalid_argument("RTC line-power grid or radius malformed");
    RtcLinePowerCoordinate out;
    const double df = frequency[1] - frequency[0];
    if (!std::isfinite(df) || df <= 0) throw std::invalid_argument("RTC line-power frequency increment invalid");
    long double total = 0;
    for (std::size_t i = 0; i < frequency.size(); ++i) {
        if (!std::isfinite(frequency[i]) || !std::isfinite(psd[i]) || psd[i] < 0 ||
            (i && frequency[i] <= frequency[i-1]) ||
            std::abs((frequency[i]-frequency[0])-i*df) >
                32*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(frequency[i])))
            throw std::invalid_argument("RTC line-power requires finite nonnegative PSD on its unchanged uniform grid");
        total += static_cast<long double>(psd[i])*df;
    }
    out.bin_increment_hz = df; out.total_stored_psd_power = static_cast<double>(total);
    out.background.resize(psd.size()); out.neighborhoods.reserve(psd.size());
    std::vector<double> neighborhood;
    for (std::size_t i = 0; i < psd.size(); ++i) {
        const double low = frequency[i]-radius_hz, high = frequency[i]+radius_hz;
        const auto first = std::lower_bound(frequency.begin(),frequency.end(),low)-frequency.begin();
        const auto last = std::upper_bound(frequency.begin(),frequency.end(),high)-frequency.begin();
        out.neighborhoods.push_back({static_cast<std::size_t>(first),static_cast<std::size_t>(last),
            low < frequency.front(),high > frequency.back()});
        neighborhood.assign(psd.begin()+first,psd.begin()+last);
        std::sort(neighborhood.begin(),neighborhood.end());
        const auto m = neighborhood.size()/2;
        out.background[i] = neighborhood.size()%2 ? neighborhood[m] : std::midpoint(neighborhood[m-1],neighborhood[m]);
    }
    if (!std::isfinite(out.total_stored_psd_power)) {
        out.cause = RtcLinePowerCause::arithmetic_unavailable; return out;
    }
    for (std::size_t i = 0; i < psd.size();) {
        if (psd[i] <= out.background[i]) { ++i; continue; }
        RtcPositiveSpectralRegion region; region.first_bin = region.peak_bin = i;
        long double excess = 0;
        for (; i < psd.size() && psd[i] > out.background[i]; ++i) {
            excess += static_cast<long double>(psd[i]-out.background[i])*df;
            if (psd[i]-out.background[i] > psd[region.peak_bin]-out.background[region.peak_bin]) region.peak_bin = i;
            region.background_neighborhood_truncated |= out.neighborhoods[i].clipped_low || out.neighborhoods[i].clipped_high;
        }
        region.past_last_bin = i; region.first_frequency_hz = frequency[region.first_bin];
        region.last_frequency_hz = frequency[i-1]; region.bin_span_hz = (i-region.first_bin)*df;
        region.extent_incomplete = region.first_bin == 0 || i == psd.size();
        region.positive_excess_power = static_cast<double>(excess);
        region.stored_psd_power_fraction = out.total_stored_psd_power > 0 ? region.positive_excess_power/out.total_stored_psd_power : NAN;
        if (out.background[region.peak_bin] > 0) {
            const double contrast = psd[region.peak_bin]/out.background[region.peak_bin];
            if (std::isfinite(contrast)) region.peak_contrast = contrast;
        }
        if (!std::isfinite(region.positive_excess_power)) {
            out.regions.clear(); out.cause = RtcLinePowerCause::arithmetic_unavailable; return out;
        }
        out.regions.push_back(region);
    }
    out.cause = RtcLinePowerCause::available; return out;
}
} // namespace rtc_line_power_detail

class RtcLinePowerEvidence {
public:
    static std::shared_ptr<const RtcLinePowerEvidence> learn(
        std::shared_ptr<const RtcNativeSpectralEvidence> input,
        std::shared_ptr<const ValSnapshot> snapshot, RtcLinePowerProfile profile, std::uint64_t attempt) {
        if (!input || !snapshot || !attempt) throw std::invalid_argument("RTC line-power requires exact spectral evidence, VAL and attempt");
        const double radius = rtc_line_power_detail::radius(profile);
        for (const auto &network : input->networks())
            if (network.input->snapshot_handle().get() != snapshot.get())
                throw std::invalid_argument("RTC line-power cannot rebind spectral VAL");
        auto out = std::shared_ptr<RtcLinePowerEvidence>(new RtcLinePowerEvidence{input,snapshot,profile,attempt});
        for (const auto &s : input->spectra()) {
            RtcLinePowerCoordinate value;
            if (s.available()) value = rtc_line_power_detail::measure(input->network(s.network).frequency_hz,s.psd,radius);
            value.network = s.network; value.detector = s.detector; value.coordinate = s.coordinate;
            out->coordinates_.push_back(std::move(value));
        }
        return out;
    }
    const auto &spectral_handle() const noexcept { return input_; }
    const auto &snapshot_handle() const noexcept { return snapshot_; }
    const auto &coordinates() const noexcept { return coordinates_; }
    auto profile() const noexcept { return profile_; }
    double radius_hz() const { return rtc_line_power_detail::radius(profile_); }
    std::uint64_t attempt() const noexcept { return attempt_; }
    static constexpr std::string_view measurement_identity = "rtc-native-line-positive-excess-v1";
    static constexpr std::string_view power_convention = "sum-stored-one-sided-PSD-times-df;all-native-bins;no-extra-endpoint-weights";
    const RtcLinePowerCoordinate &coordinate(TimestreamNetworkId n, std::uint32_t d, NativeReadoutCoordinate c) const {
        auto it = std::find_if(coordinates_.begin(),coordinates_.end(),[=](const auto &s){return s.network==n && s.detector==d && s.coordinate==c;});
        if (it == coordinates_.end()) throw std::out_of_range("RTC line-power coordinate absent");
        return *it;
    }
    // Caller supplies a named band independently of discovered regions. Bin
    // centers on either boundary are included; no fractional-bin interpolation.
    RtcLineBandMeasurement measure_band(TimestreamNetworkId n, std::uint32_t d, NativeReadoutCoordinate c,
        std::string identity, double low_hz, double high_hz) const {
        const auto &s = coordinate(n,d,c); const auto &f = input_->network(n).frequency_hz;
        if (!s.available() || identity.empty() || !std::isfinite(low_hz) || !std::isfinite(high_hz) ||
            low_hz > high_hz || low_hz < f.front() || high_hz > f.back())
            throw std::invalid_argument("RTC signed-band measurement requires available evidence and explicit in-grid band");
        RtcLineBandMeasurement out; out.identity=std::move(identity);out.requested_low_hz=low_hz;out.requested_high_hz=high_hz;
        out.first_bin=std::lower_bound(f.begin(),f.end(),low_hz)-f.begin();
        out.past_last_bin=std::upper_bound(f.begin(),f.end(),high_hz)-f.begin();
        if(out.first_bin==out.past_last_bin)throw std::invalid_argument("RTC signed band has no frequency bins");
        const auto &psd=input_->spectrum(n,d,c).psd;long double power=0,background=0,residual=0;
        for(auto i=out.first_bin;i<out.past_last_bin;++i){
            power+=static_cast<long double>(psd[i])*s.bin_increment_hz;
            background+=static_cast<long double>(s.background[i])*s.bin_increment_hz;
            residual+=static_cast<long double>(psd[i]-s.background[i])*s.bin_increment_hz;
        }
        out.stored_psd_power=static_cast<double>(power);out.background_power=static_cast<double>(background);out.signed_residual_power=static_cast<double>(residual);
        if(!std::isfinite(out.stored_psd_power)||!std::isfinite(out.background_power)||!std::isfinite(out.signed_residual_power))
            throw std::overflow_error("RTC signed-band power arithmetic unavailable");
        return out;
    }
    std::size_t logical_owned_bytes() const noexcept {
        std::size_t n=coordinates_.size()*sizeof(RtcLinePowerCoordinate);
        for(const auto &c:coordinates_)n+=c.background.size()*sizeof(double)+c.neighborhoods.size()*sizeof(RtcLineBackgroundNeighborhood)+c.regions.size()*sizeof(RtcPositiveSpectralRegion);
        return n;
    }
private:
    RtcLinePowerEvidence(std::shared_ptr<const RtcNativeSpectralEvidence> i,std::shared_ptr<const ValSnapshot> s,RtcLinePowerProfile p,std::uint64_t a)
        :input_{std::move(i)},snapshot_{std::move(s)},profile_{p},attempt_{a}{}
    std::shared_ptr<const RtcNativeSpectralEvidence> input_;
    std::shared_ptr<const ValSnapshot> snapshot_;
    RtcLinePowerProfile profile_;
    std::uint64_t attempt_;
    std::vector<RtcLinePowerCoordinate> coordinates_;
};

// Diagnostic ranking only; x/r powers have distinct units and are never pooled.
class RtcLinePowerConsideration {
public:
    static std::shared_ptr<const RtcLinePowerConsideration> rank(
        std::shared_ptr<const RtcLinePowerEvidence> lines,
        std::shared_ptr<const RtcSpectralTransientConsideration> joint,std::uint64_t attempt) {
        if(!lines||!joint||!attempt||lines->spectral_handle().get()!=joint->spectral_handle().get())
            throw std::invalid_argument("RTC line ranking requires exact joint spectral/transient evidence");
        auto out=std::shared_ptr<RtcLinePowerConsideration>(new RtcLinePowerConsideration{lines,joint,attempt});
        for(const auto &c:lines->coordinates()){
            std::vector<std::size_t> order(c.available()?c.regions.size():0);std::iota(order.begin(),order.end(),0);
            std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return c.regions[a].positive_excess_power>c.regions[b].positive_excess_power;});
            out->ranks_.push_back(std::move(order));
        }
        return out;
    }
    const auto &line_handle() const noexcept {return lines_;}
    const auto &joint_handle() const noexcept {return joint_;}
    const auto &ranks() const noexcept {return ranks_;} // aligned with coordinates()
    std::uint64_t attempt() const noexcept {return attempt_;}
    static constexpr bool shared_notch_requires_direct_x = true;
    static constexpr bool interference_admitted = false, notch_proposed = false, apply_authorized = false;
private:
    RtcLinePowerConsideration(std::shared_ptr<const RtcLinePowerEvidence> l,std::shared_ptr<const RtcSpectralTransientConsideration> j,std::uint64_t a)
        :lines_{std::move(l)},joint_{std::move(j)},attempt_{a}{}
    std::shared_ptr<const RtcLinePowerEvidence> lines_;
    std::shared_ptr<const RtcSpectralTransientConsideration> joint_;
    std::uint64_t attempt_;
    std::vector<std::vector<std::size_t>> ranks_;
};
} // namespace citlali::pipeline
