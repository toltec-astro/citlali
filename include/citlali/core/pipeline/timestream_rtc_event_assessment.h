#pragma once

#include <citlali/core/pipeline/timestream_rtc_event_background.h>
#include <map>
#include <tuple>

namespace citlali::pipeline {

// Owner-approved continuation of RTC-EVENT-BACKGROUND-001, 2026-09-09.
// These are review/event-assessment policies, not scientific-use rejection.
struct RtcEventAssessmentPolicy {
    static constexpr std::string_view identity = "rtc-event-assessment-2026-09-09-v1";
    static constexpr double trial_half_width_seconds = 0.05;
    static constexpr double recovery_seconds = 0.05;
    static constexpr double recovery_sigma = 4.0;
    static constexpr double search_seconds = 2.0;
    static constexpr double health_scale_ratio = 10.0;
    static constexpr double health_edge_fraction = 0.01;
    static constexpr double health_block_fraction = 0.8;
    static constexpr std::size_t health_minimum_blocks = 6;
    static constexpr std::size_t maximum_support_refinements = 8;
};

struct RtcEventPeerEligibility {
    TimestreamNetworkId network = -1;
    std::uint32_t detector = 0;
    std::string detector_occurrence_id;
    bool eligible = false;
};

// Eligibility is supplied by the caller's named population authority. RTC
// validates exact original parentage and does not infer it from channel order.
class RtcEventPeerPopulation {
public:
    static std::shared_ptr<const RtcEventPeerPopulation> admit(
        std::shared_ptr<const RtcSpikeEvidence> spikes, std::string source_identity,
        std::vector<RtcEventPeerEligibility> records) {
        if (!spikes || source_identity.empty())
            throw std::invalid_argument("RTC peer population requires exact spike parent and source identity");
        auto p = std::shared_ptr<RtcEventPeerPopulation>(new RtcEventPeerPopulation);
        p->spikes_ = std::move(spikes); p->identity_ = std::move(source_identity);
        for (const auto &span : p->spikes_->input_handle()->spans()) {
            const auto &net = p->spikes_->input_handle()->network(span.network_id);
            if (span.first_native_row != net.occurrence_axis().first_native_row() ||
                span.past_last_native_row != net.occurrence_axis().past_last_native_row())
                throw std::invalid_argument("RTC event assessment requires complete native observation support");
            p->eligible_[span.network_id].resize(net.detectors().size());
        }
        std::set<std::pair<TimestreamNetworkId,std::uint32_t>> seen;
        for (const auto &r : records) {
            const auto it = p->eligible_.find(r.network);
            if (it == p->eligible_.end() || r.detector >= it->second.size() ||
                !seen.emplace(r.network,r.detector).second ||
                p->spikes_->input_handle()->network(r.network).detectors()[r.detector].detector_occurrence_id != r.detector_occurrence_id)
                throw std::invalid_argument("RTC peer population has mismatched/repeated detector identity");
            it->second[r.detector] = r.eligible;
        }
        std::size_t count = 0; for (const auto &[n, v] : p->eligible_) count += v.size();
        if (seen.size() != count) throw std::invalid_argument("RTC peer population is incomplete");
        return p;
    }
    const auto &spike_handle() const noexcept { return spikes_; }
    const auto &source_identity() const noexcept { return identity_; }
    bool eligible(TimestreamNetworkId n, std::uint32_t d) const { return eligible_.at(n).at(d); }
private:
    RtcEventPeerPopulation() = default;
    std::shared_ptr<const RtcSpikeEvidence> spikes_;
    std::string identity_;
    std::map<TimestreamNetworkId,std::vector<bool>> eligible_;
};

struct RtcEventRange {
    TimestreamNativeRow first = -1, past_last = -1;
    bool present() const noexcept { return first >= 0 && past_last > first; }
    friend bool operator==(const RtcEventRange &, const RtcEventRange &) = default;
};

enum class RtcEventRecoveryCause : std::uint8_t {
    recovered, background_unavailable, onset_unavailable, search_limit,
    observation_end, acquisition_gap, invalid_support, nonfinite
};

struct RtcEventRecovery {
    RtcEventRecoveryCause cause = RtcEventRecoveryCause::background_unavailable;
    RtcEventRange affected;
    RtcEventRange confirmation;
    RtcEventRange examined;
    bool recovered() const noexcept { return cause == RtcEventRecoveryCause::recovered; }
};

struct RtcEventPeerContext {
    std::size_t eligible_peers = 0, usable_peers = 0;
    std::uint32_t strongest_peer = 0;
    std::size_t strongest_shared_samples = 0;
    double strongest_level_correlation = std::numeric_limits<double>::quiet_NaN();
    double strongest_difference_correlation = std::numeric_limits<double>::quiet_NaN();
    double strongest_edge_delay_seconds = std::numeric_limits<double>::quiet_NaN();
    // No hard coherence threshold or causal/atmospheric diagnosis.
};

struct RtcEventHealthBlock {
    std::size_t noise_block_index = 0;
    bool complete = false;
    std::array<std::size_t,2> edges{}, peer_count{};
    std::array<double,2> peer_median_scale{NAN,NAN}, scale_ratio{NAN,NAN}, edge_fraction{NAN,NAN};
};

struct RtcAssessedEvent {
    TimestreamNetworkId network = -1;
    std::uint32_t detector = 0;
    std::size_t seed = 0;
    std::vector<std::size_t> candidates;
    RtcEventRange trial_exclusion;
    std::vector<RtcEventRange> neighbor_exclusions;
    std::array<std::size_t,2> excluded_neighbor_samples{};
    std::array<RtcEventCoordinateBackground,2> background;
    std::array<RtcEventRecovery,2> recovery;
    std::array<RtcEventPeerContext,2> peers;
    std::array<bool,2> seeded{};
    double origin = 0, time_scale = 1;
    bool observation_truncated = false, gap_truncated = false;
    bool refinement_limited = false;
    std::size_t peak_scratch_rows = 0;
};

namespace rtc_event_assessment_detail {
using Key = std::pair<TimestreamNetworkId,std::uint32_t>;
using BlockKey = std::tuple<TimestreamNetworkId,TimestreamNativeRow,std::uint64_t>;
inline NativeReadoutCoordinate coord(std::size_t c) { return c==0 ? NativeReadoutCoordinate::x : NativeReadoutCoordinate::r; }
inline double time(const auto &axis, TimestreamNativeRow row) {
    return axis.native_identity(row).reconstructed_time_unix_sec();
}
inline TimestreamNativeRow lower(const auto &axis, RtcEventRange run, double t) {
    auto a=run.first,b=run.past_last;
    while(a<b) { const auto m=a+(b-a)/2; if(time(axis,m)<t) a=m+1; else b=m; }
    return a;
}
inline RtcEventRange run_for(const auto &axis, TimestreamNativeRow row) {
    for (const auto &r : axis.contiguous_runs())
        if (r.first_native_row<=row && row<r.past_last_native_row) return {r.first_native_row,r.past_last_native_row};
    throw std::invalid_argument("RTC event is outside physical native runs");
}
inline RtcEventRange trial(const auto &axis, RtcEventRange run, const RtcSpikeCandidate &s) {
    const double center=std::midpoint(time(axis,s.earlier_row),time(axis,s.later_row));
    auto a=s.earlier_row,b=s.later_row+1;
    while(a>run.first && axis.occurrence(a-1).integration_support.end_unix_sec>center-RtcEventAssessmentPolicy::trial_half_width_seconds) --a;
    while(b<run.past_last && axis.occurrence(b).integration_support.begin_unix_sec<center+RtcEventAssessmentPolicy::trial_half_width_seconds) ++b;
    return {a,b};
}
inline bool contains(const std::vector<RtcEventRange> &ranges, TimestreamNativeRow row) {
    const auto it=std::upper_bound(ranges.begin(),ranges.end(),row,[](auto r,const auto &x){return r<x.first;});
    return it!=ranges.begin() && row<std::prev(it)->past_last;
}
inline std::vector<RtcEventRange> merge(std::vector<RtcEventRange> ranges) {
    std::sort(ranges.begin(),ranges.end(),[](auto a,auto b){return std::tie(a.first,a.past_last)<std::tie(b.first,b.past_last);});
    std::vector<RtcEventRange> out;
    for(auto r:ranges) { if(!out.empty() && r.first<=out.back().past_last) out.back().past_last=std::max(out.back().past_last,r.past_last); else out.push_back(r); }
    return out;
}
inline double polynomial(const RtcEventCubicFit &fit,double u) {
    return ((fit.coefficients[3]*u+fit.coefficients[2])*u+fit.coefficients[1])*u+fit.coefficients[0];
}

inline void background(const RtcSpikeEvidence &spikes, const std::vector<std::size_t> &indices,
                       RtcAssessedEvent &e, RtcEventRange run) {
    const auto &net=spikes.input_handle()->network(e.network); const auto &axis=net.occurrence_axis();
    const auto &request=e.trial_exclusion;
    double begin=INFINITY,end=-INFINITY;
    for(auto row=request.first;row<request.past_last;++row) {
        const auto &s=axis.occurrence(row).integration_support;
        begin=std::min(begin,s.begin_unix_sec); end=std::max(end,s.end_unix_sec);
    }
    const double low=begin-RtcEventBackgroundPolicy::flank_seconds,high=end+RtcEventBackgroundPolicy::flank_seconds;
    e.origin=std::midpoint(begin,end); e.time_scale=std::max(e.origin-low,high-e.origin);
    const bool pre=axis.occurrence(run.first).integration_support.begin_unix_sec>low;
    const bool post=axis.occurrence(run.past_last-1).integration_support.end_unix_sec<high;
    e.observation_truncated=(pre && run.first==axis.first_native_row()) || (post && run.past_last==axis.past_last_native_row());
    e.gap_truncated=(pre && run.first!=axis.first_native_row()) || (post && run.past_last!=axis.past_last_native_row());
    const auto first=lower(axis,run,low),last=lower(axis,run,high);
    std::vector<RtcEventRange> masks;
    auto at=std::lower_bound(indices.begin(),indices.end(),first-1,[&](auto i,auto row){return spikes.candidates()[i].earlier_row<row;});
    // Include a neighbor's guard when its edge lies just outside the flank.
    const auto guard_first=lower(axis,run,low-RtcEventAssessmentPolicy::trial_half_width_seconds);
    at=std::lower_bound(indices.begin(),indices.end(),guard_first-1,[&](auto i,auto row){return spikes.candidates()[i].earlier_row<row;});
    for(;at!=indices.end();++at) {
        const auto &s=spikes.candidates()[*at];
        if(s.earlier_row>=run.past_last || time(axis,s.earlier_row)>high+RtcEventAssessmentPolicy::trial_half_width_seconds) break;
        if(s.earlier_row<run.first) continue;
        masks.push_back(trial(axis,run,s));
    }
    e.neighbor_exclusions=merge(std::move(masks));
    for(std::size_t c=0;c<2;++c) {
        auto &out=e.background[c]; out={};out.support_cause=RtcEventFitCause::none;
        e.excluded_neighbor_samples[c]=0;
        std::vector<std::array<double,3>> samples;
        for(auto row=first;row<last;++row) {
            const auto &s=axis.occurrence(row).integration_support;
            const int side=row<request.first && s.begin_unix_sec>=low && s.end_unix_sec<=begin ? 0 :
                row>=request.past_last && s.begin_unix_sec>=end && s.end_unix_sec<=high ? 1 : -1;
            if(side<0) continue;
            if(contains(e.neighbor_exclusions,row)) {++e.excluded_neighbor_samples[c];continue;}
            auto &sup=out.support[side];
            if(!net.state(coord(c),row,e.detector).valid()) {++sup.invalid;continue;}
            const double y=net.value(coord(c),row,e.detector);
            if(!std::isfinite(y)) {out.support_cause=RtcEventFitCause::nonfinite;continue;}
            if(sup.usable==0) {sup.first_used=row;sup.begin_unix_sec=s.begin_unix_sec;sup.end_unix_sec=s.end_unix_sec;}
            ++sup.usable;sup.last_used=row;sup.begin_unix_sec=std::min(sup.begin_unix_sec,s.begin_unix_sec);
            sup.end_unix_sec=std::max(sup.end_unix_sec,s.end_unix_sec);
            samples.push_back({(time(axis,row)-e.origin)/e.time_scale,y,static_cast<double>(side)});
        }
        e.peak_scratch_rows=std::max(e.peak_scratch_rows,samples.size());
        if(out.support_cause!=RtcEventFitCause::none) continue;
        if(out.support[0].usable<RtcEventBackgroundPolicy::minimum_samples || out.support[1].usable<RtcEventBackgroundPolicy::minimum_samples) {out.support_cause=RtcEventFitCause::insufficient_samples;continue;}
        Eigen::MatrixXd a(samples.size(),5); Eigen::VectorXd y(samples.size());
        for(std::size_t i=0;i<samples.size();++i) {auto s=samples[i];a.row(i)<<1,s[0],s[0]*s[0],s[0]*s[0]*s[0],s[2];y[i]=s[1];}
        const auto n=static_cast<Eigen::Index>(out.support[0].usable);
        out.pre_scale_fit=rtc_event_background_detail::fit(a.topLeftCorner(n,4),y.head(n));
        if(!out.pre_scale_fit.available()) continue;
        out.cubic=rtc_event_background_detail::fit(a.leftCols(4),y,out.pre_scale_fit.scale);
        out.cubic_with_offset=rtc_event_background_detail::fit(a,y,out.pre_scale_fit.scale);
    }
}

inline RtcEventRecovery recover(const RtcSpikeEvidence &spikes,const RtcAssessedEvent &e,
                                RtcEventRange run,std::size_t c) {
    RtcEventRecovery r; const auto &fit=e.background[c];
    if(!fit.available()) return r;
    const auto &net=spikes.input_handle()->network(e.network);const auto &axis=net.occurrence_axis();
    const auto &seed=spikes.candidates()[e.seed];
    const double center=std::midpoint(time(axis,seed.earlier_row),time(axis,seed.later_row));
    const double deadline=center+RtcEventAssessmentPolicy::search_seconds;
    const auto first=lower(axis,run,center-2),last=lower(axis,run,deadline);
    r.examined={first,last};
    auto good=[&](auto row) {
        if(!net.state(coord(c),row,e.detector).valid()) return false;
        const double y=net.value(coord(c),row,e.detector);
        // The additive post-side offset is deliberately NOT in this reference.
        const double predicted=polynomial(fit.cubic_with_offset,(time(axis,row)-e.origin)/e.time_scale);
        return std::isfinite(y) && std::isfinite(predicted) && std::abs(y-predicted)<=RtcEventAssessmentPolicy::recovery_sigma*fit.pre_scale_fit.scale;
    };
    TimestreamNativeRow quiet=-1,onset=-1;
    double support_end=-INFINITY;
    auto accumulate=[&](auto row) {
        const auto &s=axis.occurrence(row).integration_support;
        if(quiet<0 || s.begin_unix_sec>support_end+8*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(s.begin_unix_sec))) quiet=row;
        support_end=std::max(support_end,s.end_unix_sec);
        return support_end-axis.occurrence(quiet).integration_support.begin_unix_sec>=RtcEventAssessmentPolicy::recovery_seconds;
    };
    for(auto row=first;row<seed.later_row;++row) {
        if(!good(row)) {quiet=-1;support_end=-INFINITY;continue;}
        if(accumulate(row)) onset=row+1;
    }
    if(onset<0) {r.cause=RtcEventRecoveryCause::onset_unavailable;return r;}
    quiet=-1;support_end=-INFINITY;bool invalid=false,nonfinite=false;
    for(auto row=seed.later_row;row<last;++row) {
        const auto &s=axis.occurrence(row).integration_support;
        if(s.end_unix_sec>deadline) break;
        if(!net.state(coord(c),row,e.detector).valid()) invalid=true;
        else if(!std::isfinite(net.value(coord(c),row,e.detector))) {r.cause=RtcEventRecoveryCause::nonfinite;return r;}
        if(!good(row)) {quiet=-1;support_end=-INFINITY;continue;}
        if(accumulate(row)) {
            r.confirmation={quiet,row+1};
            // A seed can be large in only the other coordinate. No invented
            // affected cells in a coordinate that never left its background.
            r.affected=quiet>onset ? RtcEventRange{onset,quiet} : RtcEventRange{};
            r.cause=RtcEventRecoveryCause::recovered;return r;
        }
    }
    r.affected={onset,last};
    if(nonfinite) r.cause=RtcEventRecoveryCause::nonfinite;
    else if(invalid) r.cause=RtcEventRecoveryCause::invalid_support;
    else if(axis.occurrence(run.past_last-1).integration_support.end_unix_sec<deadline)
        r.cause=run.past_last==axis.past_last_native_row() ? RtcEventRecoveryCause::observation_end : RtcEventRecoveryCause::acquisition_gap;
    else r.cause=RtcEventRecoveryCause::search_limit;
    return r;
}

// Stable centered sums; no clipping, smoothing, resampling or lag optimization.
struct Correlation {
    std::size_t n=0;double ax=0,ay=0,xx=0,yy=0,xy=0;
    void add(double x,double y) {++n;const double dx=x-ax,dy=y-ay;ax+=dx/n;ay+=dy/n;xx+=dx*(x-ax);yy+=dy*(y-ay);xy+=dx*(y-ay);}
    double value() const {return n>=2 && xx>0 && yy>0 ? xy/std::sqrt(xx*yy) : NAN;}
};
inline void peers(const RtcSpikeEvidence &spikes,const RtcEventPeerPopulation &population,RtcAssessedEvent &e,RtcEventRange run) {
    const auto &net=spikes.input_handle()->network(e.network);const auto &axis=net.occurrence_axis();
    const auto &seed=spikes.candidates()[e.seed];const double center=std::midpoint(time(axis,seed.earlier_row),time(axis,seed.later_row));
    const auto first=lower(axis,run,center-2),last=lower(axis,run,center+2);
    for(std::size_t c=0;c<2;++c) {
        auto &out=e.peers[c];
        for(std::uint32_t d=0;d<net.detectors().size();++d) {
            if(d==e.detector || !population.eligible(e.network,d)) continue;
            ++out.eligible_peers;Correlation levels,diffs; bool previous=false;
            double px=0,py=0,max_delta=-1,edge_time=NAN;bool nonfinite=false;
            for(auto row=first;row<last;++row) {
                if(!net.state(coord(c),row,d).valid() || !net.state(coord(c),row,e.detector).valid()) {previous=false;continue;}
                const double x=net.value(coord(c),row,e.detector),y=net.value(coord(c),row,d);
                if(!std::isfinite(x) || !std::isfinite(y)) {nonfinite=true;break;}
                levels.add(x,y);
                if(previous) {diffs.add(x-px,y-py);if(std::abs(y-py)>max_delta) {max_delta=std::abs(y-py);edge_time=std::midpoint(time(axis,row-1),time(axis,row))-center;}}
                previous=true;px=x;py=y;
            }
            const double correlation=levels.value();
            if(nonfinite || !std::isfinite(correlation)) continue;
            ++out.usable_peers;
            if(!std::isfinite(out.strongest_level_correlation) || std::abs(correlation)>std::abs(out.strongest_level_correlation)) {
                out.strongest_peer=d;out.strongest_shared_samples=levels.n;out.strongest_level_correlation=correlation;
                out.strongest_difference_correlation=diffs.value();out.strongest_edge_delay_seconds=edge_time;
            }
        }
    }
}
} // namespace rtc_event_assessment_detail

class RtcEventAssessmentEvidence;
std::shared_ptr<const RtcEventAssessmentEvidence> learn_rtc_event_assessment(
    std::shared_ptr<const RtcSpikeEvidence>,std::shared_ptr<const RtcEventPeerPopulation>,std::uint64_t);

class RtcEventAssessmentEvidence {
public:
    const auto &spike_handle() const noexcept {return spikes_;}
    const auto &population_handle() const noexcept {return population_;}
    const auto &events() const noexcept {return events_;}
    const auto &health_blocks() const noexcept {return health_;}
    const auto &candidate_peer_context() const noexcept {return candidate_peers_;}
    std::uint64_t attempt() const noexcept {return attempt_;}
private:
    RtcEventAssessmentEvidence() = default;
    friend std::shared_ptr<const RtcEventAssessmentEvidence> learn_rtc_event_assessment(
        std::shared_ptr<const RtcSpikeEvidence>,std::shared_ptr<const RtcEventPeerPopulation>,std::uint64_t);
    std::shared_ptr<const RtcSpikeEvidence> spikes_;
    std::shared_ptr<const RtcEventPeerPopulation> population_;
    std::vector<RtcAssessedEvent> events_;
    std::vector<RtcEventHealthBlock> health_;
    std::vector<std::array<RtcEventPeerContext,2>> candidate_peers_;
    std::uint64_t attempt_=0;
};

inline std::shared_ptr<const RtcEventAssessmentEvidence> learn_rtc_event_assessment(
    std::shared_ptr<const RtcSpikeEvidence> spikes,std::shared_ptr<const RtcEventPeerPopulation> population,std::uint64_t attempt) {
    using namespace rtc_event_assessment_detail;
    if(!spikes || !population || population->spike_handle().get()!=spikes.get() || attempt==0)
        throw std::invalid_argument("RTC assessment requires exact evidence/population handles and attempt");
    auto e=std::shared_ptr<RtcEventAssessmentEvidence>(new RtcEventAssessmentEvidence);
    e->spikes_=spikes;e->population_=population;e->attempt_=attempt;
    std::map<Key,std::vector<std::size_t>> indices;
    std::map<BlockKey,std::vector<std::size_t>> contemporaries;
    std::map<TimestreamNetworkId,std::vector<NativeContiguousRun>> runs;
    for(const auto &span:spikes->input_handle()->spans()) runs[span.network_id]=spikes->input_handle()->network(span.network_id).occurrence_axis().contiguous_runs();
    auto find_run=[&](TimestreamNetworkId network,TimestreamNativeRow row) {
        for(const auto &r:runs.at(network)) if(r.first_native_row<=row && row<r.past_last_native_row) return RtcEventRange{r.first_native_row,r.past_last_native_row};
        throw std::invalid_argument("RTC assessment run not found");
    };
    e->health_.resize(spikes->blocks().size());
    e->candidate_peers_.resize(spikes->candidates().size());
    for(std::size_t i=0;i<spikes->blocks().size();++i) {
        const auto &b=spikes->blocks()[i];auto &h=e->health_[i];h.noise_block_index=i;
        const auto &axis=spikes->input_handle()->network(b.network_id).occurrence_axis();const auto run=find_run(b.network_id,b.first);
        h.complete=axis.occurrence(run.past_last-1).integration_support.end_unix_sec>=b.anchor_unix_sec+10*(b.time_block_index+1);
        contemporaries[{b.network_id,b.run_first,b.time_block_index}].push_back(i);
    }
    for(std::size_t i=0;i<spikes->candidates().size();++i) {
        const auto &s=spikes->candidates()[i];const auto &b=spikes->blocks()[s.noise_block_index];
        indices[{b.network_id,b.detector_index}].push_back(i);
        ++e->health_[s.noise_block_index].edges[s.coordinate==NativeReadoutCoordinate::x?0:1];
    }
    for(auto &[key,group]:contemporaries) for(auto i:group) {
        const auto &b=spikes->blocks()[i];auto &h=e->health_[i];
        for(std::size_t c=0;c<2;++c) {
            std::vector<double> scales;
            for(auto j:group) {const auto &p=spikes->blocks()[j];if(p.detector_index!=b.detector_index && population->eligible(p.network_id,p.detector_index) && p.coordinates[c].available()) scales.push_back(p.coordinates[c].scale);}
            h.peer_count[c]=scales.size();
            if(!scales.empty()) h.peer_median_scale[c]=rtc_spike_detail::median(scales);
            if(b.coordinates[c].available() && h.peer_median_scale[c]>0) h.scale_ratio[c]=b.coordinates[c].scale/h.peer_median_scale[c];
            if(b.coordinates[c].available() && b.coordinates[c].admitted_differences>0) h.edge_fraction[c]=static_cast<double>(h.edges[c])/b.coordinates[c].admitted_differences;
        }
    }
    for(auto &[key,list]:indices) {
        std::sort(list.begin(),list.end(),[&](auto a,auto b){const auto &x=spikes->candidates()[a],&y=spikes->candidates()[b];return std::tie(x.earlier_row,x.later_row,x.coordinate,a)<std::tie(y.earlier_row,y.later_row,y.coordinate,b);});
        const auto &axis=spikes->input_handle()->network(key.first).occurrence_axis();
        // Every original candidate gets context centered on its own edge.
        // x/r candidates at the identical edge share the same calculation.
        TimestreamNativeRow previous_row=-1;
        std::array<RtcEventPeerContext,2> previous_context;
        for(auto candidate:list) {
            const auto &s=spikes->candidates()[candidate];
            if(s.earlier_row!=previous_row) {
                RtcAssessedEvent item;item.network=key.first;item.detector=key.second;item.seed=candidate;
                peers(*spikes,*population,item,find_run(key.first,s.earlier_row));
                previous_context=item.peers;previous_row=s.earlier_row;
            }
            e->candidate_peers_[candidate]=previous_context;
        }
        std::size_t cursor=0;
        while(cursor<list.size()) {
            RtcAssessedEvent out;out.network=key.first;out.detector=key.second;out.seed=list[cursor];
            const auto &seed=spikes->candidates()[out.seed];const auto run=find_run(key.first,seed.earlier_row);
            out.trial_exclusion=trial(axis,run,seed);
            std::size_t next=cursor;
            for(std::size_t refinement=0;refinement<RtcEventAssessmentPolicy::maximum_support_refinements;++refinement) {
                background(*spikes,list,out,run);
                for(std::size_t c=0;c<2;++c) out.recovery[c]=recover(*spikes,out,run,c);
                // Recompute membership after every support refinement. A
                // numerical iteration limit never becomes confirmed support.
                out.candidates.clear();out.seeded={};
                auto boundary=seed.later_row+1;next=cursor;
                do {
                    const auto &s=spikes->candidates()[list[next]];
                    if(s.earlier_row>=run.past_last || (next>cursor && s.earlier_row>=boundary)) break;
                    out.candidates.push_back(list[next]);const auto c=s.coordinate==NativeReadoutCoordinate::x?0U:1U;out.seeded[c]=true;
                    const auto &r=out.recovery[c];
                    if(r.recovered()) boundary=std::max(boundary,r.confirmation.first);
                    else if(r.affected.present()) boundary=std::max(boundary,r.affected.past_last);
                    else boundary=std::max(boundary,out.trial_exclusion.past_last);
                    ++next;
                } while(next<list.size());
                auto expanded=out.trial_exclusion;
                for(std::size_t c=0;c<2;++c) if(out.seeded[c] && out.recovery[c].recovered() && out.recovery[c].affected.present()) {
                    expanded.first=std::min(expanded.first,out.recovery[c].affected.first);
                    expanded.past_last=std::max(expanded.past_last,out.recovery[c].affected.past_last);
                }
                if(expanded==out.trial_exclusion) break;
                if(refinement+1==RtcEventAssessmentPolicy::maximum_support_refinements) {out.refinement_limited=true;break;}
                out.trial_exclusion=expanded;
            }
            out.peers=e->candidate_peers_[out.seed];
            e->events_.push_back(std::move(out));cursor=next;
        }
    }
    return e;
}

struct RtcDetectorHealthReview {
    TimestreamNetworkId network=-1;std::uint32_t detector=0;
    std::size_t complete_blocks=0;
    std::array<std::size_t,2> available_blocks{},concerning_blocks{};
    std::array<bool,2> coordinate_concern{}, assessment_available{};
    bool concern() const noexcept {return coordinate_concern[0] || coordinate_concern[1];}
};
enum class RtcEventReviewDisposition : std::uint8_t { recovered_candidate, persistent_or_compound_unresolved, background_unavailable, no_resolved_excursion };
struct RtcEventReview {
    RtcEventReviewDisposition disposition=RtcEventReviewDisposition::background_unavailable;
    bool source_protection_unavailable=false,protected_optical_assessment_required=false;
    bool health_concern=false;
    bool offset_uncertainty_and_acceptance_required=true;
    bool background_adequacy_required=true;
    bool shared_origin_unresolved=true;
    bool spectral_context_unavailable=true;
    static constexpr bool hard_event_accepted=false;
    static constexpr bool apply_authorized=false;
};

class RtcEventAssessmentDecision {
public:
    static std::shared_ptr<const RtcEventAssessmentDecision> consider(
        std::shared_ptr<const RtcEventAssessmentEvidence> evidence,std::shared_ptr<const ValSnapshot> snapshot,std::uint64_t id) {
        if(!evidence || !snapshot || id==0 || evidence->spike_handle()->val_snapshot_handle().get()!=snapshot.get())
            throw std::invalid_argument("RTC event consideration requires exact evidence and VAL snapshot");
        auto d=std::shared_ptr<RtcEventAssessmentDecision>(new RtcEventAssessmentDecision);
        d->evidence_=evidence;d->id_=id;
        d->screening_=RtcSpikeLearningDecision::consider(evidence->spike_handle(),snapshot,id);
        std::map<rtc_event_assessment_detail::Key,RtcDetectorHealthReview> health;
        for(const auto &h:evidence->health_blocks()) {
            const auto &b=evidence->spike_handle()->blocks()[h.noise_block_index];auto &r=health[{b.network_id,b.detector_index}];r.network=b.network_id;r.detector=b.detector_index;
            if(!h.complete) continue;++r.complete_blocks;
            for(std::size_t c=0;c<2;++c) if(std::isfinite(h.scale_ratio[c]) && std::isfinite(h.edge_fraction[c])) {
                ++r.available_blocks[c];if(h.scale_ratio[c]>RtcEventAssessmentPolicy::health_scale_ratio && h.edge_fraction[c]>RtcEventAssessmentPolicy::health_edge_fraction) ++r.concerning_blocks[c];
            }
        }
        for(auto &[key,h]:health) {
            for(std::size_t c=0;c<2;++c) {
                h.coordinate_concern[c]=h.complete_blocks>=RtcEventAssessmentPolicy::health_minimum_blocks &&
                    static_cast<double>(h.concerning_blocks[c])/h.complete_blocks>=RtcEventAssessmentPolicy::health_block_fraction;
                h.assessment_available[c]=h.coordinate_concern[c] || (h.complete_blocks>=RtcEventAssessmentPolicy::health_minimum_blocks && h.available_blocks[c]==h.complete_blocks);
            }
            d->health_.push_back(h);
        }
        for(const auto &e:evidence->events()) {
            RtcEventReview r;r.health_concern=health.at({e.network,e.detector}).concern();
            bool any=false,all=!e.refinement_limited,excursion=false;
            for(std::size_t c=0;c<2;++c) if(e.seeded[c]) {any|=e.background[c].available();all&=e.recovery[c].recovered();excursion|=e.recovery[c].affected.present();}
            r.disposition=all ? (excursion ? RtcEventReviewDisposition::recovered_candidate : RtcEventReviewDisposition::no_resolved_excursion) : any ? RtcEventReviewDisposition::persistent_or_compound_unresolved : RtcEventReviewDisposition::background_unavailable;
            auto first=e.trial_exclusion.first,last=e.trial_exclusion.past_last;
            for(const auto &v:e.recovery) if(v.affected.present()) {first=std::min(first,v.affected.first);last=std::max(last,v.affected.past_last);}
            for(auto row=first;row<last;++row) {
                const auto state=evidence->spike_handle()->protection_handle()->state(e.network,e.detector,row);
                r.source_protection_unavailable|=state==RtcSpikeProtection::unavailable;
                r.protected_optical_assessment_required|=state==RtcSpikeProtection::protected_source;
            }
            d->events_.push_back(r);
        }
        return d;
    }
    const auto &evidence_handle() const noexcept {return evidence_;}
    const auto &health_reviews() const noexcept {return health_;}
    const auto &event_reviews() const noexcept {return events_;}
    const auto &original_screening_handle() const noexcept {return screening_;}
    std::uint64_t consideration() const noexcept {return id_;}
private:
    RtcEventAssessmentDecision() = default;
    std::shared_ptr<const RtcEventAssessmentEvidence> evidence_;
    std::shared_ptr<const RtcSpikeLearningDecision> screening_;
    std::vector<RtcDetectorHealthReview> health_;
    std::vector<RtcEventReview> events_;
    std::uint64_t id_=0;
};

} // namespace citlali::pipeline
