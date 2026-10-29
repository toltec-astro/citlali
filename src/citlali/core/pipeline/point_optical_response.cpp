#include <citlali/core/pipeline/point_optical_response.h>
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace citlali::pipeline {
namespace {
constexpr double huber_tuning = 1.345;
using Key = std::pair<std::int64_t, std::int64_t>; // time bin, science block

double median(std::vector<double> v) {
    v.erase(std::remove_if(v.begin(),v.end(),[](double x){return !std::isfinite(x);}),v.end());
    if(v.empty())return NAN;
    const auto m=v.size()/2;std::nth_element(v.begin(),v.begin()+m,v.end());
    const double upper=v[m];
    return v.size()%2?upper:std::midpoint(upper,*std::max_element(v.begin(),v.begin()+m));
}
double scale(const std::vector<double> &v) {
    const double m=median(v);std::vector<double> a;a.reserve(v.size());
    for(double x:v)if(std::isfinite(x))a.push_back(std::abs(x-m));
    return 1.4826*median(std::move(a));
}
void validate_request(const PointOpticalRequest &r) {
    if(!std::isfinite(r.min_relative_response)||r.min_relative_response<=0 ||
       !std::isfinite(r.bin_seconds)||r.bin_seconds<=0 ||
       !std::isfinite(r.confidence_level)||r.confidence_level<=0 ||r.confidence_level>=1 ||
       r.bootstrap_samples<2 ||r.bootstrap_samples>1000000 ||
       (1-r.confidence_level)*r.bootstrap_samples/2<1)
        throw std::invalid_argument("optical response request has invalid threshold, bins or resampling resolution");
}
struct Stats {
    long double w=0,x=0,y=0,xx=0,xy=0;std::size_t n=0;
    void add(double a,double b,double weight) {
        w+=weight;x+=weight*a;y+=weight*b;xx+=weight*a*a;xy+=weight*a*b;++n;
        if(!std::isfinite(w)||!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(xx)||!std::isfinite(xy))
            throw std::runtime_error("optical response numerical overflow in response moments");
    }
    Stats &operator+=(const Stats &b){w+=b.w;x+=b.x;y+=b.y;xx+=b.xx;xy+=b.xy;n+=b.n;
        if(!std::isfinite(w)||!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(xx)||!std::isfinite(xy))
            throw std::runtime_error("optical response numerical overflow in resampled moments");
        return *this;}
    bool solve(double &b,double &a) const {
        const auto d=w*xx-x*x;
        if(n<3 || !(w>0) || !(d>32*std::numeric_limits<double>::epsilon()*std::max(w*xx,x*x)))return false;
        b=static_cast<double>((w*xy-x*y)/d);a=static_cast<double>((y-b*x)/w);
        return std::isfinite(a)&&std::isfinite(b);
    }
};
struct Fit {
    bool available=false;double beta=NAN,intercept=NAN,correlation=NAN,rms=NAN;
    std::vector<std::size_t> rows;std::vector<double> weights;
};
Fit fit(const std::vector<double> &x,const std::vector<double> &y,
        const std::vector<Key> &keys,const std::set<std::int64_t> *blocks=nullptr) {
    Fit out;Stats s;
    for(std::size_t i=0;i<x.size();++i)if(std::isfinite(x[i])&&std::isfinite(y[i])&&
        (!blocks||blocks->contains(keys[i].second))){out.rows.push_back(i);s.add(x[i],y[i],1);}
    if(!s.solve(out.beta,out.intercept))return out;
    out.weights.assign(out.rows.size(),1);
    for(int it=0;it<20;++it) {
        std::vector<double> residual;residual.reserve(out.rows.size());
        for(auto i:out.rows)residual.push_back(y[i]-out.intercept-out.beta*x[i]);
        const double sigma=scale(residual);
        if(!std::isfinite(sigma)||sigma<=0)break;
        Stats next;
        for(std::size_t j=0;j<out.rows.size();++j){const auto i=out.rows[j];
            out.weights[j]=std::min(1.,huber_tuning*sigma/std::max(std::abs(residual[j]),std::numeric_limits<double>::min()));
            next.add(x[i],y[i],out.weights[j]);}
        double beta,intercept;if(!next.solve(beta,intercept))return Fit{};
        const bool done=std::abs(beta-out.beta)<=1e-10*(1+std::abs(out.beta))&&
            std::abs(intercept-out.intercept)<=1e-10*(1+std::abs(out.intercept));
        out.beta=beta;out.intercept=intercept;if(done)break;
    }
    long double sx=0,sy=0,sxx=0,syy=0,sxy=0,se=0;
    for(auto i:out.rows){sx+=x[i];sy+=y[i];sxx+=x[i]*x[i];syy+=y[i]*y[i];sxy+=x[i]*y[i];
        const double e=y[i]-out.intercept-out.beta*x[i];se+=e*e;}
    const auto n=out.rows.size();const long double vx=n*sxx-sx*sx,vy=n*syy-sy*sy;
    if(vx>0&&vy>0)out.correlation=static_cast<double>((n*sxy-sx*sy)/std::sqrt(vx*vy));
    out.rms=std::sqrt(static_cast<double>(se/n));out.available=true;return out;
}
std::uint64_t seed_for(const std::string &text) {
    std::uint64_t h=14695981039346656037ULL;
    for(unsigned char c:text){h^=c;h*=1099511628211ULL;}return h;
}
struct Generator {
    std::uint64_t state;
    std::uint64_t next(){auto z=(state+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
        z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
    std::size_t index(std::size_t n){const std::uint64_t m=n,threshold=-m%m;std::uint64_t x;do{x=next();}while(x<threshold);return x%m;}
};
double quantile(const std::vector<double> &v,double q) {
    const double x=q*(v.size()-1);const auto i=static_cast<std::size_t>(std::floor(x));
    return v[i]+(x-i)*(v[std::min(i+1,v.size()-1)]-v[i]);
}
PointOpticalInterval interval(const std::vector<double> &x,const std::vector<double> &y,
    const std::vector<Key> &keys,const PointOpticalRequest &request,double normalization,
    std::uint64_t seed,const std::set<std::int64_t> *selection=nullptr) {
    PointOpticalInterval out;out.seed=seed;out.reason="response_fit_unavailable";
    const auto f=fit(x,y,keys,selection);if(!f.available)return out;
    std::map<std::int64_t,Stats> grouped;
    for(std::size_t j=0;j<f.rows.size();++j){const auto i=f.rows[j];grouped[keys[i].second].add(x[i],y[i],f.weights[j]);}
    std::vector<Stats> blocks;
    // Resample the exact row population used by the full fit. A singleton or
    // constant-reference block may carry essential between-block response
    // information even though it cannot determine a slope by itself.
    for(const auto &[id,s]:grouped)blocks.push_back(s);
    out.informative_blocks=blocks.size();out.reason="insufficient_independent_science_blocks";
    if(blocks.size()<2)return out;
    std::vector<double> slopes;slopes.reserve(request.bootstrap_samples);Generator rng{seed};
    for(std::size_t k=0;k<request.bootstrap_samples;++k) {
        Stats s;for(std::size_t b=0;b<blocks.size();++b)s+=blocks[rng.index(blocks.size())];
        double beta,offset;if(!s.solve(beta,offset)){out.reason="resample_rank_unavailable";return out;}
        slopes.push_back(beta/normalization);
    }
    std::sort(slopes.begin(),slopes.end());out.resamples=slopes.size();
    const double alpha=(1-request.confidence_level)/2;
    out.lower=quantile(slopes,alpha);out.upper=quantile(slopes,1-alpha);
    if(!std::isfinite(out.lower)||!std::isfinite(out.upper)||
       out.upper-out.lower<=32*std::numeric_limits<double>::epsilon()*(1+std::abs(out.lower)+std::abs(out.upper))) {
        out.reason="degenerate_resampled_interval";out.lower=out.upper=NAN;return out;
    }
    out.available=true;out.reason="conditional_science_block_resampling";return out;
}
struct Binned {
    PtcMatrix values;
    std::vector<double> full,reference;
    std::array<std::vector<double>,2> panel,reference_panel;
};
std::vector<double> combine(const std::vector<const std::vector<double> *> &sources,std::size_t n) {
    std::vector<double> out(n,NAN),v;v.reserve(sources.size());
    for(std::size_t i=0;i<n;++i){v.clear();for(auto p:sources)if(std::isfinite((*p)[i]))v.push_back((*p)[i]);out[i]=median(v);}
    return out;
}
bool normalize(std::vector<double> &x,const std::vector<double> &target) {
    std::vector<double> xx,tt;for(std::size_t i=0;i<x.size();++i)
        if(std::isfinite(x[i])&&std::isfinite(target[i])){xx.push_back(x[i]);tt.push_back(target[i]);}
    const double a=scale(xx),b=scale(tt);if(!(a>0&&b>0&&std::isfinite(a)&&std::isfinite(b)))return false;
    for(auto &v:x)if(std::isfinite(v))v*=b/a;return true;
}
void validate_interval(const PointOpticalInterval &x,const PointOpticalRequest &r) {
    if(x.reason.empty() || (x.available && (!std::isfinite(x.lower)||!std::isfinite(x.upper)||
        x.lower>x.upper||x.informative_blocks<2||x.resamples!=r.bootstrap_samples)))
        throw std::invalid_argument("optical response restored interval is inconsistent");
}
} // namespace

std::shared_ptr<const PointOpticalEvidence> PointOpticalEvidence::learn(
    const std::vector<PointOpticalNetwork> &input,PointOpticalRequest request) {
    validate_request(request);if(input.empty())throw std::invalid_argument("optical response requires CAL networks");
    auto out=std::shared_ptr<PointOpticalEvidence>(new PointOpticalEvidence);out->request_=request;
    std::set<int> networks;std::set<std::string> detector_ids;double first=INFINITY;
    for(const auto &n:input) {
        if(n.observation_identity.empty()||n.cal_identity.empty()||n.source_mask_identity.empty()||
           n.observation_identity!=input.front().observation_identity||n.array<0||n.array>2||n.network<0||
           !networks.insert(n.network).second||n.detector_ids.empty()||n.times_unix_sec.empty()||
           n.science_blocks.size()!=n.times_unix_sec.size()||n.values.rows()!=static_cast<Eigen::Index>(n.times_unix_sec.size())||
           n.values.cols()!=static_cast<Eigen::Index>(n.detector_ids.size())||n.eligible.rows()!=n.values.rows()||
           n.eligible.cols()!=n.values.cols()||n.source_free.rows()!=n.values.rows()||n.source_free.cols()!=n.values.cols())
            throw std::invalid_argument("optical response CAL scope, identities or scheduled shapes differ");
        for(const auto &id:n.detector_ids)if(id.empty()||!detector_ids.insert(id).second)
            throw std::invalid_argument("optical response detector identities must be unique exact occurrences");
        for(std::size_t t=0;t<n.times_unix_sec.size();++t) {
            if(!std::isfinite(n.times_unix_sec[t])||(t&&n.times_unix_sec[t]<=n.times_unix_sec[t-1])||n.science_blocks[t]<-1)
                throw std::invalid_argument("optical response native times or science blocks invalid");
            if(n.science_blocks[t]>=0)first=std::min(first,n.times_unix_sec[t]);
            for(std::size_t d=0;d<n.detector_ids.size();++d)
                if(n.eligible(t,d)>1||n.source_free(t,d)>1||(n.eligible(t,d)&&!std::isfinite(n.values(t,d))))
                    throw std::invalid_argument("optical response finite CAL eligibility or source mask invalid");
        }
    }
    if(!std::isfinite(first))throw std::invalid_argument("optical response has no processing science domain");
    // One explicit analysis grid for learning only; native times and CAL slots
    // are untouched. Distinct processing blocks are never merged by a bin.
    const double origin=std::floor(first/request.bin_seconds)*request.bin_seconds;
    std::map<Key,std::size_t> indices;
    auto key=[&](const PointOpticalNetwork &n,std::size_t t) {
        const double b=std::floor((n.times_unix_sec[t]-origin)/request.bin_seconds);
        if(!std::isfinite(b)||b<0||b>10000000)throw std::invalid_argument("optical analysis grid exceeds bounded capacity");
        return Key{static_cast<std::int64_t>(b),n.science_blocks[t]};
    };
    for(const auto &n:input)for(std::size_t t=0;t<n.times_unix_sec.size();++t)if(n.science_blocks[t]>=0)indices.emplace(key(n,t),0);
    std::vector<Key> keys;for(auto &[k,i]:indices){i=keys.size();keys.push_back(k);}
    std::vector<std::int64_t> block_order;std::set<std::int64_t> seen;
    for(auto k:keys)if(seen.insert(k.second).second)block_order.push_back(k.second);
    const auto half=block_order.size()/2;
    const std::set<std::int64_t> early(block_order.begin(),block_order.begin()+half),late(block_order.begin()+half,block_order.end());
    std::vector<Binned> binned; binned.reserve(input.size());out->networks_.reserve(input.size());
    for(const auto &n:input) {
        PointOpticalNetworkEvidence facts;facts.observation_identity=n.observation_identity;facts.cal_identity=n.cal_identity;
        facts.source_mask_identity=n.source_mask_identity;facts.array=n.array;facts.network=n.network;facts.detector_ids=n.detector_ids;
        facts.detectors.resize(n.detector_ids.size());
        Binned b;b.values=PtcMatrix::Zero(keys.size(),n.detector_ids.size());
        Eigen::Matrix<std::uint32_t,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> counts=Eigen::Matrix<std::uint32_t,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::Zero(b.values.rows(),b.values.cols());
        for(std::size_t d=0;d<n.detector_ids.size();++d)facts.detectors[d].detector_id=n.detector_ids[d];
        for(std::size_t t=0;t<n.times_unix_sec.size();++t)if(n.science_blocks[t]>=0) {
            const auto row=indices.at(key(n,t));
            for(std::size_t d=0;d<n.detector_ids.size();++d)if(n.eligible(t,d)) {
                ++facts.detectors[d].available_samples;
                if(n.source_free(t,d)){++facts.detectors[d].source_free_samples;b.values(row,d)+=n.values(t,d);++counts(row,d);
                    if(!std::isfinite(b.values(row,d)))throw std::runtime_error("optical response numerical overflow in time aggregate");}
            }
        }
        for(std::size_t d=0;d<n.detector_ids.size();++d) {
            std::vector<double> v;
            for(std::size_t r=0;r<keys.size();++r)if(counts(r,d)){b.values(r,d)/=counts(r,d);v.push_back(b.values(r,d));}else b.values(r,d)=NAN;
            facts.detectors[d].binned_samples=v.size();const double center=median(v);
            for(std::size_t r=0;r<keys.size();++r)if(std::isfinite(b.values(r,d))) {
                b.values(r,d)-=center;
                if(!std::isfinite(b.values(r,d)))throw std::runtime_error("optical response numerical overflow in centered aggregate");
            }
        }
        std::vector<std::size_t> order(n.detector_ids.size());std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](auto a,auto c){return n.detector_ids[a]<n.detector_ids[c];});
        b.full.resize(keys.size());for(auto &panel:b.panel)panel.resize(keys.size());
        for(std::size_t r=0;r<keys.size();++r) {
            std::vector<double> all;std::array<std::vector<double>,2> panel;
            for(std::size_t i=0;i<order.size();++i){const double x=b.values(r,order[i]);if(std::isfinite(x)){all.push_back(x);panel[i%2].push_back(x);}}
            b.full[r]=median(all);for(int j=0;j<2;++j)b.panel[j][r]=median(panel[j]);
        }
        binned.push_back(std::move(b));out->networks_.push_back(std::move(facts));
    }
    std::array<std::vector<double>,3> array_reference;
    for(int a=0;a<3;++a){std::vector<const std::vector<double>*> v;
        for(std::size_t n=0;n<input.size();++n)if(input[n].array==a)v.push_back(&binned[n].full);
        array_reference[a]=combine(v,keys.size());}
    // Equal influence per peer network; no candidate detector or its own
    // readout network enters its temporal reference.
    for(std::size_t n=0;n<input.size();++n) {
        auto &f=out->networks_[n];auto &b=binned[n];std::vector<const std::vector<double>*> refs;
        std::array<std::vector<const std::vector<double>*>,2> panels;
        for(std::size_t p=0;p<input.size();++p)if(p!=n&&input[p].array==f.array) {
            f.reference_networks.push_back(input[p].network);refs.push_back(&binned[p].full);
            for(int j=0;j<2;++j)panels[j].push_back(&binned[p].panel[j]);}
        b.reference=combine(refs,keys.size());for(int j=0;j<2;++j)b.reference_panel[j]=combine(panels[j],keys.size());
        f.reference_reason="heldout_network_reference_unavailable";
        if(refs.empty())continue;
        f.reference_available=normalize(b.reference,array_reference[f.array]);
        for(auto &panel:b.reference_panel)f.reference_available&=normalize(panel,array_reference[f.array]);
        if(!f.reference_available){f.reference_reason="shared_reference_excitation_unavailable";continue;}
        const auto shared=interval(b.reference,b.full,keys,request,1,
            seed_for(f.observation_identity+":shared-network:"+std::to_string(f.network)));
        if(!shared.available||shared.lower<=0){f.reference_available=false;f.reference_reason="shared_network_coupling_unavailable";continue;}
        f.reference_reason="heldout_network_reference_available";
        // With just one other readout network, two panels are not independent
        // electronics. Require positive block-resampled cross-array coupling.
        f.cross_array_required=refs.size()==1;
        if(refs.size()==1) {
            bool any=false,all=true;
            for(int a=0;a<3;++a)if(a!=f.array) {
                if(std::none_of(array_reference[a].begin(),array_reference[a].end(),[](double x){return std::isfinite(x);}))continue;
                any=true;const auto ci=interval(b.reference,array_reference[a],keys,request,1,
                    seed_for(f.observation_identity+":cross-array:"+std::to_string(f.network)+":"+std::to_string(a)));
                all&=ci.available&&ci.lower>0;
            }
            f.cross_array_corroborated=any&&all;
            if(!f.cross_array_corroborated)f.reference_reason="cross_array_corroboration_unavailable";
        }
        for(std::size_t d=0;d<f.detectors.size();++d) {
            std::vector<double> y(keys.size());for(std::size_t r=0;r<keys.size();++r)y[r]=b.values(r,d);
            const auto model=fit(b.reference,y,keys);
            if(model.available){f.detectors[d].relative_response=model.beta;f.detectors[d].correlation=model.correlation;f.detectors[d].binned_residual_rms=model.rms;}
        }
    }
    for(int a=0;a<3;++a) {
        std::vector<double> slopes;
        for(const auto &n:out->networks_)if(n.array==a&&n.reference_available)
            for(const auto &d:n.detectors)if(std::isfinite(d.relative_response))slopes.push_back(d.relative_response);
        const double normal=median(slopes);
        for(auto &n:out->networks_)if(n.array==a) {
            n.response_normalization=normal;
            if(!(normal>0&&std::isfinite(normal))){n.reference_available=false;n.reference_reason="normal_response_scale_unavailable";}
            for(auto &d:n.detectors)d.relative_response=n.reference_available?d.relative_response/normal:NAN;
        }
    }
    for(std::size_t n=0;n<input.size();++n) {
        auto &f=out->networks_[n];auto &b=binned[n];
        for(std::size_t d=0;d<f.detectors.size();++d) {
            auto &z=f.detectors[d];z.interval.reason=f.reference_reason;
            if(!f.reference_available||(f.cross_array_required&&!f.cross_array_corroborated)||!z.source_free_samples)continue;
            if(!std::isfinite(z.relative_response)){z.interval.reason="response_fit_unavailable";continue;}
            std::vector<double> y(keys.size());for(std::size_t r=0;r<keys.size();++r)y[r]=b.values(r,d);
            bool complete=true;double lower=INFINITY,upper=-INFINITY;std::size_t min_blocks=std::numeric_limits<std::size_t>::max();
            for(int part=0;part<3;++part)for(int panel=0;panel<2;++panel) {
                const auto seed=seed_for(f.observation_identity+":"+f.cal_identity+":"+z.detector_id+":"+
                    std::to_string(part)+":"+std::to_string(panel)+":"+std::string(method));
                auto ci=interval(b.reference_panel[panel],y,keys,request,f.response_normalization,seed,part==0?nullptr:part==1?&early:&late);
                z.checks[part*2+panel]=ci;complete&=ci.available;
                if(ci.available){lower=std::min(lower,ci.lower);upper=std::max(upper,ci.upper);min_blocks=std::min(min_blocks,ci.informative_blocks);}
            }
            z.interval.available=complete;z.interval.reason=complete?"conditional_block_panel_and_temporal_envelope":"response_interval_or_stability_unavailable";
            if(complete){z.interval.lower=lower;z.interval.upper=upper;z.interval.informative_blocks=min_blocks;z.interval.resamples=request.bootstrap_samples;}
        }
    }
    return out;
}

std::shared_ptr<const PointOpticalEvidence> PointOpticalEvidence::restore_verified(
    PointOpticalRequest request,std::vector<PointOpticalNetworkEvidence> networks) {
    validate_request(request);if(networks.empty())throw std::invalid_argument("restored optical evidence has no networks");
    std::set<int> ids;std::set<std::string> detectors;
    for(const auto &n:networks) {
        if(n.observation_identity.empty()||n.observation_identity!=networks.front().observation_identity||n.cal_identity.empty()||
           n.source_mask_identity.empty()||n.array<0||n.array>2||n.network<0||!ids.insert(n.network).second||
           n.detector_ids.empty()||n.detector_ids.size()!=n.detectors.size()||n.reference_reason.empty()||
           (n.reference_available&&(!(n.response_normalization>0)||!std::isfinite(n.response_normalization)||n.reference_networks.empty()||
               n.cross_array_required!=(n.reference_networks.size()==1))))
            throw std::invalid_argument("restored optical network scope or reference is inconsistent");
        std::set<int> refs;for(int p:n.reference_networks)if(p==n.network||!refs.insert(p).second)
            throw std::invalid_argument("restored optical reference includes own or duplicated network");
        for(std::size_t i=0;i<n.detectors.size();++i) {
            const auto &d=n.detectors[i];if(d.detector_id.empty()||d.detector_id!=n.detector_ids[i]||!detectors.insert(d.detector_id).second||
               d.source_free_samples>d.available_samples||d.binned_samples>d.source_free_samples)
                throw std::invalid_argument("restored optical detector binding or support is inconsistent");
            validate_interval(d.interval,request);for(const auto &c:d.checks)validate_interval(c,request);
            if(d.interval.available) {
                if(!n.reference_available||(n.cross_array_required&&!n.cross_array_corroborated)||!d.source_free_samples||!std::isfinite(d.relative_response))
                    throw std::invalid_argument("restored optical interval lacks admitted reference or source support");
                for(const auto &c:d.checks)if(!c.available||d.interval.lower>c.lower||d.interval.upper<c.upper)
                    throw std::invalid_argument("restored optical interval fails its reference/temporal envelope");
            }
        }
    }
    for(const auto &n:networks)for(int ref:n.reference_networks) {
        const auto p=std::find_if(networks.begin(),networks.end(),[&](const auto &p){return p.network==ref;});
        if(p==networks.end()||p->array!=n.array)throw std::invalid_argument("restored optical reference has foreign array or missing network");
    }
    auto out=std::shared_ptr<PointOpticalEvidence>(new PointOpticalEvidence);out->request_=request;out->networks_=std::move(networks);return out;
}

std::string_view point_optical_state_name(PointOpticalState s) {
    switch(s){case PointOpticalState::prior_unavailable:return "prior_unavailable";case PointOpticalState::retained:return "retained";
        case PointOpticalState::weak_response:return "weak_response";case PointOpticalState::sign_mismatch:return "sign_mismatch";
        case PointOpticalState::unresolved:return "unresolved";}throw std::invalid_argument("unknown optical response state");
}
std::shared_ptr<const PointOpticalPlan> PointOpticalPlan::consider(std::shared_ptr<const PointOpticalEvidence> e) {
    if(!e)throw std::invalid_argument("optical response Consider requires evidence");
    const double threshold=e->request().min_relative_response;return consider(std::move(e),threshold);
}
std::shared_ptr<const PointOpticalPlan> PointOpticalPlan::consider(std::shared_ptr<const PointOpticalEvidence> e,double threshold) {
    if(!e||!std::isfinite(threshold)||threshold<=0)throw std::invalid_argument("optical response Consider requires evidence and positive threshold");
    auto out=std::shared_ptr<PointOpticalPlan>(new PointOpticalPlan);out->evidence_=std::move(e);out->threshold_=threshold;
    for(const auto &n:out->evidence_->networks()) {
        std::vector<PointOpticalDecision> decisions;
        for(const auto &d:n.detectors) {
            PointOpticalDecision v;v.reason=d.interval.reason;
            if(!d.available_samples){v.state=PointOpticalState::prior_unavailable;v.reason="no_eligible_CAL_science_samples";}
            else if(!d.source_free_samples)v.reason="source_free_learning_support_unavailable";
            else if(d.interval.available) {
                if(d.interval.lower>-threshold&&d.interval.upper<threshold){v.state=PointOpticalState::weak_response;v.reason="confidently_weak_shared_optical_response";}
                else if(d.interval.upper<-threshold){v.state=PointOpticalState::sign_mismatch;v.reason="confidently_reversed_shared_optical_response";}
                else if(d.interval.lower>threshold){v.state=PointOpticalState::retained;v.reason="positive_optical_response_established";}
                else v.reason="response_interval_overlaps_decision_boundary";
            }
            decisions.push_back(std::move(v));
        }
        out->decisions_.push_back(std::move(decisions));
    }
    return out;
}
const std::vector<PointOpticalDecision> &PointOpticalPlan::for_network(const PointOpticalNetwork &input) const {
    const auto &ns=evidence_->networks();
    for(std::size_t i=0;i<ns.size();++i)if(ns[i].network==input.network) {
        const auto &n=ns[i];if(n.array!=input.array||n.observation_identity!=input.observation_identity||n.cal_identity!=input.cal_identity||
            n.source_mask_identity!=input.source_mask_identity||n.detector_ids!=input.detector_ids)
            throw std::invalid_argument("optical response membership requested with foreign CAL, mask or detector binding");
        return decisions_[i];
    }
    throw std::invalid_argument("optical response membership requested for absent network");
}
} // namespace citlali::pipeline
