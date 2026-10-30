#include <citlali/core/pipeline/ptc_inverse_noise.h>
#include <citlali/core/pipeline/fruit_cycle.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace citlali::pipeline {
namespace {
struct Moment {
    std::size_t count=0;
    long double mean=0, square=0;
    void add(long double x) {
        ++count;const auto delta=x-mean;mean+=delta/count;
        square+=delta*(x-mean);
    }
};
struct Regression {
    std::size_t count=0;
    long double x=0,y=0,xx=0,xy=0,slope=0,offset=0,sse=0;
    bool supported=false;
    void add(long double a,long double b) {
        ++count;const auto dx=a-x,dy=b-y;x+=dx/count;y+=dy/count;
        xx+=dx*(a-x);xy+=dx*(b-y);
    }
};
struct Datum { double value; Eigen::Index detector; };

// Only the central three order statistics are needed for every leave-one-out
// median. Selection is linear in detector count and shares the reference work
// across detectors; it does not construct a dense time x detector reference.
template<class Visitor>
void visit_references(const PtcMatrix &values,const PtcMask &eligible,
    const PtcMask &source_free,const std::vector<double> &means,Visitor visit) {
    std::vector<Datum> row;row.reserve(values.cols());
    for(Eigen::Index t=0;t<values.rows();++t) {
        row.clear();
        for(Eigen::Index d=0;d<values.cols();++d)if(eligible(t,d) && source_free(t,d)) {
            const double centered=values(t,d)-means[d];
            if(!std::isfinite(centered))throw std::overflow_error("PTC noise donor centering overflow");
            row.push_back({centered,d});
        }
        if(row.size()<2)continue;
        const auto mid=row.begin()+row.size()/2;
        const auto less=[](const Datum &a,const Datum &b){return a.value<b.value;};
        std::nth_element(row.begin(),mid,row.end(),less);
        const double middle=mid->value;
        const double lower=std::max_element(row.begin(),mid,less)->value;
        const double upper=row.size()%2?std::min_element(mid+1,row.end(),less)->value:middle;
        for(const auto &item:row) {
            double reference;
            if(row.size()%2) {
                const double a=item.value<middle?middle:lower;
                const double b=item.value>middle?middle:upper;
                reference=static_cast<double>((static_cast<long double>(a)+b)/2);
            } else reference=item.value<=lower?middle:lower;
            visit(t,item.detector,reference);
        }
    }
}

std::vector<PtcDetectorNoiseEvidence> reference_noise(const PtcMatrix &values,
    const PtcMask &eligible,const PtcMask &source_free) {
    std::vector<PtcDetectorNoiseEvidence> evidence(values.cols());
    std::vector<double> means(values.cols(),0);
    std::vector<Regression> regressions(values.cols());
    for(Eigen::Index d=0;d<values.cols();++d) {
        Moment mean;
        for(Eigen::Index t=0;t<values.rows();++t)if(eligible(t,d)) {
            ++evidence[d].eligible_count;
            if(source_free(t,d))mean.add(values(t,d));
        }
        evidence[d].source_free_count=mean.count;
        means[d]=static_cast<double>(mean.mean);
    }
    visit_references(values,eligible,source_free,means,[&](auto t,auto d,double r) {
        regressions[d].add(r,values(t,d));
    });
    for(Eigen::Index d=0;d<values.cols();++d) {
        auto &e=evidence[d];auto &r=regressions[d];e.reference_count=r.count;
        if(!e.eligible_count)e.reference_reason="no-eligible-input";
        else if(r.count<=2)e.reference_reason="insufficient-source-free-reference-pairs";
        else if(!(r.xx>0) || !std::isfinite(r.xx))e.reference_reason="constant-or-nonfinite-reference";
        else {
            r.slope=r.xy/r.xx;r.offset=r.y-r.slope*r.x;
            const auto slope=static_cast<double>(r.slope),offset=static_cast<double>(r.offset);
            if(!std::isfinite(slope) || !std::isfinite(offset))e.reference_reason="nonfinite-reference-fit";
            else {e.reference_slope=slope;e.reference_offset=offset;r.supported=true;}
        }
    }
    // Direct residual power avoids cancellation in Syy-Sxy^2/Sxx and retains
    // every admitted disturbance, including long tails and isolated excursions.
    visit_references(values,eligible,source_free,means,[&](auto t,auto d,double x) {
        auto &r=regressions[d];if(!r.supported)return;
        const long double residual=static_cast<long double>(values(t,d))-r.offset-r.slope*x;
        r.sse+=residual*residual;
    });
    for(Eigen::Index d=0;d<values.cols();++d) {
        auto &e=evidence[d];const auto &r=regressions[d];
        if(!r.supported)continue;
        const double variance=static_cast<double>(r.sse/(r.count-2));
        const double coefficient=1/variance;
        if(!(variance>0) || !std::isfinite(variance) || !std::isfinite(coefficient) || !(coefficient>0))
            e.reference_reason="nonpositive-or-nonfinite-reference-variance";
        else {e.reference_variance=variance;e.cleaning_coefficient=coefficient;e.reference_reason="available";}
    }
    return evidence;
}
}

PtcInverseNoiseResult inverse_noise_ptc(const PtcMatrix &values,
    const PtcMask &eligible,const PtcMask &source_free,PtcSolverRequest request,std::string scope,PointWeighting weighting) {
    if(weighting!=PointWeighting::inverse_noise && weighting!=PointWeighting::equal_control)
        throw std::invalid_argument("POINT unsupported weighting control");
    if(values.rows()<1 || values.cols()<1 || eligible.rows()!=values.rows() || eligible.cols()!=values.cols() ||
       source_free.rows()!=values.rows() || source_free.cols()!=values.cols() ||
       (eligible>1).any() || (source_free>1).any())
        throw std::invalid_argument("PTC inverse noise requires matching matrices and binary masks");
    for(Eigen::Index t=0;t<values.rows();++t)for(Eigen::Index d=0;d<values.cols();++d)
        if(eligible(t,d) && !std::isfinite(values(t,d)))
            throw std::invalid_argument("PTC inverse noise eligible calibrated value is nonfinite");
    PtcInverseNoiseResult out;out.scope=std::move(scope);out.fit.request=request;
    out.detectors=reference_noise(values,eligible,source_free);
    for(auto &e:out.detectors) {
        e.weighting=weighting;
        if(e.cleaning_coefficient && weighting==PointWeighting::equal_control)e.cleaning_coefficient=1.;
    }
    out.effective_eligible=eligible;
    out.cleaned.values=PtcMatrix::Zero(values.rows(),values.cols());
    out.cleaned.causes=PtcMask::Constant(values.rows(),values.cols(),17);
    for(Eigen::Index d=0;d<values.cols();++d) {
        if(out.detectors[d].reference_variance)out.fit_columns.push_back(d);
        else out.effective_eligible.col(d).setZero();
    }
    if(out.fit_columns.empty()) {
        out.fit.stopping_reason="no-detector-with-supported-reference-noise";
    } else {
        PtcMatrix whitened=PtcMatrix::Zero(values.rows(),out.fit_columns.size());
        PtcMask compact(values.rows(),out.fit_columns.size());
        for(std::size_t c=0;c<out.fit_columns.size();++c) {
            const auto d=out.fit_columns[c];const double sigma=weighting==PointWeighting::equal_control?1.:std::sqrt(*out.detectors[d].reference_variance);
            compact.col(c)=out.effective_eligible.col(d);
            for(Eigen::Index t=0;t<values.rows();++t)if(compact(t,c)) {
                whitened(t,c)=values(t,d)/sigma;
                if(!std::isfinite(whitened(t,c)))throw std::overflow_error("PTC inverse-noise whitening overflow");
            }
        }
        out.prepared=PtcPrepared::prepare(whitened,compact);
        out.fit=ptc_learn(out.prepared,request,nullptr,out.scope);
        const auto applied=ptc_apply(out.prepared,out.fit);
        out.cleaned.failed_times=applied.failed_times;out.cleaned.factorizations=applied.factorizations;
        out.cleaned.factor_reuses=applied.factor_reuses;out.cleaned.seconds=applied.seconds;
        for(std::size_t c=0;c<out.fit_columns.size();++c) {
            const auto d=out.fit_columns[c];const double sigma=weighting==PointWeighting::equal_control?1.:std::sqrt(*out.detectors[d].reference_variance);
            out.cleaned.causes.col(d)=applied.causes.col(c);
            for(Eigen::Index t=0;t<values.rows();++t)if(applied.causes(t,c)==0) {
                const double value=applied.values(t,c)*sigma;
                if(!std::isfinite(value))out.cleaned.causes(t,d)=8;
                else {out.cleaned.values(t,d)=value;++out.cleaned.retained;}
            }
        }
    }
    for(Eigen::Index d=0;d<values.cols();++d) {
        auto &e=out.detectors[d];Moment remaining;
        for(Eigen::Index t=0;t<values.rows();++t)
            if(out.cleaned.causes(t,d)==0 && source_free(t,d))remaining.add(out.cleaned.values(t,d));
        e.remaining_count=remaining.count;
        if(!e.reference_variance)e.remaining_reason="reference-noise-unavailable";
        else if(remaining.count<2)e.remaining_reason="insufficient-source-free-cleaned-samples";
        else {
            const double variance=static_cast<double>(remaining.square/(remaining.count-1));
            if(!(variance>=0) || !std::isfinite(variance))e.remaining_reason="nonfinite-remaining-variance";
            else {
                e.remaining_variance=variance;
                e.map_coefficient=weighting==PointWeighting::equal_control?1.:1/std::max(*e.reference_variance,variance);
                e.remaining_reason="available";
            }
        }
    }
    return out;
}

PtcInverseNoiseFruitResult inverse_noise_fruit(std::shared_ptr<const FruitResidual> residual,
    const PtcMask &source_free,PtcSolverRequest request,std::string scope,PointWeighting weighting) {
    if(!residual)throw std::invalid_argument("PTC inverse noise requires an exact FRUIT residual");
    PtcInverseNoiseFruitResult out;out.residual=std::move(residual);
    out.noise=inverse_noise_ptc(out.residual->values(),out.residual->eligible(),source_free,request,std::move(scope),weighting);
    out.total=PtcMatrix::Zero(out.noise.cleaned.values.rows(),out.noise.cleaned.values.cols());
    for(Eigen::Index t=0;t<out.total.rows();++t)for(Eigen::Index d=0;d<out.total.cols();++d)
        if(out.noise.cleaned.causes(t,d)==0) {
            const double value=out.noise.cleaned.values(t,d)+out.residual->operand()(t,d);
            if(!std::isfinite(value))throw std::overflow_error("PTC inverse-noise FRUIT add-back overflow");
            out.total(t,d)=value;
        }
    return out;
}
} // namespace citlali::pipeline
