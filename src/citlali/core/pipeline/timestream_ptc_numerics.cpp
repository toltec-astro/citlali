#include <citlali/core/pipeline/timestream_ptc_numerics.h>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>

namespace citlali::pipeline {
namespace {
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
std::vector<PtcPattern> patterns(const PtcMask &m,bool transpose) {
    std::map<std::vector<Eigen::Index>,std::vector<Eigen::Index>> grouped;
    for(Eigen::Index i=0;i<(transpose?m.cols():m.rows());++i) {
        std::vector<Eigen::Index> entries;
        for(Eigen::Index j=0;j<(transpose?m.rows():m.cols());++j)
            if(transpose?m(j,i):m(i,j))entries.push_back(j);
        grouped[entries].push_back(i);
    }
    std::vector<PtcPattern> result;
    for(auto &[entries,occurrences]:grouped)result.push_back({entries,std::move(occurrences)});
    return result;
}
struct Inverse { Eigen::MatrixXd value; int rank=0; bool finite=false; };
Inverse inverse(const Eigen::MatrixXd &g,double tolerance) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(g);
    Inverse out;
    if(solver.info()!=Eigen::Success || !solver.eigenvalues().allFinite())return out;
    const auto &e=solver.eigenvalues();const double cut=std::max(0.,e.maxCoeff())*tolerance;
    Eigen::VectorXd reciprocals=e;
    for(Eigen::Index i=0;i<e.size();++i){const bool use=e[i]>cut;out.rank+=use;reciprocals[i]=use?1./e[i]:0.;}
    out.value=solver.eigenvectors()*reciprocals.asDiagonal()*solver.eigenvectors().transpose();
    out.finite=out.value.allFinite();return out;
}
Eigen::MatrixXd coefficients(const PtcPrepared &p,const Eigen::MatrixXd &b,PtcFit &fit) {
    Eigen::MatrixXd a=Eigen::MatrixXd::Zero(p.centered.rows(),b.cols());
    for(const auto &pattern:p.time_patterns) {
        if(pattern.entries.empty())continue;
        Eigen::MatrixXd selected(pattern.entries.size(),b.cols());
        for(std::size_t j=0;j<pattern.entries.size();++j)selected.row(j)=b.row(pattern.entries[j]);
        const auto inv=inverse(selected.transpose()*selected,fit.request.relative_rank_tolerance);
        if(!inv.finite)throw std::runtime_error("coefficient-normal-matrix-failure");
        ++fit.coefficient_factorizations;fit.coefficient_factor_reuses+=pattern.occurrences.size()-1;
        const Eigen::MatrixXd projector=inv.value*selected.transpose();
        Eigen::VectorXd z(pattern.entries.size());
        for(auto t:pattern.occurrences) {
            for(std::size_t j=0;j<pattern.entries.size();++j)z[j]=p.centered(t,pattern.entries[j]);
            a.row(t)=(projector*z).transpose();
        }
    }
    return a;
}
double objective(const PtcPrepared &p,const Eigen::MatrixXd &b,const Eigen::MatrixXd &a) {
    double sum=0;
    for(Eigen::Index t=0;t<p.centered.rows();++t)for(Eigen::Index d=0;d<p.centered.cols();++d)
        if(p.eligible(t,d)){const double r=p.centered(t,d)-a.row(t).dot(b.row(d));sum+=r*r;}
    return sum;
}
void initialize(const PtcPrepared &p,PtcFit &out) {
    const auto started=Clock::now();
    const Eigen::MatrixXd mask=p.eligible.cast<double>().matrix();
    const Eigen::MatrixXd counts=mask.transpose()*mask;
    const Eigen::MatrixXd numerator=p.centered.transpose()*p.centered;
    Eigen::MatrixXd covariance(numerator.rows(),numerator.cols());
    for(Eigen::Index d=0;d<covariance.rows();++d)for(Eigen::Index e=0;e<covariance.cols();++e) {
        if(counts(d,e)<=1)throw std::runtime_error("insufficient-pairwise-initialization-overlap");
        covariance(d,e)=numerator(d,e)/(counts(d,e)-1.);
    }
    out.covariance_seconds=seconds(started);const auto eig_at=Clock::now();
    if(!covariance.allFinite())throw std::runtime_error("covariance-nonfinite");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(covariance);
    if(eig.info()!=Eigen::Success || !eig.eigenvalues().allFinite())throw std::runtime_error("covariance-decomposition-failed");
    const auto &values=eig.eigenvalues();
    if(values[values.size()-out.request.rank]<=std::max(0.,values.maxCoeff())*out.request.relative_rank_tolerance)
        throw std::runtime_error("requested-basis-rank-unavailable");
    // Pairwise covariance need not be PSD. Negative trailing eigenvalues are
    // retained as a known limitation; every selected mode must be positive.
    out.basis=eig.eigenvectors().rightCols(out.request.rank).rowwise().reverse();
    out.decomposition_seconds=seconds(eig_at);
}
PtcApplied apply(const PtcPrepared &p,const PtcFit &fit,const PtcMatrix &values) {
    const auto start=Clock::now();PtcApplied out;
    out.values=PtcMatrix::Zero(values.rows(),values.cols());out.causes=PtcMask::Ones(values.rows(),values.cols());
    for(const auto &pattern:p.time_patterns) {
        if(pattern.entries.empty())continue;
        const bool fit_valid=fit.converged && fit.basis.rows()==values.cols() && fit.basis.cols()==fit.request.rank;
        Eigen::MatrixXd selected;Inverse inv;
        if(fit_valid) {
            selected.resize(pattern.entries.size(),fit.request.rank);
            for(std::size_t j=0;j<pattern.entries.size();++j)selected.row(j)=fit.basis.row(pattern.entries[j]);
            inv=inverse(selected.transpose()*selected,fit.request.relative_rank_tolerance);
            ++out.factorizations;out.factor_reuses+=pattern.occurrences.size()-1;
        }
        const bool rank_ok=fit_valid && inv.finite && inv.rank==fit.request.rank;
        Eigen::MatrixXd projector;
        if(rank_ok)projector=inv.value*selected.transpose();
        Eigen::VectorXd z(pattern.entries.size());
        for(auto t:pattern.occurrences) {
            std::uint8_t cause=!fit_valid?2:!rank_ok?4:0;
            for(std::size_t j=0;j<pattern.entries.size();++j)z[j]=values(t,pattern.entries[j]);
            if(!z.allFinite())cause|=8;
            Eigen::VectorXd cleaned;
            if(!cause){cleaned=z-selected*(projector*z);if(!cleaned.allFinite())cause=8;}
            if(cause)++out.failed_times;
            for(std::size_t j=0;j<pattern.entries.size();++j) {
                const auto d=pattern.entries[j];out.causes(t,d)=cause;
                if(!cause){out.values(t,d)=cleaned[j];++out.retained;}
            }
        }
    }
    out.seconds=seconds(start);return out;
}
}
PtcPrepared PtcPrepared::prepare(const PtcMatrix &values,const PtcMask &mask) {
    const auto start=Clock::now();
    if(values.rows()<1 || values.cols()<2 || mask.rows()!=values.rows() || mask.cols()!=values.cols() || (mask>1).any())
        throw std::invalid_argument("PTC requires rectangular CAL values and explicit binary eligibility");
    PtcPrepared p;p.eligible=mask;p.centered=PtcMatrix::Zero(values.rows(),values.cols());p.mean=Eigen::VectorXd::Zero(values.cols());
    for(Eigen::Index d=0;d<values.cols();++d) {
        std::size_t count=0;long double sum=0;
        for(Eigen::Index t=0;t<values.rows();++t)if(mask(t,d)) {
            if(!std::isfinite(values(t,d)))throw std::invalid_argument("PTC admitted CAL value is nonfinite");
            sum+=values(t,d);++count;
        }
        if(count)p.mean[d]=static_cast<double>(sum/count);
        p.eligible_count+=count;
        for(Eigen::Index t=0;t<values.rows();++t)if(mask(t,d))p.centered(t,d)=values(t,d)-p.mean[d];
    }
    for(Eigen::Index t=0;t<values.rows();++t)if(mask.row(t).all())++p.complete_times;
    p.time_patterns=patterns(mask,false);p.detector_patterns=patterns(mask,true);p.preparation_seconds=seconds(start);return p;
}
PtcFit ptc_learn(const PtcPrepared &p,PtcSolverRequest request) {
    if((request.method!=PtcMethod::observed_als && request.method!=PtcMethod::pairwise_covariance) || request.rank<=0 || request.iteration_limit<1 || !std::isfinite(request.relative_objective_tolerance) ||
       request.relative_objective_tolerance<=0 || request.relative_objective_tolerance>=1 ||
       !std::isfinite(request.relative_rank_tolerance) || request.relative_rank_tolerance<=0 || request.relative_rank_tolerance>=1)
        throw std::invalid_argument("PTC numerical request is invalid");
    const auto started=Clock::now();PtcFit out;out.request=request;
    try {
        if(request.rank>std::min(p.centered.cols(),p.centered.rows()-1))throw std::runtime_error("requested-rank-exceeds-shape");
        auto at=Clock::now();initialize(p,out);out.initialization_seconds=seconds(at);
        at=Clock::now();auto a=coefficients(p,out.basis,out);out.coefficient_seconds+=seconds(at);
        at=Clock::now();out.objective.push_back(objective(p,out.basis,a));out.check_seconds+=seconds(at);
        if(request.method==PtcMethod::pairwise_covariance){out.converged=true;out.stopping_reason="noniterative-complete";}
        int small=0;
        for(int iteration=0;request.method==PtcMethod::observed_als && iteration<request.iteration_limit;++iteration) {
            at=Clock::now();Eigen::MatrixXd next=Eigen::MatrixXd::Zero(out.basis.rows(),out.basis.cols());
            for(const auto &pattern:p.detector_patterns) {
                Eigen::MatrixXd selected(pattern.entries.size(),request.rank);
                for(std::size_t j=0;j<pattern.entries.size();++j)selected.row(j)=a.row(pattern.entries[j]);
                const auto inv=inverse(selected.transpose()*selected,request.relative_rank_tolerance);
                // An unidentified detector loading must not be silently dropped.
                if(!inv.finite || inv.rank!=request.rank)throw std::runtime_error("basis-update-deficient-support");
                const Eigen::MatrixXd projector=inv.value*selected.transpose();Eigen::VectorXd z(pattern.entries.size());
                for(auto d:pattern.occurrences) {
                    for(std::size_t j=0;j<pattern.entries.size();++j)z[j]=p.centered(pattern.entries[j],d);
                    next.row(d)=(projector*z).transpose();
                }
            }
            if(!next.allFinite() || inverse(next.transpose()*next,request.relative_rank_tolerance).rank!=request.rank)
                throw std::runtime_error("basis-update-lost-requested-rank");
            Eigen::HouseholderQR<Eigen::MatrixXd> qr(next);
            out.basis=qr.householderQ()*Eigen::MatrixXd::Identity(next.rows(),request.rank);
            out.basis_seconds+=seconds(at);
            // New basis: no old basis-dependent factorization is reused.
            at=Clock::now();a=coefficients(p,out.basis,out);out.coefficient_seconds+=seconds(at);
            at=Clock::now();const double loss=objective(p,out.basis,a),previous=out.objective.back();
            out.objective.push_back(loss);++out.iterations;out.check_seconds+=seconds(at);
            if(!std::isfinite(loss) || loss>previous+1e-10*std::max(1.,previous))throw std::runtime_error("objective-nonfinite-or-increased");
            const double decrease=(previous-loss)/std::max(previous,1e-300);
            if(decrease<=request.relative_objective_tolerance || loss<=1e-24*p.centered.squaredNorm())++small;else small=0;
            if(small>=2){out.converged=true;out.stopping_reason="two-small-relative-objective-decreases";break;}
        }
        if(!out.converged)out.stopping_reason="iteration-limit-not-converged";
    } catch(const std::runtime_error &e){out.converged=false;out.stopping_reason=e.what();}
    out.fit_seconds=seconds(started);return out;
}
PtcApplied ptc_apply(const PtcPrepared &p,const PtcFit &fit){return apply(p,fit,p.centered);}
PtcApplied ptc_response(const PtcPrepared &p,const PtcFit &fit,const PtcMatrix &response) {
    if(response.rows()!=p.centered.rows() || response.cols()!=p.centered.cols())throw std::invalid_argument("PTC response differs from CAL grid");
    return apply(p,fit,response);
}
} // namespace citlali::pipeline
