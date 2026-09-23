#include <citlali/core/pipeline/fruit_cycle.h>
#include <cmath>
#include <stdexcept>
namespace citlali::pipeline {
FruitPointModel::FruitPointModel(std::string id,mapmaking::ForwardMapGrid g,double cx,double cy,double dx,double dy,
    std::vector<double> v,std::vector<std::uint8_t> valid):identity_(std::move(id)),grid_(std::move(g)),cx_(cx),cy_(cy),dx_(dx),dy_(dy),values_(std::move(v)),valid_(std::move(valid)) {
    if(identity_.empty() || !grid_.rows || !grid_.columns || values_.size()!=grid_.rows*grid_.columns || valid_.size()!=values_.size() ||
        !std::isfinite(cx+cy+dx+dy) || dx>=0 || dy<=0)throw std::invalid_argument("FRUIT invalid model grid");
    // Entire computational disk must be represented; a crop cannot silently
    // truncate its interior and classify it as outside.
    if((cx-.5)*-dx*3600<40 || (grid_.columns+.5-cx)*-dx*3600<40 ||
       (cy-.5)*dy*3600<40 || (grid_.rows+.5-cy)*dy*3600<40)throw std::invalid_argument("FRUIT truncated model disk");
    for(std::size_t p=0;p<values_.size();++p)if(valid_[p]>1 || (valid_[p] && !std::isfinite(values_[p])))
        throw std::invalid_argument("FRUIT invalid model support/value");
}
FruitOperand FruitPointModel::at(double lon,double lat) const {
    if(!std::isfinite(lon) || !std::isfinite(lat))return {0,FruitModelSupport::unavailable};
    const auto p=mapmaking::containing_pixel(grid_,cx_+lon/dx_,cy_+lat/dy_);
    if(!p)return {0,FruitModelSupport::outside};
    const double x=(double(*p%grid_.columns)+1-cx_)*dx_*3600,y=(double(*p/grid_.columns)+1-cy_)*dy_*3600;
    if(std::hypot(x,y)>40)return {0,FruitModelSupport::outside};
    if(!valid_[*p])return {0,FruitModelSupport::unavailable};
    return {values_[*p],values_[*p]==0?FruitModelSupport::excluded:FruitModelSupport::admitted};
}
PtcMatrix fruit_injected_parent(const PtcMatrix &y,const PtcMask &e,const PtcMatrix &i,const PtcMask &s) {
    if(y.rows()!=e.rows() || y.cols()!=e.cols() || y.rows()!=i.rows() || y.cols()!=i.cols() ||
       y.rows()!=s.rows() || y.cols()!=s.cols())throw std::invalid_argument("FRUIT injection shape mismatch");
    PtcMatrix result=y;
    for(Eigen::Index k=0;k<y.size();++k){
        if(e.data()[k]>1 || s.data()[k]>1)throw std::invalid_argument("FRUIT injection nonbinary support");
        if(!e.data()[k])continue;
        if(!s.data()[k] || !std::isfinite(y.data()[k]) || !std::isfinite(i.data()[k]))
            throw std::invalid_argument("FRUIT injection unavailable/nonfinite parent or projection");
        result.data()[k]+=i.data()[k];
        if(!std::isfinite(result.data()[k]))throw std::invalid_argument("FRUIT injection overflow");
    }return result;
}
std::shared_ptr<const FruitResidual> FruitResidual::subtract(std::string id,const PtcMatrix &y,const PtcMask &eligible,PtcMatrix operand,PtcMask supported) {
    if(id.empty() || y.rows()!=eligible.rows() || y.cols()!=eligible.cols() || y.rows()!=operand.rows() || y.cols()!=operand.cols() ||
        y.rows()!=supported.rows() || y.cols()!=supported.cols())throw std::invalid_argument("FRUIT residual identity/shape mismatch");
    auto r=std::shared_ptr<FruitResidual>(new FruitResidual);r->identity_=std::move(id);r->operand_=std::move(operand);
    r->values_=PtcMatrix::Zero(y.rows(),y.cols());r->eligible_=PtcMask::Zero(y.rows(),y.cols());
    for(Eigen::Index i=0;i<y.size();++i){
        if(eligible.data()[i]>1 || supported.data()[i]>1)throw std::invalid_argument("FRUIT nonbinary eligibility");
        if(!eligible.data()[i] || !supported.data()[i]){r->operand_.data()[i]=0;continue;}
        const auto a=y.data()[i],b=r->operand_.data()[i];const auto value=a-b;
        if(!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(value))throw std::invalid_argument("FRUIT nonfinite subtraction");
        r->values_.data()[i]=value;r->eligible_.data()[i]=1;
    }return r;
}
FruitLearned FruitLearned::learn(std::shared_ptr<const FruitResidual> r,PtcSolverRequest request) {
    if(!r)throw std::invalid_argument("FRUIT missing residual");
    FruitLearned l;l.residual_=std::move(r);l.prepared_=PtcPrepared::prepare(l.residual_->values(),l.residual_->eligible());
    l.fit_=ptc_learn(l.prepared_,request);return l;
}
FruitLearned::Result FruitLearned::apply(const std::shared_ptr<const FruitResidual> &exact) const {
    if(exact.get()!=residual_.get())throw std::invalid_argument("FRUIT stale residual/learned state");
    auto c=ptc_apply(prepared_,fit_);auto total=c.values;
    for(Eigen::Index i=0;i<total.size();++i)if(c.causes.data()[i]==0){
        total.data()[i]+=residual_->operand().data()[i];
        if(!std::isfinite(total.data()[i]))throw std::invalid_argument("FRUIT nonfinite rejoin");
    }return {std::move(c),std::move(total)};
}
} // namespace citlali::pipeline
