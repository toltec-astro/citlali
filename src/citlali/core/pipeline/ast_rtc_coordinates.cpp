#include <citlali/core/pipeline/ast_rtc_coordinates.h>
#include <citlali/core/utils/pointing.h>
#include <numbers>

namespace citlali::pipeline {
std::shared_ptr<const AstRtcCoordinates> AstRtcCoordinates::realize_v2(
    std::shared_ptr<const RtcOutputGrid> grid,NativeObservationScope scope,
    std::string telescope,std::shared_ptr<const RawTelescopeTrajectory> trajectory,
    double ra0,double dec0,std::string offset_identity,
    std::shared_ptr<const NativePointingOffsetModel> offsets,
    std::vector<AstRtcDetectorGeometry> geometry) {
    if(!grid || !trajectory || !offsets || offset_identity.empty() ||
        grid->align_handle()->scope()!=scope || geometry.size()!=grid->detectors().size() ||
        telescope!=grid->align_handle()->ast_views_handle()->raw_product_handle()->source_handle()->metadata().source_artifact_identity ||
        !std::isfinite(ra0) || !std::isfinite(dec0) || std::abs(dec0)>std::numbers::pi/2)
        throw std::invalid_argument("AST RTC coordinates require exact grid, telescope, scope, geometry and center");
    auto out=std::shared_ptr<AstRtcCoordinates>(new AstRtcCoordinates);
    out->grid_=std::move(grid);out->trajectory_=std::move(trajectory);out->offsets_=std::move(offsets);
    out->telescope_identity_=std::move(telescope);out->offset_identity_=std::move(offset_identity);
    out->center_ra_=ra0;out->center_dec_=dec0;out->geometry_=std::move(geometry);
    const auto &raw=out->trajectory_->telescope_data();
    for(const auto *name:{"TelRa","TelDec","TelElCor","ActParAng"})
        if(!raw.contains(name))throw std::invalid_argument(std::string("AST required pointing field absent: ")+name);
    // Interpolate/project each exact network schedule once. Detector geometry
    // remains detector-local; time/elevation and telescope motion stay shared.
    struct ScheduledTelescope {
        std::vector<std::size_t> slots;
        NativeTelescopeData evaluated;
        NativePointingOffsetsArcsec offsets;
    };
    std::vector<ScheduledTelescope> scheduled(out->grid_->time_axes().size());
    for(const auto &axis:out->grid_->time_axes()) {
        const auto index=static_cast<std::size_t>(&axis-out->grid_->time_axes().data());
        auto &shared=scheduled[index];std::vector<double> times;
        out->elevations_.emplace_back(axis.assigned_unix_sec.size(),NAN);
        for(std::size_t s=0;s<axis.assigned_unix_sec.size();++s) {
            const auto t=axis.assigned_unix_sec[s];const auto &os=out->offsets_->support_times_unix_sec();
            if(t<out->trajectory_->support_start_unix_sec() || t>out->trajectory_->support_end_unix_sec() ||
               (out->offsets_->source_value_count()==2 && (t<os[0] || t>os[1])))continue;
            shared.slots.push_back(s);times.push_back(t);
        }
        if(times.empty())continue;
        Eigen::VectorXd target=Eigen::Map<const Eigen::VectorXd>(times.data(),times.size());
        shared.evaluated=evaluate_raw_telescope_trajectory_at(*out->trajectory_,target);
        shared.offsets=out->offsets_->evaluate_at(target);
        auto &evaluated=shared.evaluated;
        auto &ra=evaluated.at("TelRa"),&dec=evaluated.at("TelDec");
        for(Eigen::Index i=0;i<ra.size();++i) {
            const double denominator=std::sin(dec0)*std::sin(dec[i])+std::cos(dec0)*std::cos(dec[i])*std::cos(ra[i]-ra0);
            if(!(denominator>1e-12))throw std::invalid_argument("AST V2 tangent projection outside declared local domain");
        }
        Eigen::VectorXd lon(ra.size()),lat(ra.size());
        engine_utils::gnomonic_projection(ra,dec,ra0,dec0,lon,lat);
        evaluated["ra_phys"]=std::move(lon);evaluated["dec_phys"]=std::move(lat);
        evaluated["TelElAct"]-=evaluated.at("TelElCor");
        for(std::size_t i=0;i<shared.slots.size();++i) {
            const double el=evaluated.at("TelElAct")[i]*180/std::numbers::pi;
            if(!std::isfinite(el))throw std::invalid_argument("AST V2 pointing produced nonfinite coordinates");
            out->elevations_[index][shared.slots[i]]=el;
        }
    }
    for(std::size_t d=0;d<out->geometry_.size();++d) {
        const auto &g=out->grid_->detectors()[d];const auto &geo=out->geometry_[d];
        const auto &expected=out->grid_->align_handle()->paired_handle()->network(g.network).detector(g.detector);
        if(geo.detector.network_id!=expected.network_id || geo.detector.storage_column!=expected.storage_column ||
           geo.detector.detector_occurrence_id!=expected.detector_occurrence_id ||
           geo.detector.detector_association_record_id!=expected.detector_association_record_id ||
           geo.detector.tone_or_channel_id!=expected.tone_or_channel_id || geo.selected_apt_row.empty() ||
           geo.array<0 || geo.array>2 || !std::isfinite(geo.x_t_arcsec) || !std::isfinite(geo.y_t_arcsec))
            throw std::invalid_argument("AST RTC detector geometry is missing or foreign");
        out->directions_.emplace_back(g.scheduled_count,2);
        auto &directions=out->directions_.back();directions.setConstant(NAN);
        for(std::size_t s=0;s<g.scheduled_count;++s)if(!std::isfinite(out->elevations_[g.time_axis][s])) {
            const auto state=out->grid_->state(d,s);
            if(state.x_available || state.r_available)
                throw std::invalid_argument("AST required RTC-grid pointing is outside telescope/offset support");
        }
        auto &shared=scheduled[g.time_axis];if(shared.slots.empty())continue;
        auto [y,x]=engine_utils::calc_det_pointing(shared.evaluated,geo.x_t_arcsec,geo.y_t_arcsec,
                                                "radec",shared.offsets,"array",true);
        for(std::size_t i=0;i<shared.slots.size();++i) {
            if(!std::isfinite(x[i]) || !std::isfinite(y[i]))
                throw std::invalid_argument("AST V2 pointing produced nonfinite coordinates");
            directions(shared.slots[i],0)=x[i]*180/std::numbers::pi;
            directions(shared.slots[i],1)=y[i]*180/std::numbers::pi;
        }
    }
    return out;
}
std::optional<AstRtcCoordinates::Direction> AstRtcCoordinates::at(std::size_t detector,std::size_t slot) const {
    const auto &values=directions_.at(detector);
    const auto &g=grid_->detectors().at(detector);
    const auto el=elevations_.at(g.time_axis).at(slot);
    if(!std::isfinite(el))return std::nullopt;
    return Direction{values(slot,0),values(slot,1),el};
}
} // namespace citlali::pipeline
