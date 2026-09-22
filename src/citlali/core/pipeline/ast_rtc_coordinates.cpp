#include <citlali/core/pipeline/ast_rtc_coordinates.h>
#include <citlali/core/utils/pointing.h>
#include <numbers>

namespace citlali::pipeline {
namespace {
template<class Map> struct ReadOnlyPointingSeries {
    const Map &source;
    const auto &operator[](const std::string &key) const { return source.at(key); }
};
}

AstRtcScheduleEvaluation evaluate_ast_rtc_schedule(const RawTelescopeTrajectory &trajectory,
    const NativePointingOffsetModel &offsets,double ra0,double dec0,const Eigen::VectorXd &times) {
    if(!std::isfinite(ra0) || !std::isfinite(dec0) || std::abs(dec0)>std::numbers::pi/2 ||
       !times.size() || !times.array().isFinite().all())
        throw std::invalid_argument("AST RTC schedule or center invalid");
    const auto &support=offsets.support_times_unix_sec();
    for(Eigen::Index i=0;i<times.size();++i)
        if((i && times[i]<=times[i-1]) || times[i]<trajectory.support_start_unix_sec() ||
           times[i]>trajectory.support_end_unix_sec() ||
           (offsets.source_value_count()==2 && (times[i]<support[0] || times[i]>support[1])))
            throw std::invalid_argument("AST RTC schedule outside exact telescope/offset support");
    AstRtcScheduleEvaluation out;
    out.telescope=evaluate_raw_telescope_trajectory_at(trajectory,times);
    out.offsets=offsets.evaluate_at(times);
    auto &evaluated=out.telescope;
    const auto &ra=evaluated.at("TelRa"),&dec=evaluated.at("TelDec");
    for(Eigen::Index i=0;i<ra.size();++i) {
        const double denominator=std::sin(dec0)*std::sin(dec[i])+std::cos(dec0)*std::cos(dec[i])*std::cos(ra[i]-ra0);
        if(!(denominator>1e-12))throw std::invalid_argument("AST V2 tangent projection outside declared local domain");
    }
    Eigen::VectorXd lon(ra.size()),lat(ra.size());
    engine_utils::gnomonic_projection(ra,dec,ra0,dec0,lon,lat);
    evaluated["ra_phys"]=std::move(lon);evaluated["dec_phys"]=std::move(lat);
    evaluated["TelElAct"]-=evaluated.at("TelElCor");
    out.elevation_deg=evaluated.at("TelElAct")*180/std::numbers::pi;
    if(!out.elevation_deg.array().isFinite().all())throw std::invalid_argument("AST nonfinite scheduled elevation");
    return out;
}
Eigen::Matrix<double,Eigen::Dynamic,2> AstRtcScheduleEvaluation::detector_tangent_deg(double x_t,double y_t) const {
    Eigen::Matrix<double,Eigen::Dynamic,2> out(elevation_deg.size(),2);
    if(!std::isfinite(x_t) || !std::isfinite(y_t)){out.setConstant(NAN);return out;}
    // Give the legacy numerical helper checked, read-only lookup without
    // copying shared telescope vectors for each detector.
    ReadOnlyPointingSeries evaluated{telescope};ReadOnlyPointingSeries correction{offsets};
    auto [y,x]=engine_utils::calc_det_pointing(evaluated,x_t,y_t,"radec",correction,"array",true);
    for(Eigen::Index i=0;i<x.size();++i) {
        if(!std::isfinite(x[i]) || !std::isfinite(y[i]))throw std::invalid_argument("AST V2 pointing produced nonfinite coordinates");
        out(i,0)=x[i]*180/std::numbers::pi;out(i,1)=y[i]*180/std::numbers::pi;
    }
    return out;
}
AstRtcTanGrid AstRtcTanGrid::centered(double ra,double dec,double pixel,double x,double y) {
    if(!std::isfinite(ra) || !std::isfinite(dec) || std::abs(dec)>std::numbers::pi/2 ||
       !std::isfinite(pixel) || pixel<=0 || !std::isfinite(x) || !std::isfinite(y) || x<0 || y<0)
        throw std::invalid_argument("AST target TAN grid request invalid");
    const double scale=pixel/3600.;
    const double hx=std::ceil(x/scale+0.5),hy=std::ceil(y/scale+0.5);
    if(!std::isfinite(hx) || !std::isfinite(hy) || hx>100000 || hy>100000 ||
       (2*hx+1)*(2*hy+1)>100000000)
        throw std::invalid_argument("AST target grid exceeds bounded MAP realization capacity");
    return {static_cast<std::size_t>(2*hy+1),static_cast<std::size_t>(2*hx+1),
            hx+1,hy+1,-scale,scale,ra*180/std::numbers::pi,dec*180/std::numbers::pi};
}
std::pair<double,double> AstRtcTanGrid::continuous_pixel(double lon,double lat) const {
    return {lon/cdelt1_deg+crpix1,lat/cdelt2_deg+crpix2};
}
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
        AstRtcScheduleEvaluation evaluated;
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
        shared.evaluated=evaluate_ast_rtc_schedule(*out->trajectory_,*out->offsets_,ra0,dec0,target);
        for(std::size_t i=0;i<shared.slots.size();++i)
            out->elevations_[index][shared.slots[i]]=shared.evaluated.elevation_deg[i];
    }
    for(std::size_t d=0;d<out->geometry_.size();++d) {
        const auto &g=out->grid_->detectors()[d];const auto &geo=out->geometry_[d];
        const auto &expected=out->grid_->align_handle()->paired_handle()->network(g.network).detector(g.detector);
        if(geo.detector.network_id!=expected.network_id || geo.detector.storage_column!=expected.storage_column ||
           geo.detector.detector_occurrence_id!=expected.detector_occurrence_id ||
           geo.detector.detector_association_record_id!=expected.detector_association_record_id ||
           geo.detector.tone_or_channel_id!=expected.tone_or_channel_id || geo.selected_apt_row.empty() ||
           geo.array<0 || geo.array>2)
            throw std::invalid_argument("AST RTC detector geometry is missing or foreign");
        out->directions_.emplace_back(g.scheduled_count,2);
        auto &directions=out->directions_.back();directions.setConstant(NAN);
        for(std::size_t s=0;s<g.scheduled_count;++s)if(!std::isfinite(out->elevations_[g.time_axis][s])) {
            const auto state=out->grid_->state(d,s);
            if(state.x_available || state.r_available)
                throw std::invalid_argument("AST required RTC-grid pointing is outside telescope/offset support");
        }
        // SCI-AST-057: absent detector offsets affect only this coordinate.
        // Preserve the exact binding and shared telescope elevation; no zero offset.
        if(!std::isfinite(geo.x_t_arcsec) || !std::isfinite(geo.y_t_arcsec))continue;
        auto &shared=scheduled[g.time_axis];if(shared.slots.empty())continue;
        const auto projected=shared.evaluated.detector_tangent_deg(geo.x_t_arcsec,geo.y_t_arcsec);
        for(std::size_t i=0;i<shared.slots.size();++i)
            directions.row(shared.slots[i])=projected.row(i);
    }
    return out;
}
std::optional<double> AstRtcCoordinates::telescope_elevation_deg(std::size_t detector,std::size_t slot) const {
    const auto &g=grid_->detectors().at(detector);
    const double el=elevations_.at(g.time_axis).at(slot);
    return std::isfinite(el)?std::optional<double>{el}:std::nullopt;
}
AstRtcCoordinateCause AstRtcCoordinates::cause(std::size_t detector,std::size_t slot) const {
    const auto elevation=telescope_elevation_deg(detector,slot); // validates exact slot
    const auto &g=geometry_.at(detector);
    if(!std::isfinite(g.x_t_arcsec) || !std::isfinite(g.y_t_arcsec))return AstRtcCoordinateCause::missing_detector_geometry;
    return elevation?AstRtcCoordinateCause::available:AstRtcCoordinateCause::outside_telescope_support;
}
std::optional<AstRtcCoordinates::Direction> AstRtcCoordinates::at(std::size_t detector,std::size_t slot) const {
    if(cause(detector,slot)!=AstRtcCoordinateCause::available)return std::nullopt;
    const auto &values=directions_.at(detector);
    return Direction{values(slot,0),values(slot,1),*telescope_elevation_deg(detector,slot)};
}
} // namespace citlali::pipeline
