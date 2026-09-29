// Private adapter: exact files/identities only; AST, CAL and model owners supply
// all coordinate, calibration and containing-pixel numerical operations.
#include <citlali/core/pipeline/point_rtc_source_probe.h>
#include "rtc_point_selection_record.h"
namespace {
struct PointRtcProbeExecution {
  std::shared_ptr<const NativePairedReadoutObservation> parent;
  YAML::Node receipt;
  std::shared_ptr<const RtcPointRunSelectionPlan> rejection;
};
PointRtcProbeExecution execute_point_rtc_probe(
    std::shared_ptr<const NativePairedReadoutObservation> original,
    const YAML::Node &cfg,const apt::VerifiedBundle &verified,
    const CanonicalAptDetectorRelationV2 &relation,const std::vector<int> &channels,
    const CalObservationInput &observation,std::shared_ptr<const AstScanMotionProduct> motion,
    const fs::path &output) {
  const auto request=cfg["rtc_source_probe"];
  require(request["schema"].as<std::string>()=="point-rtc-source-probe-v1","unknown RTC source probe");
  const auto branch=request["branch"].as<std::string>();
  require(branch=="REF" || branch=="C-core-plus" || branch=="C-core-minus" || branch=="C-shoulder","unknown source branch");
  require((request["authority"].as<std::string>()=="owner-RTC-point-transfer-2026-09-28" && cfg["observation"].as<int>()==152420 && cfg["network"].as<int>()>=1 && cfg["network"].as<int>()<=4) || request["authority"].as<std::string>()=="owner-RTC-cost-response-2026-09-27" || (request["authority"].as<std::string>()=="owner-RTC-step-POINT-2026-09-27" && (cfg["observation"].as<int>()==152420 || cfg["observation"].as<int>()==152430)) || (request["authority"].as<std::string>()=="owner-RTC-step-consequences-2026-09-28" && cfg["observation"].as<int>()==152430 && cfg["network"].as<int>()>=1 && cfg["network"].as<int>()<=4),"probe authority absent");
  const int nw=cfg["network"].as<int>();const auto &network=original->network(nw);
  require(network.mapping_authority().x.meaning_id=="kidscpp:gainlintrend:x:meaning" &&
      network.mapping_authority().x.unit_or_scale_id=="dimensionless","probe raw x convention differs");
  auto ast_views=AstScanMotionNetworkViews::admit(original->scope(),motion,{network.occurrence_axis().native_timing_handle()});
  auto align=IdentityRouteAlignContext::admit(original,ast_views,ValSnapshot::initial(original));
  const auto rows=network.occurrence_count(),nd=network.detector_count();
  const auto first=network.occurrence_axis().first_native_row();
  Eigen::VectorXd times(rows);
  for(Eigen::Index t=0;t<rows;++t)times[t]=align->occurrence_assignment(nw,first+t).assigned_time_unix_sec;
  auto schedule=evaluate_ast_rtc_schedule(*observation.trajectory,*observation.offsets,
      observation.ra0,observation.dec0,times);
  std::unique_ptr<FruitPointModel> model;std::string raster_sha;
  if(branch!="REF") {
    const auto path=checked_file(request["raster"]);raster_sha=request["raster"]["sha256"].as<std::string>();
    fitsfile *f=nullptr;int status=0,any=0;fits_open_file(&f,path.c_str(),READONLY,&status);
    long axes[2];int ndim=0;fits_get_img_dim(f,&ndim,&status);fits_get_img_size(f,2,axes,&status);
    require(!status && ndim==2 && axes[0]==41 && axes[1]==41,"RTC source raster geometry");
    auto key=[&](const char *name){double v=NAN;fits_read_key(f,TDOUBLE,name,&v,nullptr,&status);return v;};
    const double cx=key("CRPIX1"),cy=key("CRPIX2"),dx=key("CDELT1"),dy=key("CDELT2"),ra=key("CRVAL1"),dec=key("CRVAL2"),equinox=key("EQUINOX");
    char frame[FLEN_VALUE],unit[FLEN_VALUE];fits_read_key(f,TSTRING,"RADESYS",frame,nullptr,&status);fits_read_key(f,TSTRING,"BUNIT",unit,nullptr,&status);
    require(!status && std::string(frame)=="FK5" && equinox==2000 && std::string(unit)=="mJy/nominal beam" &&
        std::abs(dx+2./3600)<1e-15 && std::abs(dy-2./3600)<1e-15 &&
        std::abs(ra-observation.ra0*180/std::numbers::pi)<1e-10 &&
        std::abs(dec-observation.dec0*180/std::numbers::pi)<1e-10,"RTC source frame/unit/center mismatch");
    std::vector<double> brightness(1681);std::vector<std::uint8_t> valid(1681);
    fits_read_img(f,TDOUBLE,1,1681,nullptr,brightness.data(),&any,&status);
    fits_movnam_hdu(f,IMAGE_HDU,const_cast<char*>("VALID"),0,&status);fits_read_img(f,TBYTE,1,1681,nullptr,valid.data(),&any,&status);
    fits_close_file(f,&status);require(!status,"RTC source raster read failed");
    model=std::make_unique<FruitPointModel>("sha256:"+raster_sha,mapmaking::ForwardMapGrid{41,41,"retained-source-raster"},cx,cy,dx,dy,std::move(brightness),std::move(valid));
  }
  const auto directory=output/"source-probe";fs::create_directory(directory);
  write_matrix(directory/"original-x.f64",network.values(NativeReadoutCoordinate::x));
  write_matrix(directory/"original-r.f64",network.values(NativeReadoutCoordinate::r));
  write_matrix(directory/"assigned-time.f64",times);
  NativePairedReadoutMatrix injected=network.values(NativeReadoutCoordinate::x);
  PtcMatrix lon(rows,nd),lat(rows,nd),multiplier(rows,nd),increment=PtcMatrix::Zero(rows,nd);
  lon.setConstant(NAN);lat.setConstant(NAN);multiplier.setConstant(NAN);
  PtcMask valid=PtcMask::Zero(rows,nd),mapping_available=valid,valid_r=valid;
  auto atmosphere=CalAtmosphereSurface::frozen();
  require(observation.opacity.tau225.has_value(),"RTC source probe lacks accepted opacity mapping");
  YAML::Node record;record["schema"]="point-rtc-source-probe-v1";record["branch"]=branch;
  record["original_x_sha256"]=native_plane_digest(network.values(NativeReadoutCoordinate::x));
  record["original_r_sha256"]=native_plane_digest(network.values(NativeReadoutCoordinate::r));
  record["native_rows"]=rows;record["detectors"]=nd;record["channels"]=channels;record["raster_sha256"]=raster_sha;
  record["assigned_time_policy"]="exact-ALIGN-integration-midpoint;no-backward-interpolation";
  record["units"]="raw dimensionless delta_f/f; increment=sky_mJy_per_nominal_beam/(selected_flxscale*CAL_atmosphere)";
  record["scope"]="x-only finite response at ALIGN-to-RTC; not optical/IQ/ALIGN or r-leakage validation";
  record["tau225"]=*observation.opacity.tau225;
  std::size_t unavailable=0,changed=0;double calibration_error=0;
  for(Eigen::Index d=0;d<nd;++d) {
    const auto edge=std::find_if(relation.bindings().begin(),relation.bindings().end(),[&](const auto &e){return e.network==nw && e.channel==channels[d];});
    require(edge!=relation.bindings().end(),"probe missing detector relation");
    const auto row=std::find_if(verified.apt.rows.begin(),verified.apt.rows.end(),[&](const auto &r){return r.uid==edge->output_uid;});
    require(row!=verified.apt.rows.end(),"probe missing selected APT row");
    const double scale=field(*row,"flxscale"),xt=field(*row,"x_t"),yt=field(*row,"y_t");
    const bool usable=edge->disposition==apt::RelationDisposition::matched && std::isfinite(scale) && scale!=0 && std::isfinite(xt) && std::isfinite(yt);
    if(usable) {const auto xy=schedule.detector_tangent_deg(xt,yt);lon.col(d)=xy.col(0);lat.col(d)=xy.col(1);}
    for(Eigen::Index t=0;t<rows;++t) {
      valid_r(t,d)=network.state(NativeReadoutCoordinate::r,first+t,d).valid();
      valid(t,d)=network.state(NativeReadoutCoordinate::x,first+t,d).valid();
      const auto correction=usable?atmosphere->correction(row->array,*observation.opacity.tau225,schedule.elevation_deg[t]):std::nullopt;
      if(!correction){++unavailable;continue;}
      multiplier(t,d)=scale**correction;mapping_available(t,d)=1;
      if(model) {
        const auto q=point_rtc_source_probe(injected(t,d),valid(t,d),model->at(lon(t,d),lat(t,d)),multiplier(t,d));
        injected(t,d)=q.x;increment(t,d)=q.raw_increment;changed+=q.raw_increment!=0;
        calibration_error=std::max(calibration_error,std::abs(q.raw_increment*multiplier(t,d)-q.brightness));
      }
    }
  }
  write_matrix(directory/"longitude-deg.f64",lon);write_matrix(directory/"latitude-deg.f64",lat);
  write_matrix(directory/"cal-multiplier.f64",multiplier);write_matrix(directory/"increment-x.f64",increment);
  map_write_array<std::uint8_t>(directory/"input-r-valid.u8",{valid_r.data(),std::size_t(valid_r.size())});
  map_write_array<std::uint8_t>(directory/"input-valid.u8",{valid.data(),std::size_t(valid.size())});
  map_write_array<std::uint8_t>(directory/"mapping-available.u8",{mapping_available.data(),std::size_t(mapping_available.size())});
  record["mapping_unavailable_cells"]=unavailable;record["changed_cells"]=changed;
  record["mapping_unavailable_policy"]="original unchanged; CAL-unavailable selected-APT/geometry cells, never fabricated calibration";
  record["calibration_only_max_abs_error_mJy_beam"]=calibration_error;
  auto parent=original;
  if(model) {
    auto mapping=std::make_shared<NativeReadoutMappingAuthority>(network.mapping_authority());
    mapping->paired_xr_record_id+=":pre-RTC-x-source:"+branch+":sha256:"+raster_sha;
    std::vector<NativePairedReadoutNetwork> nets;
    const auto xs=network.states(NativeReadoutCoordinate::x),rs=network.states(NativeReadoutCoordinate::r);
    nets.push_back(NativePairedReadoutNetwork::admit(network.occurrence_axis_handle(),network.detectors(),mapping,
        std::move(injected),network.values(NativeReadoutCoordinate::r),{xs.begin(),xs.end()},{rs.begin(),rs.end()}));
    parent=std::make_shared<const NativePairedReadoutObservation>(NativePairedReadoutObservation::admit(original->scope(),{nw},std::move(nets)));
  }
  std::shared_ptr<const RtcPointRunSelectionPlan> rejection;
  if(cfg["rtc_point_run_selection"]) {
    require(request["authority"].as<std::string>()=="owner-RTC-point-transfer-2026-09-28" &&
      cfg["rtc_point_run_selection"].as<std::string>()==RtcPointRunSelectionPlan::identity,
      "RTC point selector outside bounded policy route");
    const auto &selected=parent->network(nw);
    std::vector<RtcPointRun> runs;
    require(first==0,"RTC point selector requires complete native row origin");
    for(auto r:network.occurrence_axis().contiguous_runs())runs.push_back({r.first_native_row,r.past_last_native_row});
    auto doubles=[](const auto &m){return std::span<const double>(m.data(),m.size());};
    auto bytes=[](const auto &m){return std::span<const std::uint8_t>(m.data(),m.size());};
    const auto pointing=cfg["ast_acceptance"]["sha256"].as<std::string>();
    rejection=std::make_shared<const RtcPointRunSelectionPlan>(RtcPointRunSelectionPlan::consider(RtcPointRunSelectionEvidence::learn({
      std::size_t(rows),std::size_t(nd),doubles(selected.values(NativeReadoutCoordinate::x)),
      doubles(selected.values(NativeReadoutCoordinate::r)),doubles(times),doubles(lon),doubles(lat),
      bytes(valid),bytes(valid_r),bytes(mapping_available),channels,runs,
      times.size()>1 ? (times[1]-times[0]) : 0.,
      selected.mapping_authority().paired_xr_record_id,pointing})));
    write_yaml(directory/"run-selection.yaml",citlali::cli::detail::rtc_point_selection_record(*rejection));
    record["run_selection_sha256"]=citlali::utils::sha256_file(directory/"run-selection.yaml");
  }
  record["injected_x_sha256"]=native_plane_digest(parent->network(nw).values(NativeReadoutCoordinate::x));
  record["r_unchanged"]=record["original_r_sha256"].as<std::string>()==native_plane_digest(parent->network(nw).values(NativeReadoutCoordinate::r));
  record["input_validity_unchanged"]=true;record["learning_plan_frozen"]=false;
  for(const auto &p:fs::directory_iterator(directory))record["files"][p.path().filename().string()]=citlali::utils::sha256_file(p.path());
  write_yaml(directory/"receipt.yaml",record);return {parent,record,rejection};
}
}
