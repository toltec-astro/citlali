// Private configured POINT boundary. Only parsing, exact persisted-domain
// joins and stage orchestration live here; AST/health/PTC/FRUIT/MAP own science.
#include <citlali/core/pipeline/point_optical_response.h>
#include <citlali/core/pipeline/ptc_inverse_noise.h>
#include <citlali/core/pipeline/ptc_empirical_map_handoff.h>
#include <citlali/core/pipeline/val_empirical_map_handoff.h>
#include "point_weighted_bindings.h"
#include "point_recovery_bindings.h"
namespace {
const std::array<std::string,3> weighted_arrays{"a1100","a1400","a2000"};
const std::array<double,3> weighted_beams{5.,7.5,10.}, weighted_source_radii{35.,42.5,50.};
std::string point_support_digest(const PtcMask &mask) {
    citlali::utils::Sha256 hash;
    hash.update(std::to_string(mask.rows())+":"+std::to_string(mask.cols())+":row-major-binary:");
    std::vector<std::uint8_t> row(mask.cols());
    for(Eigen::Index t=0;t<mask.rows();++t) {
        for(Eigen::Index d=0;d<mask.cols();++d)row[d]=mask(t,d)!=0;
        hash.update(row.data(),row.size());
    }
    return hash.finish();
}
void point_number(YAML::Node node,const char *key,double value) {
    if(std::isfinite(value))node[key]=value;else node[key]=YAML::Null;
}
double point_number(const YAML::Node &node,const char *key) {
    return node[key] && !node[key].IsNull()?node[key].as<double>():NAN;
}
YAML::Node point_interval(const PointOpticalInterval &x) {
    YAML::Node n;n["available"]=x.available;point_number(n,"lower",x.lower);point_number(n,"upper",x.upper);
    n["informative_blocks"]=x.informative_blocks;n["resamples"]=x.resamples;n["seed"]=x.seed;n["reason"]=x.reason;return n;
}
PointOpticalInterval point_interval(const YAML::Node &n) {
    return {n["available"].as<bool>(),point_number(n,"lower"),point_number(n,"upper"),
        n["informative_blocks"].as<std::size_t>(),n["resamples"].as<std::size_t>(),n["seed"].as<std::uint64_t>(),n["reason"].as<std::string>()};
}
YAML::Node point_health_record(const PointOpticalPlan &plan) {
    YAML::Node record;record["schema"]="citlali-point-optical-health-v1";
    record["method"]=std::string(PointOpticalEvidence::method);record["uncertainty_limitations"]=std::string(PointOpticalEvidence::uncertainty);
    const auto &r=plan.evidence_handle()->request();record["min_relative_response"]=plan.min_relative_response();
    record["bin_seconds"]=r.bin_seconds;record["confidence_level"]=r.confidence_level;record["bootstrap_samples"]=r.bootstrap_samples;
    for(std::size_t ni=0;ni<plan.evidence_handle()->networks().size();++ni) {
        const auto &e=plan.evidence_handle()->networks()[ni];YAML::Node n;
        n["observation_identity"]=e.observation_identity;n["cal_identity"]=e.cal_identity;n["source_mask_identity"]=e.source_mask_identity;
        n["array"]=e.array;n["network"]=e.network;n["detector_ids"]=e.detector_ids;n["reference_networks"]=e.reference_networks;
        n["reference_available"]=e.reference_available;n["cross_array_required"]=e.cross_array_required;
        n["cross_array_corroborated"]=e.cross_array_corroborated;n["reference_reason"]=e.reference_reason;
        point_number(n,"response_normalization",e.response_normalization);
        for(std::size_t di=0;di<e.detectors.size();++di) {
            const auto &d=e.detectors[di];const auto &decision=plan.decisions()[ni][di];YAML::Node v;
            v["detector_id"]=d.detector_id;v["available_samples"]=d.available_samples;v["source_free_samples"]=d.source_free_samples;v["binned_samples"]=d.binned_samples;
            point_number(v,"relative_response",d.relative_response);point_number(v,"correlation",d.correlation);point_number(v,"binned_residual_rms",d.binned_residual_rms);
            v["interval"]=point_interval(d.interval);for(const auto &c:d.checks)v["checks"].push_back(point_interval(c));
            v["state"]=std::string(point_optical_state_name(decision.state));v["reason"]=decision.reason;n["detectors"].push_back(v);
        }
        record["networks"].push_back(n);
    }
    return record;
}
std::shared_ptr<const PointOpticalPlan> point_restore_health(const YAML::Node &record,double threshold) {
    require(record["schema"].as<std::string>()=="citlali-point-optical-health-v1" &&
        record["method"].as<std::string>()==std::string(PointOpticalEvidence::method) &&
        record["min_relative_response"].as<double>()==threshold,"POINT frozen health method/threshold differs");
    PointOpticalRequest request;request.min_relative_response=threshold;
    require(record["bin_seconds"].as<double>()==request.bin_seconds && record["confidence_level"].as<double>()==request.confidence_level &&
        record["bootstrap_samples"].as<std::size_t>()==request.bootstrap_samples,"POINT frozen uncertainty method differs");
    std::vector<PointOpticalNetworkEvidence> evidence;
    for(const auto &n:record["networks"]) {
        PointOpticalNetworkEvidence e;e.observation_identity=n["observation_identity"].as<std::string>();
        e.cal_identity=n["cal_identity"].as<std::string>();e.source_mask_identity=n["source_mask_identity"].as<std::string>();
        e.array=n["array"].as<int>();e.network=n["network"].as<int>();e.detector_ids=n["detector_ids"].as<std::vector<std::string>>();
        e.reference_networks=n["reference_networks"].as<std::vector<int>>();e.reference_available=n["reference_available"].as<bool>();
        e.cross_array_required=n["cross_array_required"].as<bool>();e.cross_array_corroborated=n["cross_array_corroborated"].as<bool>();
        e.reference_reason=n["reference_reason"].as<std::string>();e.response_normalization=point_number(n,"response_normalization");
        for(const auto &v:n["detectors"]) {
            PointOpticalDetectorEvidence d;d.detector_id=v["detector_id"].as<std::string>();d.available_samples=v["available_samples"].as<std::size_t>();
            d.source_free_samples=v["source_free_samples"].as<std::size_t>();d.binned_samples=v["binned_samples"].as<std::size_t>();
            d.relative_response=point_number(v,"relative_response");d.correlation=point_number(v,"correlation");d.binned_residual_rms=point_number(v,"binned_residual_rms");
            d.interval=point_interval(v["interval"]);require(v["checks"].size()==6,"POINT incomplete conditional intervals");
            for(std::size_t j=0;j<6;++j)d.checks[j]=point_interval(v["checks"][j]);e.detectors.push_back(std::move(d));
        }
        evidence.push_back(std::move(e));
    }
    auto plan=PointOpticalPlan::consider(PointOpticalEvidence::restore_verified(request,std::move(evidence)));
    for(std::size_t i=0;i<plan->decisions().size();++i)for(std::size_t d=0;d<plan->decisions()[i].size();++d)
        citlali::cli::detail::validate_point_health_decision(record["networks"][i]["detectors"][d],point_optical_state_name(plan->decisions()[i][d].state),plan->decisions()[i][d].reason);
    return plan;
}
struct WeightedPointSegment {std::int64_t scan;std::vector<std::int64_t> slots;};
struct WeightedPointNetwork {
    SavedMapNetwork saved;PointOpticalNetwork optical;PtcMatrix coordinates;
    std::vector<WeightedPointSegment> segments;std::string processing_identity;
};
WeightedPointNetwork point_load_cal(const YAML::Node &binding,MapFileLedger &ledger,const std::string &config_sha) {
    WeightedPointNetwork n;n.saved=load_saved_map_network(binding,ledger,true,true);auto &s=n.saved;auto &o=n.optical;
    require(s.input["effective_config"]["sha256"].as<std::string>()==config_sha,"POINT network effective config differs");
    const auto processing=s.root/"processing-scans.yaml";const auto published=map_selected_fields(s.root/"receipt.yaml","",{"processing_scans_sha256"});
    const auto processing_sha=ledger.bind(processing,binding["processing_sha256"].as<std::string>());n.processing_identity="sha256:"+processing_sha;
    const auto scans=YAML::LoadFile(processing.string());
    const auto intervals=citlali::cli::detail::validate_point_processing_binding(published,s.input,scans,binding["processing_sha256"].as<std::string>(),processing_sha,s.native_times.size());
    o.observation_identity=s.observation_identity;o.cal_identity="sha256:"+s.cal_sha;o.array=s.array;o.network=s.network;o.detector_ids=s.detectors;
    o.source_mask_identity=o.cal_identity+":"+n.processing_identity+":delta-source-AltAz:radius-arcsec="+std::to_string(weighted_source_radii[s.array]);
    o.science_blocks.assign(s.scheduled,-1);o.times_unix_sec.resize(s.scheduled);
    Eigen::VectorXd times(s.scheduled);for(std::size_t t=0;t<s.scheduled;++t)times[t]=o.times_unix_sec[t]=s.native_times[s.first+t*s.factor];
    for(const auto &interval:intervals) {
        const auto lo=interval.first,hi=interval.past_last;WeightedPointSegment segment{interval.scan,{}};
        for(std::size_t t=0;t<s.scheduled;++t)if(const auto row=s.first+t*s.factor;row>=static_cast<std::size_t>(lo) && row<static_cast<std::size_t>(hi)) {
            require(o.science_blocks[t]<0,"POINT overlapping science support");o.science_blocks[t]=segment.scan;segment.slots.push_back(t);
        }
        if(!segment.slots.empty())n.segments.push_back(std::move(segment));
    }
    require(!n.segments.empty(),"POINT no processing science support");
    o.values=PtcMatrix::Zero(s.scheduled,s.detectors.size());o.eligible=PtcMask::Zero(o.values.rows(),o.values.cols());o.source_free=o.eligible;
    const auto ast=evaluate_ast_rtc_schedule(*s.observation.trajectory,*s.observation.offsets,s.observation.ra0,s.observation.dec0,times,AstRtcMapFrame::delta_source_altaz);
    n.coordinates.resize(s.scheduled,2*s.detectors.size());
    for(std::size_t d=0;d<s.detectors.size();++d) {
        const auto &record=s.cal["detectors"][d];const auto count=record["available"].as<std::size_t>();
        const auto stem="channel-"+std::to_string(s.channels[d]);const auto root=s.root/"donor-continuity/cal";
        for(const auto *suffix:{"-value.f64","-slot.i64"})ledger.bind(root/(stem+suffix),record["files"][stem+suffix].as<std::string>());
        const auto values=map_read_array<double>(root/(stem+"-value.f64"),count);const auto slots=map_read_array<std::int64_t>(root/(stem+"-slot.i64"),count);
        std::size_t seen=0;for(std::size_t t=0;t<s.scheduled;++t)if(s.cal_causes[d][t]==0) {
            require(seen<count && slots[seen]==static_cast<std::int64_t>(t) && std::isfinite(values[seen]),"POINT CAL values/support join differs");
            o.values(t,d)=values[seen++];o.eligible(t,d)=o.science_blocks[t]>=0;
        }
        require(seen==count,"POINT excess CAL values");const auto [gx,gy]=s.geometry[d];const auto xy=ast.detector_tangent_deg(gx,gy);
        n.coordinates.col(2*d)=xy.col(0);n.coordinates.col(2*d+1)=xy.col(1);
        const auto mask=ast.outside_source_radius(gx,gy,weighted_source_radii[s.array]);
        for(std::size_t t=0;t<s.scheduled;++t)o.source_free(t,d)=mask[t] && o.eligible(t,d);
    }
    return n;
}
FruitOffsetModel point_read_model(const YAML::Node &binding,MapFileLedger &ledger,int array,double pixel,
    const std::string &observation,const std::string &health_sha,int cycle,PointWeighting weighting) {
    const auto path=checked_file(binding["candidate"]);const auto sha=ledger.bind(path);
    const auto feedback_path=checked_file(binding["feedback"]),bundle_path=checked_file(binding["bundle"]);
    ledger.bind(feedback_path);const auto bundle_sha=ledger.bind(bundle_path);
    const auto feedback=YAML::LoadFile(feedback_path.string()),bundle=YAML::LoadFile(bundle_path.string());
    require(bundle["coefficient_family"].as<std::string>()==PtcEmpiricalCoefficients::family_for(weighting),
        "POINT feedback belongs to a different weighting arm");
    const auto map_sha=ledger.bind(bundle_path.parent_path()/"map.fits",feedback["map_sha256"].as<std::string>());
    citlali::cli::detail::validate_point_feedback_provenance(feedback,bundle,{sha,bundle_sha,map_sha,observation,"sha256:"+health_sha,weighted_arrays[array],cycle-1});
    struct Handle {fitsfile *value=nullptr;~Handle(){if(value){int status=0;fits_close_file(value,&status);}}} handle;
    int status=0,any=0;fits_open_file(&handle.value,path.c_str(),READONLY,&status);auto *f=handle.value;
    require(!status,"POINT feedback FITS open failed");int ndim=0;long axes[2]{0,0};fits_get_img_dim(f,&ndim,&status);fits_get_img_size(f,2,axes,&status);
    auto number=[&](const char *k){double v=NAN;fits_read_key(f,TDOUBLE,k,&v,nullptr,&status);return v;};
    auto string=[&](const char *k){char v[FLEN_VALUE]{};fits_read_key(f,TSTRING,k,v,nullptr,&status);return std::string(v);};
    const double cx=number("CRPIX1"),cy=number("CRPIX2"),dx=number("CDELT1"),dy=number("CDELT2"),ox=number("CRVAL1"),oy=number("CRVAL2");
    const bool header=string("ASTFRAME")=="delta-source-AltAz" && string("CTYPE1")=="AZOFFSET" && string("CTYPE2")=="ELOFFSET" &&
        string("CUNIT1")=="arcsec" && string("CUNIT2")=="arcsec" && string("ARRAY")==weighted_arrays[array] && string("BUNIT")=="mJy/beam" && number("NOMBFWHM")==weighted_beams[array];
    require(!status && ndim==2 && axes[0]>0 && axes[1]>0 && header && dx==-pixel && dy==pixel && ox==0 && oy==0,"POINT feedback grid/frame/unit differs");
    require(static_cast<std::size_t>(axes[0])<=std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(axes[1]),"POINT feedback plane size overflow");
    const std::size_t size=static_cast<std::size_t>(axes[0])*static_cast<std::size_t>(axes[1]);std::vector<double> values(size);std::vector<std::uint8_t> valid(size),admitted(size);
    auto check=[&](){int data=0,hdu=0;fits_verify_chksum(f,&data,&hdu,&status);require(!status && data==1 && hdu==1,"POINT feedback FITS checksum differs");};
    check();fits_read_img(f,TDOUBLE,1,size,nullptr,values.data(),&any,&status);
    auto mask=[&](const char *name,std::vector<std::uint8_t> &v){fits_movnam_hdu(f,IMAGE_HDU,const_cast<char*>(name),0,&status);long shape[2]{};int dimensions=0;fits_get_img_dim(f,&dimensions,&status);fits_get_img_size(f,2,shape,&status);
        require(!status && dimensions==2 && shape[0]==axes[0] && shape[1]==axes[1] && string("BUNIT")=="1" && string("ARRAY")==weighted_arrays[array] &&
            string("ASTFRAME")=="delta-source-AltAz" && string("CTYPE1")=="AZOFFSET" && string("CTYPE2")=="ELOFFSET" &&
            string("CUNIT1")=="arcsec" && string("CUNIT2")=="arcsec" && number("CRPIX1")==cx && number("CRPIX2")==cy &&
            number("CDELT1")==dx && number("CDELT2")==dy && number("CRVAL1")==0 && number("CRVAL2")==0,"POINT feedback plane shape/frame differs");
        check();fits_read_img(f,TBYTE,1,size,nullptr,v.data(),&any,&status);};
    // The authenticated configured-feedback producer creates unique named
    // planes. Check that invariant here as well as the complete payload digest.
    mask("VALID",valid);mask("ADMITTED",admitted);int hdu_count=0;fits_get_num_hdus(f,&hdu_count,&status);unsigned valid_count=0,admitted_count=0;
    for(int hdu=1;hdu<=hdu_count;++hdu){int type=0;fits_movabs_hdu(f,hdu,&type,&status);char name[FLEN_VALUE]{};int key_status=0;fits_read_key(f,TSTRING,"EXTNAME",name,nullptr,&key_status);
        if(!key_status){valid_count+=std::string(name)=="VALID";admitted_count+=std::string(name)=="ADMITTED";}}
    require(!status && valid_count==1 && admitted_count==1,"POINT feedback named support planes differ");
    fits_close_file(f,&status);handle.value=nullptr;require(!status,"POINT feedback read failed");
    for(std::size_t i=0;i<size;++i)require(valid[i]<=1 && admitted[i]<=1 && (!admitted[i] || valid[i]) &&
        (!valid[i] || std::isfinite(values[i])) && (admitted[i] || !valid[i] || values[i]==0),"POINT feedback support/value mismatch");
    return FruitOffsetModel("sha256:"+sha,{std::size_t(axes[1]),std::size_t(axes[0]),"sha256:"+sha+":grid"},cx,cy,dx,dy,std::move(values),std::move(valid),40.);
}
YAML::Node point_noise_record(const PtcDetectorNoiseEvidence &e) {
    YAML::Node n;n["eligible_samples"]=e.eligible_count;n["source_free_samples"]=e.source_free_count;n["reference_samples"]=e.reference_count;n["remaining_samples"]=e.remaining_count;
    point_number(n,"reference_variance",e.reference_variance.value_or(NAN));point_number(n,"remaining_variance",e.remaining_variance.value_or(NAN));
    point_number(n,"cleaning_coefficient",e.cleaning_coefficient.value_or(NAN));point_number(n,"map_coefficient",e.map_coefficient.value_or(NAN));
    n["reference_reason"]=e.reference_reason;n["remaining_reason"]=e.remaining_reason;return n;
}
// Diagnostic rasters are separately authenticated signed source operands, not
// ordinary threshold-selected source publications. The native FRUIT owner
// retains the same AltAz containing-pixel projection used by ordinary feedback.
FruitOffsetModel point_recovery_raster(const YAML::Node &binding,MapFileLedger &ledger,
    const std::string &array) {
    const auto path=checked_file(binding);const auto sha=ledger.bind(path);
    const auto n=YAML::LoadFile(path.string());
    require(n["schema"].as<std::string>()=="citlali-point-recovery-raster-v1" &&
        n["array"].as<std::string>()==array && n["frame"].as<std::string>()=="delta-source-AltAz" &&
        n["unit"].as<std::string>()=="mJy/nominal beam","recovery raster identity differs");
    const auto values=n["values"].as<std::vector<double>>();
    require(values.size()==1681,"recovery raster shape differs");
    std::vector<std::uint8_t> valid(1681,1);
    for(std::size_t i=0;i<values.size();++i) {
        const double x=(20.-double(i%41))*2.,y=(double(i/41)-20.)*2.;
        require(std::isfinite(values[i]) && (std::hypot(x,y)<36. || values[i]==0.),
            "recovery raster nonfinite or outside prescribed taper");
    }
    return FruitOffsetModel("sha256:"+sha,{41,41,"recovery-grid:"+sha},21,21,-2,2,values,valid,40.);
}
int execute_weighted_point(const fs::path &input,const fs::path &output) {
    require(output.is_absolute() && !fs::exists(output),"weighted POINT needs a fresh absolute output");
    const fs::path staging=output.string()+".pending";require(!fs::exists(staging),"weighted POINT pending output exists");
    MapFileLedger ledger;const auto request_sha=ledger.bind(input);const auto request=YAML::LoadFile(input.string());
    const std::set<std::string> fields{"schema","stage","effective_config","networks","health","cycle_index","models","verbose","weighting_control","recovery"};
    for(const auto &e:request)require(fields.contains(e.first.as<std::string>()),"weighted POINT unknown request field");
    require(request["schema"].as<std::string>()=="citlali-configured-weighted-point-v1","weighted POINT schema differs");
    require(!request["recovery"] || request["stage"].as<std::string>()=="cycle","recovery cannot change health");
    const auto stage=request["stage"].as<std::string>();require(stage=="health" || stage=="cycle","weighted POINT unknown stage");
    const auto config_path=checked_file(request["effective_config"]);const auto config_sha=ledger.bind(config_path);const auto config=YAML::LoadFile(config_path.string());
    const auto cfg=config["successor"];require(cfg["schema"].as<std::string>()=="citlali-point-v2" && cfg["migration"].as<std::string>()=="replace-legacy-point-controls-v2" &&
        cfg["cleaning"]["method"].as<std::string>()=="noise-weighted-observed-als-v1","weighted POINT requires explicit configured profile");
    const bool comparison=cfg["weight_comparison"] && cfg["weight_comparison"].as<bool>();
    const auto control=request["weighting_control"]?request["weighting_control"].as<std::string>():"inverse-noise";
    require(control=="inverse-noise" || (comparison && stage=="cycle" && control=="equal-control"),
        "POINT equal weighting requires explicit comparison opt-in");
    const auto weighting=control=="equal-control"?PointWeighting::equal_control:PointWeighting::inverse_noise;
    const double threshold=cfg["detector_response"]["min_relative_response"].as<double>();
    const double pixel=cfg["map"]["pixel_size_arcsec"].as<double>(),cut=cfg["map"]["coverage_cut"].as<double>();
    require(std::isfinite(pixel) && pixel>0 && std::isfinite(cut) && cut>=0 && cut<=1,"weighted POINT grid request invalid");
    std::vector<WeightedPointNetwork> networks;std::set<int> seen;std::array<bool,3> arrays{};
    for(const auto &n:request["networks"]) {
        networks.push_back(point_load_cal(n,ledger,config_sha));auto &v=networks.back();
        require(seen.insert(v.saved.network).second && v.saved.observation_identity==networks.front().saved.observation_identity,"weighted POINT network/observation domain differs");arrays[v.saved.array]=true;
    }
    require(std::all_of(arrays.begin(),arrays.end(),[](bool v){return v;}),"weighted POINT requires all three arrays");
    std::sort(networks.begin(),networks.end(),[](const auto &a,const auto &b){return a.saved.network<b.saved.network;});
    fs::create_directories(staging);
    if(stage=="health") {
        std::vector<PointOpticalNetwork> inputs;for(auto &n:networks)inputs.push_back(std::move(n.optical));
        PointOpticalRequest health_request;health_request.min_relative_response=threshold;
        const auto health=PointOpticalPlan::consider(PointOpticalEvidence::learn(inputs,health_request));
        auto record=point_health_record(*health);record["effective_config_sha256"]=config_sha;record["request_sha256"]=request_sha;
        record["source_revision"]=std::string(CITLALI_GIT_VERSION);write_yaml(staging/"health.yaml",record);
        ledger.verify();fs::rename(staging,output);return 0;
    }
    const auto health_path=checked_file(request["health"]);const auto health_sha=ledger.bind(health_path);const auto health_record=YAML::LoadFile(health_path.string());
    require(health_record["effective_config_sha256"].as<std::string>()==config_sha,"POINT health config differs");
    const auto health=point_restore_health(health_record,threshold);require(health->evidence_handle()->networks().size()==networks.size(),"POINT health network cohort differs");
    const int cycle=request["cycle_index"].as<int>();require(cycle>=0 && (cycle==0 || (cfg["feedback"]["enabled"].as<bool>() && cycle<=cfg["feedback"]["max_iters"].as<int>())),"POINT cycle outside configured budget");
    require((cycle==0 && (!request["models"] || request["models"].size()==0)) || (cycle>0 && request["models"].size()==3),"POINT feedback model set differs");
    std::string recovery_role,recovery_sha;YAML::Node recovery;
    if(request["recovery"]) {
        const auto path=checked_file(request["recovery"]);recovery_sha=ledger.bind(path);
        recovery=YAML::LoadFile(path.string());
        citlali::cli::detail::validate_point_recovery(recovery,networks.front().saved.observation_identity,config_sha,health_sha,cycle);
        require(weighting==PointWeighting::inverse_noise && pixel==2. && cut==.15 &&
            cfg["feedback"]["max_iters"].as<int>()==10 && threshold==.1 &&
            cfg["cleaning"]["ranks"]["a1100"].as<int>()==3 &&
            cfg["cleaning"]["ranks"]["a1400"].as<int>()==4 &&
            cfg["cleaning"]["ranks"]["a2000"].as<int>()==6,"recovery recipe differs");
        recovery_role=recovery["case"].as<std::string>();
    }
    YAML::Node diagnostics;diagnostics["schema"]="citlali-weighted-point-cycle-v1";diagnostics["cycle_index"]=cycle;
    diagnostics["frozen_health_identity"]="sha256:"+health_sha;diagnostics["coefficient_family"]=std::string(PtcEmpiricalCoefficients::family_for(weighting));
    diagnostics["normalization_interpretation"]=std::string(PtcEmpiricalCoefficients::interpretation);diagnostics["request_sha256"]=request_sha;
    for(int array=0;array<3;++array) {
        const auto directory=staging/weighted_arrays[array];fs::create_directory(directory);
        double ex=0,ey=0;for(const auto &n:networks)if(n.saved.array==array)
            for(Eigen::Index t=0;t<n.optical.values.rows();++t)for(Eigen::Index d=0;d<n.optical.values.cols();++d)if(n.optical.eligible(t,d)) {
                const auto x=n.coordinates(t,2*d),y=n.coordinates(t,2*d+1);if(std::isfinite(x)&&std::isfinite(y)){ex=std::max(ex,std::abs(x));ey=std::max(ey,std::abs(y));}
            }
        const auto wcs=AstRtcTanGrid::source_offset(pixel,ex,ey);mapmaking::ForwardMapGrid grid{wcs.rows,wcs.columns,"CAL-health-independent-grid:"+config_sha+":"+weighted_arrays[array]};
        std::optional<FruitOffsetModel> model;if(cycle)model.emplace(point_read_model(request["models"][weighted_arrays[array]],ledger,array,3600*wcs.cdelt2_deg,networks.front().saved.observation_identity,health_sha,cycle,weighting));
        std::vector<double> values;std::vector<mapmaking::ForwardMapMember> members;std::size_t contributors=0;
        YAML::Node plan;plan["schema"]="citlali-weighted-point-map-plan-v1";plan["request_sha256"]=request_sha;plan["cycle_index"]=cycle;
        plan["frozen_health_identity"]="sha256:"+health_sha;plan["array"]=weighted_arrays[array];plan["observation"]=networks.front().saved.observation_identity;
        if(comparison)plan["weighting_control"]=control;
        plan["target_grid_policy"]="all science-supported original CAL coordinates, frozen before optical and noise exclusions";
        std::optional<FruitOffsetModel> injected;
        std::vector<FruitOffsetModel> reference_models;
        std::vector<std::vector<double>> reference_values;
        if(request["recovery"]) {
            injected.emplace(point_recovery_raster(recovery["arrays"][weighted_arrays[array]],ledger,weighted_arrays[array]));
            plan["diagnostic_recovery"]["identity"]="sha256:"+recovery_sha;
            plan["diagnostic_recovery"]["case"]=recovery_role;
            plan["diagnostic_recovery"]["seed_applied"]=recovery_role=="SEED" && cycle==1;
            if(recovery_role=="REF") {
                for(const auto &ref:recovery["weighted_references"][weighted_arrays[array]])
                    reference_models.push_back(point_recovery_raster(ref,ledger,weighted_arrays[array]));
                reference_values.resize(reference_models.size());
            }
        }
        if(model)plan["source_model"]=model->identity();else plan["source_model"]="M0:zero-operand";
        auto frame=plan["WCS"];frame["ASTFRAME"]="delta-source-AltAz";frame["CTYPE1"]="AZOFFSET";frame["CTYPE2"]="ELOFFSET";frame["CUNIT1"]="arcsec";frame["CUNIT2"]="arcsec";
        frame["CRPIX1"]=wcs.crpix1;frame["CRPIX2"]=wcs.crpix2;frame["CDELT1"]=3600*wcs.cdelt1_deg;frame["CDELT2"]=3600*wcs.cdelt2_deg;frame["CRVAL1"]=0.;frame["CRVAL2"]=0.;frame["rows"]=wcs.rows;frame["columns"]=wcs.columns;
        for(const auto &n:networks)if(n.saved.array==array) {
            const auto &s=n.saved;const auto &decisions=health->for_network(n.optical);std::size_t si=0;std::vector<bool> contributed(s.detectors.size(),false);
            for(const auto &segment:n.segments) {
                const auto rows=segment.slots.size(),cols=s.detectors.size();PtcMatrix original(rows,cols),operand=PtcMatrix::Zero(rows,cols);
                PtcMask eligible(rows,cols),source_free(rows,cols),model_supported=PtcMask::Ones(rows,cols);
                PtcMatrix added;PtcMask injection_supported;
                if(injected) {added=PtcMatrix::Zero(rows,cols);injection_supported=PtcMask::Ones(rows,cols);}
                std::size_t source_free_injected=0;double source_free_power=0.;
                for(std::size_t t=0;t<rows;++t)for(std::size_t d=0;d<cols;++d) {
                    const auto slot=segment.slots[t];original(t,d)=n.optical.values(slot,d);eligible(t,d)=n.optical.eligible(slot,d) && decisions[d].retained();source_free(t,d)=n.optical.source_free(slot,d);
                    if(injected && eligible(t,d)) {
                        const auto q=injected->at(3600*n.coordinates(slot,2*d),3600*n.coordinates(slot,2*d+1));
                        injection_supported(t,d)=q.support!=FruitModelSupport::unavailable;
                        require(injection_supported(t,d),"recovery projection unavailable on eligible original sample");
                        added(t,d)=q.value;
                        if(source_free(t,d) && q.value!=0.) {++source_free_injected;source_free_power+=q.value*q.value;}
                    }
                    if(model && eligible(t,d)) {const auto p=model->at(3600*n.coordinates(slot,2*d),3600*n.coordinates(slot,2*d+1));operand(t,d)=p.value;model_supported(t,d)=p.support!=FruitModelSupport::unavailable;}
                }
                if(injected) {
                    citlali::cli::detail::apply_point_recovery(recovery_role,cycle,
                        original,operand,eligible,added,injection_supported);
                }
                const auto identity="sha256:"+request_sha+":network="+std::to_string(s.network)+":segment="+std::to_string(si++);
                const auto residual=FruitResidual::subtract(identity+":CAL-minus-model",original,eligible,std::move(operand),std::move(model_supported));
                PtcSolverRequest solver;solver.rank=cfg["cleaning"]["ranks"][weighted_arrays[array]].as<int>();
                auto result=inverse_noise_fruit(residual,source_free,solver,identity,weighting);const auto &noise=result.noise;
                PtcMapPublicationIdentity id{identity,identity+":apply","sha256:"+s.cal_sha,s.rtc_identity,n.processing_identity,s.observation_identity,
                    "scan="+std::to_string(segment.scan)+":"+identity,identity+":retained",s.network,array,solver.rank,s.detectors,segment.slots};
                const auto q=PtcMapPublication::bind_verified_segment(std::move(id),std::move(result.total),noise.cleaned.causes,noise.fit.converged);
                std::vector<PtcEmpiricalDetectorRecord> records;YAML::Node evidence;evidence["identity"]=identity;evidence["network"]=s.network;evidence["scan"]=segment.scan;
                if(injected) {
                    evidence["recovery_source_free_nonzero"]=source_free_injected;
                    point_number(evidence,"recovery_source_free_squared_signal",source_free_power);
                    citlali::utils::Sha256 parent_hash;
                    parent_hash.update(reinterpret_cast<const std::uint8_t *>(original.data()),std::size_t(original.size())*sizeof(double));
                    evidence["recovery_parent_sha256"]=parent_hash.finish();
                }
                evidence["rank"]=solver.rank;evidence["converged"]=noise.fit.converged;evidence["stopping_reason"]=noise.fit.stopping_reason;evidence["retained_samples"]=q->retained();
                std::size_t unavailable=0;double lowest=INFINITY,highest=0;
                for(std::size_t d=0;d<cols;++d) {
                    records.push_back({s.detectors[d],noise.detectors[d]});const auto &nd=noise.detectors[d];
                    if(nd.map_coefficient){lowest=std::min(lowest,*nd.map_coefficient);highest=std::max(highest,*nd.map_coefficient);}
                    if(decisions[d].retained() && !nd.map_coefficient)++unavailable;
                    if(request["verbose"].as<bool>() || (decisions[d].retained() && !nd.map_coefficient)) {
                        auto e=point_noise_record(nd);e["detector_id"]=s.detectors[d];e["channel"]=s.channels[d];evidence["detector_details"].push_back(e);
                    }
                }
                evidence["health_retained_without_map_weight"]=unavailable;point_number(evidence,"map_weight_min",lowest);point_number(evidence,"map_weight_max",highest>0?highest:NAN);
                PtcMask mapped_support;
                if(comparison || injected) {
                    mapped_support=PtcMask::Zero(rows,cols);
                    evidence["input_support_sha256"]=point_support_digest(eligible);
                    evidence["source_free_sha256"]=point_support_digest(source_free);
                    evidence["residual_support_sha256"]=point_support_digest(residual->eligible());
                    evidence["cleaned_support_sha256"]=point_support_digest((noise.cleaned.causes==0).cast<std::uint8_t>());
                    citlali::utils::Sha256 domain;
                    for(const auto &detector:s.detectors)domain.update(std::to_string(detector.size())+":"+detector);
                    domain.update(":slots:");for(const auto slot:segment.slots)domain.update(std::to_string(slot)+",");
                    evidence["ordered_domain_sha256"]=domain.finish();
                }
                if(!q->retained()) {
                    if(comparison || injected)evidence["mapped_support_sha256"]=point_support_digest(mapped_support);
                    plan["segments"].push_back(evidence);continue;
                }
                const auto g=PtcEmpiricalCoefficients::publish(q,identity+":coefficients",identity+":noise",std::move(records),q->identity().support_reference,weighting);
                const auto admission=ValEmpiricalMapAdmission::bind(q,g,{q->identity().publication,q->identity().cal_parent,q->identity().rtc_grid,q->identity().detectors,q->identity().slots});
                for(std::size_t t=0;t<rows;++t)for(std::size_t d=0;d<cols;++d)if(q->member(t,d)) {
                    const auto slot=segment.slots[t];const auto decision=admission.at(t,d,s.cal_causes[d][slot]);if(!decision.pass())continue;
                    const auto weight=g->value(decision.coefficient_qc,t,d);require(weight.has_value(),"POINT admitted coefficient absent");
                    const auto [x,y]=wcs.continuous_pixel(n.coordinates(slot,2*d),n.coordinates(slot,2*d+1));
                    if(const auto p=mapmaking::containing_pixel(grid,x,y)) {members.push_back({values.size(),*p,*weight});values.push_back(q->values()(t,d));
                        if(comparison || injected)mapped_support(t,d)=1;
                        for(std::size_t k=0;k<reference_models.size();++k) {
                            const auto projected=reference_models[k].at(3600*n.coordinates(slot,2*d),3600*n.coordinates(slot,2*d+1));
                            // Independent full-grid cell-center evaluation checks the
                            // AST-to-source raster association, including half ties.
                            const double ax=(double(*p%grid.columns)+1-wcs.crpix1)*3600*wcs.cdelt1_deg;
                            const double ay=(double(*p/grid.columns)+1-wcs.crpix2)*3600*wcs.cdelt2_deg;
                            const auto expected=reference_models[k].at(ax,ay);
                            require(projected.support!=FruitModelSupport::unavailable && expected.support!=FruitModelSupport::unavailable &&
                                projected.value==expected.value,"recovery projection does not match MAP cell");
                            reference_values[k].push_back(projected.value);
                        }
                        if(!contributed[d]){contributed[d]=true;++contributors;}}
                }
                if(comparison || injected)evidence["mapped_support_sha256"]=point_support_digest(mapped_support);
                plan["segments"].push_back(evidence);
            }
            for(std::size_t d=0;d<contributed.size();++d)if(contributed[d])plan["contributing_channels"][std::to_string(s.network)].push_back(s.channels[d]);
        }
        const std::string domain="sha256:"+request_sha+":"+weighted_arrays[array]+":ordered-physical-PTC-support";
        const auto op=mapmaking::ForwardMapOperator::resolve(grid,domain,values.size(),std::move(members),{cut,{}});
        require(!op.output_rows().empty(),"POINT required array has no supported weighted map: "+weighted_arrays[array]);const auto result=op.apply(domain,values);
        for(std::size_t k=0;k<reference_models.size();++k) {
            const auto mapped=op.apply(domain,reference_values[k]);double error=0.;
            for(std::size_t j=0;j<op.output_rows().size();++j) {
                const auto p=op.output_rows()[j];
                const double ax=(double(p%grid.columns)+1-wcs.crpix1)*3600*wcs.cdelt1_deg;
                const double ay=(double(p/grid.columns)+1-wcs.crpix2)*3600*wcs.cdelt2_deg;
                const auto expected=reference_models[k].at(ax,ay).value;
                error=std::max(error,std::abs(mapped.signal[j]-expected));
                require(op.normalization()[p]>0 && std::abs(mapped.signal[j]-expected)<=1e-10*std::max(1.,std::abs(expected)),
                    "weighted injection MAP identity failed");
            }
            auto verified=YAML::Node();verified["raster"]=reference_models[k].identity();
            verified["actual_ref_members"]=op.members().size();verified["supported_pixels"]=op.output_rows().size();
            verified["max_absolute_error"]=error;verified["numerical_rtol_floor_one"]=1e-10;
            plan["diagnostic_recovery"]["weighted_references"].push_back(verified);
        }
        write_yaml(directory/"plan.yaml",plan);map_fits(directory/"map.fits",wcs,op,result,weighted_arrays[array],weighted_beams[array],true);
        YAML::Node bundle;bundle["schema"]="citlali-forward-map-bundle-v1";bundle["role"]="SCI-MAP:base_observation_map@1";bundle["outcome"]="success";
        bundle["array"]=weighted_arrays[array];bundle["observation"]=networks.front().saved.observation_identity;bundle["WCS"]=frame;
        bundle["nominal_beam_fwhm_arcsec"]=weighted_beams[array];bundle["signal_unit"]="mJy/beam";bundle["measured_beam_or_full_response_inferred"]=false;
        bundle["coefficient_family"]=std::string(PtcEmpiricalCoefficients::family_for(weighting));bundle["normalization_interpretation"]=std::string(PtcEmpiricalCoefficients::interpretation);
        bundle["cycle_index"]=cycle;bundle["frozen_health_identity"]="sha256:"+health_sha;bundle["contributing_detectors"]=contributors;bundle["contributing_samples"]=values.size();
        for(const auto *name:{"map.fits","plan.yaml"})bundle["files"][name]=citlali::utils::sha256_file(directory/name);
        bundle["product_identity"]="weighted-POINT:"+request_sha+":"+bundle["files"]["map.fits"].as<std::string>();
        write_yaml(directory/"bundle.yaml",bundle);diagnostics["arrays"][weighted_arrays[array]]=bundle;
    }
    write_yaml(staging/"cycle.yaml",diagnostics);ledger.verify();fs::rename(staging,output);return 0;
}
} // namespace
