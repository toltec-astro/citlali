// Private saved-product application adapter. Scientific arithmetic remains in
// PTC, AST and MAP owners; this boundary parses and verifies persisted sources.
#include <citlali/core/pipeline/ptc_map_handoff.h>
#include <citlali/core/pipeline/val_map_handoff.h>
#include <citlali/core/mapmaking/forward_map.h>
#include <citlali/core/mapmaking/forward_map_admission.h>
#include <fitsio.h>
namespace {
namespace mapmaking=citlali::mapmaking;
struct MapFileLedger {
    std::map<std::string,std::string> files;
    std::string bind(const fs::path &path,const std::string &expected={}) {
        const auto digest=citlali::utils::sha256_file(path);
        require(expected.empty() || digest==expected,"MAP source digest mismatch: "+path.string());
        const auto [it,inserted]=files.emplace(fs::absolute(path).string(),digest);
        require(inserted || it->second==digest,"MAP source changed during read");return digest;
    }
    void verify() const {for(const auto &[p,h]:files)require(citlali::utils::sha256_file(p)==h,"MAP source changed during application: "+p);}
};
YAML::Node map_prefix(const fs::path &path) {
    std::ifstream f(path);require(bool(f),"MAP receipt unavailable");
    std::string line,text;
    while(std::getline(f,line)) {if(line.starts_with("spectra:"))break;text+=line+'\n';}
    return YAML::Load(text);
}
// Extract only named top-level fields of a receipt or a named map section.
// Large diagnostic spectra/relearning payloads are not parsed or consumed.
YAML::Node map_selected_fields(const fs::path &path,const std::string &section,
                              const std::set<std::string> &fields) {
    std::ifstream f(path);require(bool(f),"MAP receipt unavailable");
    std::string line,text;bool inside=section.empty(),keep=false;
    const std::size_t indent=section.empty()?0:2;
    while(std::getline(f,line)) {
        if(!section.empty() && line==section+":"){inside=true;continue;}
        if(!inside)continue;
        const auto start=line.find_first_not_of(' ');
        if(start==std::string::npos)continue;
        if(start<indent)break;
        if(start==indent) {
            const auto colon=line.find(':',start);
            keep=colon!=std::string::npos && fields.contains(line.substr(start,colon-start));
        }
        if(keep)text+=line.substr(indent)+'\n';
    }
    return YAML::Load(text);
}
template<class T> std::vector<T> map_read_array(const fs::path &p,std::size_t count) {
    require(std::endian::native==std::endian::little,"saved MAP reader requires declared little-endian host");
    require(count<=std::numeric_limits<std::size_t>::max()/sizeof(T) && fs::file_size(p)==count*sizeof(T),"MAP component shape/byte length mismatch: "+p.string());
    std::vector<T> out(count);std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(out.data()),count*sizeof(T));
    require(bool(f),"MAP required component read failed");return out;
}
template<class T> void map_write_array(const fs::path &p,std::span<const T> data) {
    std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size_bytes());f.close();
    require(bool(f),"MAP required component write failed: "+p.string());
}
struct SavedMapNetwork {
    fs::path root,ptcroot;
    YAML::Node input,learn,cal,ptc,apply;
    std::string ptc_sha,cal_sha,rtc_identity;
    std::vector<double> native_times;
    std::vector<int> channels;
    std::vector<std::string> detectors;
    std::vector<std::vector<std::uint16_t>> cal_causes;
    std::vector<std::pair<double,double>> geometry;
    CalObservationInput observation;
    int network,array,factor,obsnum;
    std::int64_t first;
    std::size_t scheduled;
};
SavedMapNetwork load_saved_map_network(const YAML::Node &request,MapFileLedger &ledger) {
    SavedMapNetwork out;out.root=request["root"].as<std::string>();out.ptcroot=out.root/"donor-continuity"/"ptc";
    require(out.root.is_absolute(),"MAP source root must be absolute");
    ledger.bind(out.root/"receipt.yaml",request["receipt_sha256"].as<std::string>());
    const auto final=map_prefix(out.root/"receipt.yaml");
    require(final["completed_through"].as<std::string>()=="PTC" && !final["legacy_fallback_performed"].as<bool>(),"MAP requires an actual completed successor PTC route");
    const auto input_path=checked_file(request["input"]);ledger.bind(input_path,final["configuration_sha256"].as<std::string>());
    out.input=YAML::LoadFile(input_path.string());
    ledger.bind(out.root/"learning-receipt.yaml");out.learn=map_prefix(out.root/"learning-receipt.yaml");
    for(const auto *key:{"configuration_sha256","original_parent","native_time_sha256","rows",
                         "network","observation","detectors","native_runs"})
        require(out.learn[key] && final[key] && YAML::Dump(out.learn[key])==YAML::Dump(final[key]),
                std::string("MAP learning/final identity mismatch: ")+key);
    const auto apply_path=out.root/"donor-continuity"/"apply-receipt.yaml";
    ledger.bind(apply_path);
    const std::set<std::string> apply_fields{"original_parent","attempt","realized_detectors","output_grid"};
    out.apply=map_selected_fields(apply_path,"",apply_fields);
    const auto final_apply=map_selected_fields(out.root/"receipt.yaml","donor_continuity",apply_fields);
    for(const auto &key:apply_fields)
        require(out.apply[key] && final_apply[key] && YAML::Dump(out.apply[key])==YAML::Dump(final_apply[key]),
                "MAP apply/final identity mismatch: "+key);
    require(out.apply["original_parent"].as<std::string>()==out.learn["original_parent"].as<std::string>(),"MAP RTC parent mismatch");
    out.ptc_sha=ledger.bind(out.ptcroot/"receipt.yaml",request["ptc_receipt_sha256"].as<std::string>());
    out.ptc=YAML::LoadFile((out.ptcroot/"receipt.yaml").string());
    out.cal_sha=ledger.bind(out.root/"donor-continuity/cal/receipt.yaml",out.ptc["source_CAL_receipt_sha256"].as<std::string>());
    out.cal=YAML::LoadFile((out.root/"donor-continuity/cal/receipt.yaml").string());
    require(out.ptc["schema"].as<std::string>()=="citlali-ptc-output-v2" && out.cal["schema"].as<std::string>()=="citlali-cal-output-v1" &&
        out.ptc["rank"].as<int>()>0 && (out.ptc["method"].as<std::string>()=="observed-als" || (out.ptc["method"].as<std::string>()=="masked-covariance-count-v1" && out.input["observation"].as<int>()==152430 && (out.input["network"].as<int>()==11 || out.input["network"].as<int>()==12))) &&
        out.ptc["unit"].as<std::string>()=="mJy/nominal-beam" && out.cal["unit"].as<std::string>()=="mJy/nominal-beam" &&
        out.ptc["centering"].as<std::string>()=="per-detector-eligible-arithmetic-mean-not-restored" &&
        out.ptc["use_policy"].as<std::string>()=="ptc-cal-observed-entry-use-2026-09-18-v2" &&
        out.ptc["scaling"].as<std::string>()=="identity" && out.ptc["metric"].as<std::string>()=="binary-eligible-Euclidean" &&
        out.ptc["excluded_storage"].as<std::string>()=="zero-sentinel-never-a-measurement;eligibility-and-causes-authoritative" &&
        out.ptc["input_VAL_generation"].as<std::uint64_t>()==out.cal["output_VAL_generation"].as<std::uint64_t>() &&
        out.cal["input_RTC_attempt"].as<std::uint64_t>()==out.apply["attempt"].as<std::uint64_t>(),"MAP PTC/CAL/RTC publication identity or method incompatible");
    require(YAML::Dump(final["PTC"])==YAML::Dump(out.ptc) && YAML::Dump(final["CAL"])==YAML::Dump(out.cal),"MAP final receipt differs from required child publication");
    out.network=out.learn["network"].as<int>();out.array=out.cal["detectors"][0]["array"].as<int>();
    out.obsnum=out.learn["observation"].as<int>();
    require(((out.input["development_reference"] && out.input["development_reference"].as<std::string>()=="bright-point-base-v1") || out.obsnum==152389 || out.obsnum==152390 || (out.obsnum==152430 && out.array==2 && (out.network==11 || out.network==12))) && out.input["observation"].as<int>()==out.obsnum &&
        out.array>=0 && out.array<3,"MAP owner binding requires selected observation and known array");
    const auto telescope=checked_file(out.input["telescope"]),effective=checked_file(out.input["effective_config"]),manifest=checked_file(out.input["manifest"]);
    ledger.bind(telescope);ledger.bind(effective);const auto aptsha=ledger.bind(manifest);
    require(out.cal["AST_telescope"].as<std::string>()=="sha256:"+citlali::utils::sha256_file(telescope) &&
        out.cal["selected_child_APT"].as<std::string>()=="sha256:"+aptsha &&
        out.cal["AST_method"].as<std::string>()==std::string(AstRtcCoordinates::method),"MAP AST/calibration source differs");
    const auto effective_cfg=YAML::LoadFile(effective.string());unsigned offset_count=0;
    for(const auto &entry:effective_cfg["interface_sync_offset"]){const auto offset=entry["toltec"+std::to_string(out.network)];
        if(offset){require(offset.as<double>()==0,"saved MAP adapter only supports verified zero interface offset");++offset_count;}}
    require(offset_count==1,"saved MAP interface offset absent or duplicated");
    out.observation=prepare_cal_observation(out.input,NativeObservationScope{out.obsnum,0,2});
    require(out.cal["AST_center_ra_rad"].as<double>()==out.observation.ra0 && out.cal["AST_center_dec_rad"].as<double>()==out.observation.dec0 &&
        out.cal["AST_pointing_offset"].as<std::string>()==out.observation.offset_identity,"MAP AST center/offset generation differs");
    const auto verified=apt::verify_bundle_filesystem(manifest,true);
    const auto apt_relation=pipeline::admit_canonical_apt_detector_relation_v2(verified);
    require(apt_relation.observation().observation==out.obsnum && apt_relation.observation().subobservation==0 &&
        apt_relation.observation().scan==2,"MAP selected APT observation differs from PTC source");
    for(const auto &[name,unit]:std::array<std::pair<std::string,std::string>,3>{{{"flxscale","mJy/beam/xs"},{"x_t","arcsec"},{"y_t","arcsec"}}}) {
        const auto f=std::find_if(verified.fields.begin(),verified.fields.end(),[&](const auto &v){return v.name==name;});
        require(f!=verified.fields.end() && f->unit==unit,"MAP selected APT field units differ");
    }
    const auto n=out.learn["rows"].as<std::size_t>();
    ledger.bind(out.root/"native-time.f64",out.learn["native_time_sha256"].as<std::string>());
    out.native_times=map_read_array<double>(out.root/"native-time.f64",n);
    require(n>0 && out.learn["native_runs"][0][0].as<std::int64_t>()==0,"saved MAP native origin unsupported");
    for(std::size_t i=0;i<n;++i)require(std::isfinite(out.native_times[i]) && (!i || out.native_times[i]>out.native_times[i-1]),"MAP native times invalid");
    const auto ldet=out.learn["detectors"],cdet=out.cal["detectors"],adet=out.apply["realized_detectors"],gdet=out.apply["output_grid"]["detectors"];
    require(ldet.size()==cdet.size() && adet.size()==ldet.size() && gdet.size()==ldet.size(),"MAP detector domains differ");
    std::set<int> seen;
    for(std::size_t d=0;d<ldet.size();++d) {
        const int ch=ldet[d]["channel"].as<int>();require(seen.insert(ch).second,"MAP repeated channel");
        require(ldet[d]["detector"].as<std::size_t>()==d && cdet[d]["channel"].as<int>()==ch && adet[d]["channel"].as<int>()==ch &&
            gdet[d]["channel"].as<int>()==ch && cdet[d]["array"].as<int>()==out.array,"MAP detector association mismatch");
        const auto factor=adet[d]["factor"].as<int>();const auto first=adet[d]["phase_native_rows"].as<std::int64_t>();
        const auto scheduled=gdet[d]["scheduled_slots"].as<std::size_t>();
        require(factor>0 && first>=0 && first<factor && scheduled==(n-1-first)/factor+1,"MAP saved RTC schedule invalid");
        if(d)require(factor==out.factor && first==out.first && scheduled==out.scheduled,"MAP PTC group has differing RTC schedules");
        else {out.factor=factor;out.first=first;out.scheduled=scheduled;}
        const auto occurrence=ldet[d]["occurrence"].as<std::string>();const auto row_text=occurrence.substr(occurrence.rfind(":row=")+5);
        const auto uid=std::stoll(row_text);const auto row=std::find_if(verified.apt.rows.begin(),verified.apt.rows.end(),[&](const auto &r){return r.uid==uid;});
        require(row!=verified.apt.rows.end() && row->network==out.network && row->channel==ch && row->array==out.array &&
            cdet[d]["selected_row"].as<std::string>()=="sha256:"+aptsha+":row="+row_text,"MAP selected detector/APT row mismatch");
        out.channels.push_back(ch);out.detectors.push_back(occurrence+":association="+ldet[d]["array_association"].as<std::string>());
        out.geometry.emplace_back(field(*row,"x_t"),field(*row,"y_t"));
        const auto name="channel-"+std::to_string(ch)+"-causes.u16";const auto p=out.root/"donor-continuity/cal"/name;
        ledger.bind(p,cdet[d]["files"][name].as<std::string>());out.cal_causes.push_back(map_read_array<std::uint16_t>(p,scheduled));
    }
    out.rtc_identity="sha256:"+ledger.files.at(fs::absolute(apply_path).string())+":attempt="+std::to_string(out.apply["attempt"].as<std::uint64_t>());
    return out;
}
struct SavedSegmentPlan {
    std::size_t network_index,segment,retained;
    fs::path coordinates,publication,coefficient;
    std::string identity,support,coordinate_sha;
};
std::shared_ptr<const PtcMapPublication> load_map_segment(const SavedMapNetwork &n,std::size_t index,MapFileLedger &ledger) {
    const auto s=n.ptc["segments"][index];const auto times=s["scheduled_times"].as<std::size_t>();
    const auto cols=s["output_detector_indices"].as<std::vector<std::size_t>>();
    require(times>0 && cols.size()==s["output_detectors"].as<std::size_t>() && s["network"].as<int>()==n.network,"MAP PTC segment domain mismatch");
    const auto stem="segment-"+std::to_string(index);std::set<std::string> expected;
    for(const auto *suffix:{"-input.f64","-centered.f64","-mean.f64","-basis.f64","-cleaned.f64","-eligible.u8","-causes.u8","-slots.i64"})expected.insert(stem+suffix);
    require(s["files"].size()==expected.size(),"MAP PTC component inventory incomplete");
    for(const auto &entry:s["files"]) {
        const auto name=entry.first.as<std::string>();require(expected.erase(name)==1,"MAP unexpected or repeated PTC component");
        ledger.bind(n.ptcroot/name,entry.second.as<std::string>());
    }
    require(expected.empty(),"MAP required PTC component absent");
    PtcMapPublicationIdentity id;id.publication="sha256:"+n.ptc_sha+":"+stem;
    id.application="sha256:"+n.ptc_sha;id.cal_parent="sha256:"+n.cal_sha;id.rtc_grid=n.rtc_identity;
    id.processing=n.ptc["processing_generation"].as<std::string>();id.observation="obs:"+std::to_string(n.obsnum)+"_0_2";
    id.segment="scan="+s["scan"].as<std::string>()+":"+stem;id.network=n.network;id.array=n.array;id.rank=n.ptc["rank"].as<int>();
    id.support_reference=id.publication+":complete-retained-support:causes="+s["files"][stem+"-causes.u8"].as<std::string>();
    id.slots=map_read_array<std::int64_t>(n.ptcroot/(stem+"-slots.i64"),times);
    const auto interval=s["native_interval"].as<std::vector<std::int64_t>>();
    require(interval.size()==2 && interval[0]>=0 && interval[0]<interval[1],"MAP invalid PTC native interval");
    for(auto slot:id.slots) {
        require(slot>=0 && static_cast<std::size_t>(slot)<n.scheduled,"MAP RTC slot outside exact saved grid");
        const auto row=n.first+slot*n.factor;require(row>=interval[0] && row<interval[1] && static_cast<std::size_t>(row)<n.native_times.size(),"MAP PTC slot/native relation differs");
    }
    std::set<std::size_t> seen;
    for(auto d:cols){require(d<n.detectors.size() && seen.insert(d).second,"MAP PTC output detector index invalid");id.detectors.push_back(n.detectors[d]);}
    const auto values=map_read_array<double>(n.ptcroot/(stem+"-cleaned.f64"),times*cols.size());
    const auto causes=map_read_array<std::uint8_t>(n.ptcroot/(stem+"-causes.u8"),times*cols.size());
    auto q=PtcMapPublication::bind_verified_segment(std::move(id),
        Eigen::Map<const PtcMatrix>(values.data(),times,cols.size()),Eigen::Map<const PtcMask>(causes.data(),times,cols.size()),s["converged"].as<bool>());
    require(q->retained()==s["retained"].as<std::size_t>(),"MAP PTC retained support mismatch");return q;
}
void map_fits(const fs::path &path,const AstRtcTanGrid &wcs,const mapmaking::ForwardMapOperator &op,
    const mapmaking::ForwardMapValues &result,const std::string &array,double beam) {
    fitsfile *f=nullptr;int status=0;
    fits_create_file(&f,path.c_str(),&status);require(!status,"MAP required FITS create failed");
    const auto size=wcs.rows*wcs.columns;
    std::vector<double> signal(size,NAN);for(std::size_t i=0;i<op.output_rows().size();++i)signal[op.output_rows()[i]]=result.signal[i];
    auto plane=[&](const char *name,int bitpix,int datatype,const void *data,const char *unit) {
        long axes[2]{static_cast<long>(wcs.columns),static_cast<long>(wcs.rows)};
        fits_create_img(f,bitpix,2,axes,&status);
        if(*name)fits_update_key_str(f,"EXTNAME",name,nullptr,&status);
        fits_update_key_str(f,"CTYPE1","RA---TAN",nullptr,&status);fits_update_key_str(f,"CTYPE2","DEC--TAN",nullptr,&status);
        fits_update_key_str(f,"CUNIT1","deg",nullptr,&status);fits_update_key_str(f,"CUNIT2","deg",nullptr,&status);
        fits_update_key_str(f,"RADESYS","FK5",nullptr,&status);fits_update_key_dbl(f,"EQUINOX",2000,15,nullptr,&status);
        for(const auto &[key,value]:std::array<std::pair<const char*,double>,6>{{{"CRPIX1",wcs.crpix1},{"CRPIX2",wcs.crpix2},{"CRVAL1",wcs.crval1_deg},{"CRVAL2",wcs.crval2_deg},{"CDELT1",wcs.cdelt1_deg},{"CDELT2",wcs.cdelt2_deg}}})
            fits_update_key_dbl(f,key,value,15,nullptr,&status);
        fits_update_key_str(f,"BUNIT",unit,nullptr,&status);fits_update_key_str(f,"ARRAY",array.c_str(),nullptr,&status);
        fits_update_key_dbl(f,"NOMBFWHM",beam,15,"Fixed nominal calibration FWHM, arcsec; not measured PSF",&status);
        fits_write_img(f,datatype,1,size,const_cast<void*>(data),&status);fits_write_chksum(f,&status);
    };
    plane("",DOUBLE_IMG,TDOUBLE,signal.data(),"mJy/beam");
    plane("NUMERATOR",DOUBLE_IMG,TDOUBLE,result.numerator.data(),"mJy/beam");
    plane("NORMALIZATION",DOUBLE_IMG,TDOUBLE,op.normalization().data(),"1");
    std::vector<long long> hits(op.contributing_hits().begin(),op.contributing_hits().end());
    plane("HITS",LONGLONG_IMG,TLONGLONG,hits.data(),"count");
    plane("NORM_SUPPORT",BYTE_IMG,TBYTE,op.normalization_support().data(),"1");
    plane("SCI_SUPPORT",BYTE_IMG,TBYTE,op.science_support().data(),"1");
    int close_status=0;fits_close_file(f,&close_status);require(!status && !close_status,"MAP required FITS publication failed");
    // Reopen and verify actual signal bytes; completion follows required I/O.
    fits_open_file(&f,path.c_str(),READONLY,&status);std::vector<double> reread(size);int any=0;
    fits_read_img(f,TDOUBLE,1,size,nullptr,reread.data(),&any,&status);fits_close_file(f,&status);
    require(!status,"MAP required FITS reload failed");
    for(std::size_t i=0;i<size;++i)require(reread[i]==signal[i] || (std::isnan(reread[i]) && std::isnan(signal[i])),"MAP FITS signal reload mismatch");
}
int execute_saved_map(const fs::path &input,const fs::path &output) {
    const auto started=std::chrono::steady_clock::now();const auto cpu=std::clock();
    require(!fs::exists(output) && output.is_absolute(),"MAP output must be a new absolute directory");
    const fs::path staging=output.string()+".pending";require(!fs::exists(staging),"MAP pending directory already exists");
    MapFileLedger ledger;const auto request_sha=ledger.bind(input);const auto request=YAML::LoadFile(input.string());
    require(request["schema"].as<std::string>()=="citlali-map-saved-ptc-v1","MAP requires an exact saved-PTC input request");
    const std::set<std::string> fields{"schema","owner_binding","coverage_cut","pixel_size_arcsec","networks"};
    for(const auto &entry:request)require(fields.contains(entry.first.as<std::string>()),"MAP unknown request field");
    const auto owner_path=checked_file(request["owner_binding"]);const auto owner_sha=ledger.bind(owner_path);
    const auto owner=YAML::LoadFile(owner_path.string());
    require(owner["identity"].as<std::string>()=="SCI-MAP-FIRST-FORWARD-OWNER-2026-09-22" &&
        owner["frame"]["reference_system"].as<std::string>()=="FK5" && owner["frame"]["equinox"].as<double>()==2000 &&
        owner["frame"]["source_already_in_frame"].as<bool>() && owner["frame"]["transformation"].as<std::string>()=="identity" &&
        owner["frame"]["source_fields"].as<std::vector<std::string>>()==std::vector<std::string>{"SourceRaAct","SourceDecAct"} &&
        owner["measured_beams_are_separate"].as<bool>() && !owner["realized_response_inferred"].as<bool>() &&
        owner["coefficient_family"].as<std::string>()==std::string(PtcUniformCoefficients::family),"MAP selected owner frame/coefficient binding differs");
    const std::array<std::string,3> arrays{"a1100","a1400","a2000"};const std::array<double,3> beams{5,7.5,10};
    for(int i=0;i<3;++i)require(owner["nominal_beam_fwhm_arcsec"][arrays[i]].as<double>()==beams[i],"MAP nominal-beam binding differs");
    const double cut=request["coverage_cut"].as<double>(),pixel=request["pixel_size_arcsec"].as<double>();
    require(std::isfinite(cut) && cut>=0 && cut<=1 && std::isfinite(pixel) && pixel>0,"MAP cut or pixel request invalid; above-one override route not implemented");
    require(request["networks"].IsSequence() && request["networks"].size()>0,"MAP needs explicit saved network membership");
    fs::create_directories(staging);YAML::Node attempt;
    attempt["schema"]="citlali-forward-map-attempt-v1";attempt["request_identity"]="sha256:"+request_sha;
    attempt["owner_binding"]="sha256:"+owner_sha;attempt["MAP_authority"]="SCI-MAP-v0.2-r0.4";
    attempt["source_revision"]=std::string(CITLALI_GIT_VERSION);attempt["outcome"]="incomplete";
    attempt["FRUIT_executed"]=false;attempt["upstream_science_reexecuted"]=false;
    attempt["reached_stages"].push_back("requested");write_yaml(staging/"attempt.yaml",attempt);
    try {
        std::vector<SavedMapNetwork> networks;std::set<int> seen;
        for(const auto &n:request["networks"]) {
            networks.push_back(load_saved_map_network(n,ledger));require(seen.insert(networks.back().network).second,"MAP repeated network publication");
        }
        std::sort(networks.begin(),networks.end(),[](const auto &a,const auto &b){return a.network<b.network;});
        for(const auto &n:networks)require(n.obsnum==networks.front().obsnum,"MAP cannot combine different observations in a base observation map");
        const auto observation_identity="obs:"+std::to_string(networks.front().obsnum)+"_0_2";
        const auto loaded=std::chrono::steady_clock::now();
        attempt["input_loading_seconds"]=std::chrono::duration<double>(loaded-started).count();
        attempt["reached_stages"].push_back("effective");
        std::size_t products=0;
        for(int array=0;array<3;++array) {
            std::size_t expected_retained=0;for(const auto &n:networks)if(n.array==array)expected_retained+=n.ptc["available"].as<std::size_t>();
            if(!expected_retained)continue;
            const auto array_started=std::chrono::steady_clock::now();
            const auto directory=staging/arrays[array];fs::create_directory(directory);fs::create_directory(directory/"handoff");
            std::vector<SavedSegmentPlan> plans;double extent_x=0,extent_y=0;std::size_t admitted=0,invalid_coordinate=0,retained=0;
            double ra0=NAN,dec0=NAN;YAML::Node sources;
            for(std::size_t ni=0;ni<networks.size();++ni) {
                const auto &n=networks[ni];if(n.array!=array)continue;
                if(std::isfinite(ra0))require(ra0==n.observation.ra0 && dec0==n.observation.dec0,"MAP input center differs within observation");
                else {ra0=n.observation.ra0;dec0=n.observation.dec0;}
                YAML::Node source;source["network"]=n.network;source["root"]=n.root.string();source["PTC_receipt_sha256"]=n.ptc_sha;
                source["upstream_build"]=n.ptc["build_identity"].as<std::string>();source["failed_fits"]=n.ptc["failed_fits"].as<std::size_t>();
                source["detectors"]=n.detectors.size();sources.push_back(source);
                std::set<std::int64_t> used_slots;
                for(std::size_t si=0;si<n.ptc["segments"].size();++si) {
                    const auto q=load_map_segment(n,si,ledger);retained+=q->retained();
                    for(auto slot:q->identity().slots)require(used_slots.insert(slot).second,"MAP overlapping PTC segments");
                    if(!q->retained())continue;
                    const auto name="network-"+std::to_string(n.network)+"-segment-"+std::to_string(si);
                    const auto pub=directory/"handoff"/(name+"-ptc.yaml"),coef=directory/"handoff"/(name+"-coefficient.yaml");
                    YAML::Node publication;publication["schema"]="citlali-ptc-map-segment-v1";publication["identity"]=q->identity().publication;
                    publication["publication_unit"]="complete-segment";publication["application"]=q->identity().application;
                    publication["CAL_parent"]=q->identity().cal_parent;publication["RTC_grid"]=q->identity().rtc_grid;
                    publication["observation"]=q->identity().observation;publication["segment"]=q->identity().segment;
                    publication["processing"]=q->identity().processing;publication["array"]=arrays[array];publication["network"]=n.network;
                    publication["rank"]=q->identity().rank;publication["detector_occurrences"]=q->identity().detectors;
                    publication["source_PTC_root"]=n.ptcroot.string();publication["source_PTC_receipt_sha256"]=n.ptc_sha;
                    publication["source_segment"]=YAML::Clone(n.ptc["segments"][si]);publication["support_reference"]=q->identity().support_reference;
                    publication["support_definition"]="every and only cause==0 cell, row-major through exact RTC slots and detector occurrences";
                    publication["retained"]=q->retained();publication["selection_lineage"]="sha256:"+owner_sha;
                    publication["source_CAL_root"]=(n.root/"donor-continuity/cal").string();publication["source_CAL_receipt_sha256"]=n.cal_sha;
                    publication["RTC_native_phase"]=n.first;publication["RTC_factor"]=n.factor;publication["RTC_scheduled_slots"]=n.scheduled;
                    publication["complete"]=true;write_yaml(pub,publication);
                    const auto g=PtcUniformCoefficients::publish(q,{"sha256:"+owner_sha,owner["coefficient_family"].as<std::string>()},
                        q->identity().publication+":uniform:selection="+owner_sha,q->identity().support_reference);
                    YAML::Node coefficient;coefficient["identity"]=g->identity();coefficient["PTC_publication_sha256"]=citlali::utils::sha256_file(pub);
                    coefficient["family"]=std::string(g->family);coefficient["registry"]=std::string(g->registry);coefficient["admission"]=std::string(g->admission);
                    coefficient["source_sha256"]=std::string(g->source_sha256);coefficient["profile"]=std::string(g->profile);
                    coefficient["scalar"]=1.;coefficient["unit"]="1";coefficient["normalization"]="analytic-arithmetic-mean-to-one";
                    coefficient["stored_support_reference"]=q->identity().support_reference;
                    coefficient["consumer_permission"]="SCI-MAP:GRANTED:UFC-OD-008";
                    coefficient["selection"]=g->selection().identity;coefficient["fixed_coefficient_sampling_uncertainty"]="none";
                    coefficient["interpretation"]="nonprecision gridding coefficient; no inverse variance, sensitivity or response meaning";
                    write_yaml(coef,coefficient);
                    const auto reload=YAML::LoadFile(coef.string());require(reload["scalar"].as<double>()==1 && reload["unit"].as<std::string>()=="1" &&
                        reload["stored_support_reference"].as<std::string>()==q->identity().support_reference,"PTC uniform payload fidelity failed");
                    YAML::Node qc;qc["schema"]="citlali-ptc-uniform-handoff-evaluation-v1";
                    qc["identity"]=g->identity()+":QC:SCI-MAP";qc["profile"]=std::string(g->profile);
                    qc["payload_sha256"]=citlali::utils::sha256_file(coef);qc["PTC_publication_sha256"]=citlali::utils::sha256_file(pub);
                    qc["VAL_registry_sha256"]="a8fdf512507c877c22aa5457566f66782f89a466504115812b5dc7b5d46dfb5f";
                    qc["VAL_registry"]="SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-noi-stage-a-r0.18-ptc-uniform-r0.4-2026-09-06";
                    qc["source_register_sha256"]="fb13be222a9a8bb928fa172253f65aaf6ede081a88e901cdf6eb3799fad35f6e";
                    qc["common_named_use_sha256"]="c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c";
                    qc["consumer_composition"]="PTC-UNIFORM-CONSUMER-COMPOSITION r0.4/2026-09-06";
                    UniformDecision shared_qc;bool found_member=false;
                    for(std::size_t t=0;t<q->identity().slots.size() && !found_member;++t)
                        for(std::size_t d=0;d<q->identity().detectors.size();++d)if(q->member(t,d)) {
                            shared_qc=ValUniformHandoffEvaluator::evaluate(*g,q,t,d,true);found_member=true;break;
                        }
                    require(found_member && shared_qc.pass(),"PTC uniform shared evaluation failed");
                    qc["requested"]=shared_qc.requested?"requested":"not_requested";qc["realization"]=shared_qc.realized?"realized":"not_produced";
                    qc["logical_occurrence_domain"]=q->identity().support_reference;
                    qc["applicability_rule"]="applicable exactly on q retained support; outside is inapplicable and has no coefficient";
                    qc["eligibility_rule"]="eligible exactly when applicable and all immutable UQ restrictions are T";
                    for(int uq=0;uq<7;++uq)qc["restrictions"]["UQ-0"+std::to_string(uq+1)]=shared_qc.restrictions[uq]==UniformFact::true_value?"T":"nonpassing";
                    for(const auto *fact:{"scalar","unit","integrity","member"})qc["payload_facts"][fact]="T";
                    qc["integrity_evidence"]="required payload write closed, digest bound, decoder reloaded, scalar/unit/support independently compared";
                    qc["lifecycle_evidence"]="new write-once q and g links; exact immutable input generation; source nonmutation checked before final product publication";
                    write_yaml(directory/"handoff"/(name+"-coefficient-qc.yaml"),qc);
                    Eigen::VectorXd times(q->identity().slots.size());for(Eigen::Index t=0;t<times.size();++t)times[t]=n.native_times[n.first+q->identity().slots[t]*n.factor];
                    const auto schedule=evaluate_ast_rtc_schedule(*n.observation.trajectory,*n.observation.offsets,ra0,dec0,times);
                    const auto cols=n.ptc["segments"][si]["output_detector_indices"].as<std::vector<std::size_t>>();
                    PtcMatrix coordinate(times.size(),2*cols.size());
                    for(std::size_t d=0;d<cols.size();++d) {
                        const auto [x,y]=n.geometry[cols[d]];const auto xy=schedule.detector_tangent_deg(x,y);
                        coordinate.col(2*d)=xy.col(0);coordinate.col(2*d+1)=xy.col(1);
                    }
                    const auto admission=ValMapAdmission::bind(q,g,{q->identity().publication,q->identity().cal_parent,q->identity().rtc_grid,q->identity().detectors,q->identity().slots});
                    std::vector<double> compact;compact.reserve(2*q->retained());
                    std::size_t local_admitted=0,occurrence=0;
                    std::vector<std::uint8_t> decisions((q->retained()+7)/8,0);
                    for(std::size_t t=0;t<q->identity().slots.size();++t)for(std::size_t d=0;d<cols.size();++d)if(q->member(t,d)) {
                        const auto decision=admission.at(t,d,n.cal_causes[cols[d]][q->identity().slots[t]]);
                        const double x=coordinate(t,2*d),y=coordinate(t,2*d+1);
                        // PTC support remains unchanged. MAP's local decision and
                        // AST invalidity are retained in the placement domain.
                        compact.push_back(x);compact.push_back(y);
                        if(decision.pass())decisions[occurrence/8]|=std::uint8_t(1u<<(occurrence%8));
                        ++occurrence;
                        if(decision.pass()){++admitted;++local_admitted;if(std::isfinite(x)&&std::isfinite(y)){
                            extent_x=std::max(extent_x,std::abs(x));extent_y=std::max(extent_y,std::abs(y));
                        }else ++invalid_coordinate;}
                    }
                    const auto coords=directory/"handoff"/(name+"-ast.f64");map_write_array<double>(coords,compact);
                    YAML::Node binding;binding["schema"]="citlali-ast-saved-rtc-map-coordinates-v1";
                    binding["PTC_publication"]=q->identity().publication;binding["CAL_parent"]=q->identity().cal_parent;
                    binding["RTC_grid"]=q->identity().rtc_grid;binding["AST_telescope"]=n.cal["AST_telescope"].as<std::string>();
                    binding["AST_pointing_offset"]=n.observation.offset_identity;binding["owner_frame_binding"]="sha256:"+owner_sha;
                    binding["coordinates_file"]=coords.filename().string();binding["sha256"]=citlali::utils::sha256_file(coords);
                    binding["layout"]="little-endian f64 pairs tangent_lon_deg,tangent_lat_deg in exact PTC retained-support order";
                    binding["MAP_admitted"]=local_admitted;binding["MAP_profile"]=std::string(mapmaking::ForwardMapAdmission::profile);
                    binding["MAP_profile_binding"]=std::string(mapmaking::ForwardMapAdmission::binding);
                    binding["coefficient_QC"]=std::string(g->profile);
                    binding["coefficient_QC_evaluation_sha256"]=citlali::utils::sha256_file(directory/"handoff"/(name+"-coefficient-qc.yaml"));
                    binding["MAP_decision_request"]="requested";binding["MAP_decision_applicability"]="applicable: ordinary positive-rank PTC route";
                    binding["MAP_decision_realization"]="realized";
                    binding["MAP_eligibility_rule"]="PTC retained support AND zero inherited CAL cause AND passing exact uniform QC; independent of pixel placement";
                    binding["MAP_ineligible_count"]=q->retained()-local_admitted;binding["coefficient_payload_sha256"]=citlali::utils::sha256_file(coef);
                    binding["decision_representation"]="shared immutable uniform facts; occurrence membership and MAP CAL/PTC causes resolved from exact parents; requested/applicable/eligible/realized required";
                    binding["unavailable_coordinates"]="NaN denotes AST coordinate unavailability; MAP admission is evaluated separately; never sky zero";
                    const auto ast_file=directory/"handoff"/(name+"-ast.yaml");write_yaml(ast_file,binding);
                    const auto bits=directory/"handoff"/(name+"-map-eligible.bits");map_write_array<std::uint8_t>(bits,decisions);
                    YAML::Node evaluation;evaluation["schema"]="citlali-val-map-occurrence-evaluation-v1";
                    evaluation["registry"]=std::string(ValMapAdmission::registry);evaluation["registry_sha256"]=std::string(ValMapAdmission::registry_sha256);
                    evaluation["source_register"]=std::string(ValMapAdmission::source_register);evaluation["source_register_sha256"]=std::string(ValMapAdmission::source_register_sha256);
                    evaluation["profile"]=std::string(mapmaking::ForwardMapAdmission::profile);evaluation["binding"]=std::string(mapmaking::ForwardMapAdmission::binding);
                    evaluation["role"]="SCI-MAP:base_observation_map@1";evaluation["object_domain"]=q->identity().support_reference;
                    evaluation["object_rule"]="each exact (publication, detector occurrence, RTC slot) in retained row-major support; atomic_only";
                    evaluation["publication_sha256"]=citlali::utils::sha256_file(pub);evaluation["coefficient_sha256"]=citlali::utils::sha256_file(coef);
                    evaluation["AST_binding_sha256"]=citlali::utils::sha256_file(ast_file);
                    evaluation["coefficient_QC_sha256"]=binding["coefficient_QC_evaluation_sha256"].as<std::string>();
                    evaluation["eligibility_file"]=bits.filename().string();evaluation["eligibility_sha256"]=citlali::utils::sha256_file(bits);
                    evaluation["requested"]="requested";evaluation["applicability"]="applicable";evaluation["realization"]="realized";
                    evaluation["eligibility_encoding"]="little-bit-order packed bits in exact retained support: 1 eligible, 0 ineligible; trailing padding zero";
                    evaluation["eligible"]=local_admitted;evaluation["ineligible"]=q->retained()-local_admitted;
                    evaluation["causes"]="exact source CAL u16 and PTC u8 causes retained through publication parent, detector columns and RTC slots; uniform QC restrictions retained separately";
                    evaluation["response_covariance_and_engineering_classification"]="advisory; no veto or science-qualification promotion";
                    evaluation["identity_convention"]="sha256 of complete immutable evaluation file; exact identity recorded in frozen MAP plan";
                    evaluation["allocation_identity"]="SCI-MAP-PROFILE-IDENTITY-ALLOCATION/2026-09-08";
                    evaluation["allocation_sha256"]="2afbf10026784ac4528aa799b9f4362bf5f2bf328f77d3a3edb187c3221523f4";
                    evaluation["manifest_identity"]="SCI-MAP-PROFILE-ADMISSION-MANIFEST/2026-09-08";
                    evaluation["manifest_sha256"]="57229670cd540ce2f8aeecf5d1a568414d0a85b6a9d6cf5a74b7e698b6ac5ec3";
                    evaluation["shared_core_identity"]="SCI-MAP-v0.2-SHARED-AUTHORITY/r0.4";
                    evaluation["shared_core_sha256"]="3b81ea8180dcc3f5f37964ac1177feb417fef6a1cdcc590b33f9431b54bc1e31";
                    evaluation["PTC_MAP_boundary_identity"]="SCI-PTC_TO_SCI-MAP_BOUNDARY v0.2-draft.1/r0.4";
                    evaluation["PTC_MAP_boundary_sha256"]="58291e1c879dc9116e81a9e3fd4323caf79a176c51e296d8119efb6937272ef8";
                    evaluation["AST_original_footprint_boundary_sha256"]="f7be703d17320de6f6ecfc3a3974d54799540fff1d8b2d9491c941a7ea3b45a4";
                    evaluation["MAP_reached_stages"]=std::vector<std::string>{"requested","effective"};
                    evaluation["evaluator_build"]=std::string(CITLALI_GIT_VERSION);
                    evaluation["fixtures"]=std::vector<std::string>{"tests/test_forward_map.cpp:PtcMapHandoff/ForwardMapAdmission","tools/timestream_successor/test_forward_map_reload.py"};
                    evaluation["actual_evidence"]="bound q/g/QC/AST/CAL files and exact occurrence eligibility bits; native second-pass decision replay verifies artifact content before accumulation";
                    const auto evaluation_file=directory/"handoff"/(name+"-map-evaluation.yaml");write_yaml(evaluation_file,evaluation);
                    require(YAML::Dump(YAML::LoadFile(evaluation_file.string()))==YAML::Dump(evaluation) &&
                        map_read_array<std::uint8_t>(bits,decisions.size())==decisions,"VAL MAP artifact verification failed");
                    plans.push_back({ni,si,q->retained(),coords,pub,coef,q->identity().publication,q->identity().support_reference,binding["sha256"].as<std::string>()});
                }
            }
            require(retained==expected_retained,"MAP aggregate PTC support differs");
            const auto coordinates_done=std::chrono::steady_clock::now();
            const auto wcs=AstRtcTanGrid::centered(ra0,dec0,pixel,extent_x,extent_y);
            YAML::Node plan;plan["schema"]="citlali-forward-map-plan-v1";plan["role"]="SCI-MAP:base_observation_map@1";
            plan["observation"]=observation_identity;plan["array"]=arrays[array];plan["sources"]=sources;
            plan["request_identity"]="sha256:"+request_sha;plan["owner_binding"]="sha256:"+owner_sha;
            plan["coverage_cut"]=cut;plan["pixel_size_arcsec"]=pixel;plan["projection"]="SCI-MAP:one_hot_containing_pixel@1";
            plan["target_grid_policy"]="pre-accumulation centered odd grid enclosing all MAP-admitted valid AST coordinates with half-pixel margin";
            auto frame=plan["WCS"];frame["RADESYS"]="FK5";frame["EQUINOX"]=2000.;frame["CTYPE1"]="RA---TAN";frame["CTYPE2"]="DEC--TAN";
            frame["CUNIT1"]="deg";frame["CUNIT2"]="deg";frame["CRPIX1"]=wcs.crpix1;frame["CRPIX2"]=wcs.crpix2;
            frame["CRVAL1"]=wcs.crval1_deg;frame["CRVAL2"]=wcs.crval2_deg;frame["CDELT1"]=wcs.cdelt1_deg;frame["CDELT2"]=wcs.cdelt2_deg;
            frame["rows"]=wcs.rows;frame["columns"]=wcs.columns;frame["axis_order"]="x=RA,y=Dec;storage(y,x)";
            frame["source_to_target_frame"]="identity;owner-declared source FK5/J2000.0";
            frame["time_scale"]="UNIX UTC assigned RTC times; zero verified interface offset";
            frame["site_model_state"]="no new celestial frame, EOP or refraction transform; existing telescope/pointing-offset realization retained";
            std::size_t offset=0;
            for(const auto &s:plans){YAML::Node column;column["PTC_publication"]=s.identity;column["support_reference"]=s.support;
                column["first_input_row"]=offset;column["input_rows"]=s.retained;column["publication_file"]=fs::relative(s.publication,directory).string();
                column["publication_sha256"]=citlali::utils::sha256_file(s.publication);column["coefficient_file"]=fs::relative(s.coefficient,directory).string();
                column["coefficient_sha256"]=citlali::utils::sha256_file(s.coefficient);column["AST_coordinate_sha256"]=s.coordinate_sha;
                column["AST_coordinate_file"]=fs::relative(s.coordinates,directory).string();
                const auto ast_record=s.coordinates.parent_path()/(s.coordinates.stem().string()+".yaml");
                const auto qc_record=s.coefficient.parent_path()/(s.coefficient.stem().string()+"-qc.yaml");
                const auto stem=s.coordinates.stem().string();const auto prefix=stem.substr(0,stem.size()-4);
                const auto decision_record=s.coordinates.parent_path()/(prefix+"-map-evaluation.yaml");
                const auto decision_bits=s.coordinates.parent_path()/(prefix+"-map-eligible.bits");
                for(const auto &[key,file]:std::array<std::pair<std::string,fs::path>,4>{{{"AST_binding",ast_record},{"coefficient_QC",qc_record},{"MAP_evaluation",decision_record},{"MAP_eligibility",decision_bits}}}) {
                    column[key+"_file"]=fs::relative(file,directory).string();column[key+"_sha256"]=citlali::utils::sha256_file(file);
                }
                column["MAP_evaluation_identity"]="sha256:"+column["MAP_evaluation_sha256"].as<std::string>();
                plan["ordered_column_domain"].push_back(column);offset+=s.retained;}
            plan["reached_stages"]=std::vector<std::string>{"requested","effective","observation_resolved"};
            write_yaml(directory/"plan.yaml",plan);const auto plan_sha=citlali::utils::sha256_file(directory/"plan.yaml");
            mapmaking::ForwardMapGrid grid{wcs.rows,wcs.columns,"sha256:"+plan_sha+":WCS"};
            const std::string domain="sha256:"+plan_sha+":ordered-PTC-support";
            std::vector<double> values;values.reserve(retained);std::vector<mapmaking::ForwardMapMember> members;members.reserve(admitted-invalid_coordinate);
            for(const auto &s:plans) {
                const auto q=load_map_segment(networks[s.network_index],s.segment,ledger);
                require(q->identity().publication==s.identity && q->identity().support_reference==s.support,"MAP planned PTC domain changed");
                require(citlali::utils::sha256_file(s.coordinates)==s.coordinate_sha,"MAP planned AST payload changed");
                const auto &n=networks[s.network_index];
                const auto column_index=static_cast<std::size_t>(&s-plans.data());
                require(citlali::utils::sha256_file(s.coefficient)==plan["ordered_column_domain"][column_index]["coefficient_sha256"].as<std::string>(),"PTC coefficient payload integrity failed");
                const auto column=plan["ordered_column_domain"][column_index];
                for(const auto *key:{"publication","coefficient","AST_coordinate","AST_binding","coefficient_QC","MAP_evaluation","MAP_eligibility"})
                    require(citlali::utils::sha256_file(directory/column[std::string(key)+"_file"].as<std::string>())==
                            column[std::string(key)+"_sha256"].as<std::string>(),"MAP planned required handoff changed");
                const auto decision_bits=map_read_array<std::uint8_t>(directory/column["MAP_eligibility_file"].as<std::string>(),(s.retained+7)/8);
                const auto payload=YAML::LoadFile(s.coefficient.string());
                require(payload["scalar"].as<double>()==1 && payload["unit"].as<std::string>()=="1" &&
                    payload["family"].as<std::string>()==std::string(PtcUniformCoefficients::family),"PTC coefficient payload fidelity failed");
                const auto g=PtcUniformCoefficients::publish(q,{payload["selection"].as<std::string>(),payload["family"].as<std::string>()},
                    payload["identity"].as<std::string>(),payload["stored_support_reference"].as<std::string>());
                const auto admission=ValMapAdmission::bind(q,g,{q->identity().publication,q->identity().cal_parent,q->identity().rtc_grid,q->identity().detectors,q->identity().slots});
                const auto cols=n.ptc["segments"][s.segment]["output_detector_indices"].as<std::vector<std::size_t>>();
                const auto xy=map_read_array<double>(s.coordinates,2*s.retained);std::size_t k=0;
                for(std::size_t t=0;t<q->identity().slots.size();++t)for(std::size_t d=0;d<q->identity().detectors.size();++d)if(q->member(t,d)) {
                    const auto row=values.size();values.push_back(q->values()(t,d));
                    const auto [x,y]=wcs.continuous_pixel(xy[2*k],xy[2*k+1]);++k;
                    const auto decision=admission.at(t,d,n.cal_causes[cols[d]][q->identity().slots[t]]);
                    require(decision.pass()==bool(decision_bits[(k-1)/8] & (1u<<((k-1)%8))),"VAL MAP decision replay differs");
                    if(!decision.pass())continue;
                    const auto gamma=g->value(decision.coefficient_qc,t,d);
                    require(gamma && std::isfinite(*gamma) && *gamma>0,"MAP coefficient numerical classification failed");
                    if(const auto p=mapmaking::containing_pixel(grid,x,y))members.push_back({row,*p,*gamma});
                }
                require(k==s.retained,"MAP planned support changed");
            }
            auto op=mapmaking::ForwardMapOperator::resolve(grid,domain,values.size(),std::move(members),{cut,{}});
            const auto resolved=std::chrono::steady_clock::now();
            attempt["array_lifecycle"][arrays[array]]=std::vector<std::string>{"requested","effective","observation_resolved","applied"};
            const auto result=op.apply(domain,values);const auto accumulated=std::chrono::steady_clock::now();
            if(op.output_rows().empty()) {YAML::Node no;no["plan_sha256"]=plan_sha;no["outcome"]="not_produced";
                no["cause"]="no_support_authorized_output_rows";write_yaml(directory/"attempt.yaml",no);continue;}
            std::vector<std::int32_t> pixels(values.size(),-1);for(const auto &m:op.members())pixels[m.input_row]=static_cast<std::int32_t>(m.pixel);
            map_write_array<std::int32_t>(directory/"operator-pixel.i32",pixels);
            map_write_array<std::size_t>(directory/"output-rows.u64",op.output_rows());
            map_fits(directory/"map.fits",wcs,op,result,arrays[array],beams[array]);
            YAML::Node bundle;bundle["schema"]="citlali-forward-map-bundle-v1";bundle["role"]="SCI-MAP:base_observation_map@1";
            bundle["product_identity"]="SCI-MAP-base:"+plan_sha+":"+citlali::utils::sha256_file(directory/"map.fits");
            bundle["base_parent"]=bundle["product_identity"].as<std::string>();bundle["plan_sha256"]=plan_sha;bundle["input_domain"]=domain;bundle["grid_identity"]=grid.identity;
            bundle["MAP_authority"]="SCI-MAP-v0.2-r0.4";bundle["MAP_shared_source_sha256"]="3b81ea8180dcc3f5f37964ac1177feb417fef6a1cdcc590b33f9431b54bc1e31";
            bundle["reached_stages"]=std::vector<std::string>{"requested","effective","observation_resolved","applied","realized"};
            bundle["outcome"]="success";bundle["WCS"]=frame;bundle["observation"]=observation_identity;bundle["array"]=arrays[array];bundle["sources"]=sources;
            bundle["signal_unit"]="mJy/beam";bundle["quantity"]="nonpolarimetric total-intensity-equivalent calibrated x; no Stokes-I claim";
            bundle["nominal_beam_parent"]="sha256:"+owner_sha+":"+arrays[array];bundle["nominal_beam_fwhm_arcsec"]=beams[array];
            bundle["measured_beam_or_full_response_inferred"]=false;bundle["coefficient_family"]=std::string(PtcUniformCoefficients::family);
            bundle["normalization_interpretation"]="sum of dimensionless gridding coefficients; not precision or significance";
            bundle["PTC_additive_reference"]="per-detector eligible arithmetic means removed and not restored; source segment means/bases retained by exact parent links";
            bundle["upstream_conditioned_response"]="unavailable: complete RTC/CAL/PTC astronomical response not supplied";
            bundle["full_procedure_response"]="unavailable: not measured";bundle["covariance"]="unavailable: not zero and not independent";
            bundle["MAP_local_operator"]="exact sparse one-hot operator: input columns from plan; pixel.i32=-1 means no contribution; coefficient=1; output rows from output-rows.u64; divide by NORMALIZATION";
            bundle["upstream_eligible_original_footprint_exposure"]="unavailable: original ALIGN-grid coordinate/exposure carrier not published";
            bundle["retained_original_footprint_exposure"]="unavailable: original ALIGN-grid coordinate/exposure carrier not published";
            bundle["NOI"]="not requested; unavailable";bundle["FLT"]="not requested; unavailable";bundle["FRUIT"]="deferred; not executed";
            bundle["PTC_retained"]=retained;bundle["MAP_route_admitted"]=admitted;bundle["AST_invalid_coordinates"]=invalid_coordinate;
            bundle["contributing_occurrences"]=op.members().size();bundle["scientific_rows"]=op.output_rows().size();bundle["coverage_cut"]=cut;
            bundle["reference_normalization"]=op.reference_normalization();if(op.reference_index())bundle["reference_index"]=*op.reference_index();
            bundle["pixels_outside_science_support"]="unavailable (FITS NaN), distinct from measured zero";
            for(const auto *name:{"plan.yaml","map.fits","operator-pixel.i32","output-rows.u64"})bundle["files"][name]=citlali::utils::sha256_file(directory/name);
            bundle["AST_and_handoff_seconds"]=std::chrono::duration<double>(coordinates_done-array_started).count();
            bundle["projection_and_resolution_seconds"]=std::chrono::duration<double>(resolved-coordinates_done).count();
            bundle["accumulation_seconds"]=std::chrono::duration<double>(accumulated-resolved).count();
            bundle["serialization_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-accumulated).count();
            write_yaml(directory/"bundle.yaml",bundle);++products;
            attempt["array_lifecycle"][arrays[array]]=bundle["reached_stages"];
            attempt["products"][arrays[array]]=citlali::utils::sha256_file(directory/"bundle.yaml");
            std::cout<<"MAP "<<arrays[array]<<": "<<op.members().size()<<" contributions, "<<op.output_rows().size()<<" science rows\n";
        }
        ledger.verify();YAML::Node manifest;for(const auto &[p,h]:ledger.files)manifest[p]=h;write_yaml(staging/"input-manifest.yaml",manifest);
        attempt["input_manifest_sha256"]=citlali::utils::sha256_file(staging/"input-manifest.yaml");
        attempt["outcome"]=products?"success":"not_produced";
        attempt["lifecycle_scope"]="root request/effective only; independent five-stage lifecycle per array";
        if(!products)attempt["cause"]="no_support_authorized_output_rows_in_requested_arrays";
        attempt["wall_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        attempt["cpu_seconds"]=double(std::clock()-cpu)/CLOCKS_PER_SEC;struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
#ifdef __APPLE__
        attempt["process_peak_rss_bytes"]=usage.ru_maxrss;
#else
        attempt["process_peak_rss_bytes"]=usage.ru_maxrss*1024;
#endif
        write_yaml(staging/"attempt.yaml",attempt);fs::rename(staging,output);return 0;
    } catch(const std::exception &e) {
        attempt["outcome"]="failed";attempt["cause"]=e.what();attempt["no_complete_output_directory"]=true;
        write_yaml(staging/"attempt.yaml",attempt);throw;
    }
}
} // namespace
