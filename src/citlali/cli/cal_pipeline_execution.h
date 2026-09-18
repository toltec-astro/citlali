// Private application adapter. YAML/NetCDF/APT parsing stays outside CAL/AST.
#include <citlali/core/pipeline/timestream_cal_pipeline.h>
#include "cal_opacity_header.h"

namespace {
YAML::Node execute_connected_cal(const CalRtcSource &source,const YAML::Node &cfg,
    const apt::VerifiedBundle &verified,const pipeline::CanonicalAptDetectorRelationV2 &relation,
    const std::vector<int> &channels,const fs::path &output) {
    const auto started=std::chrono::steady_clock::now();
    const auto grid=source.rtc_terminal_handle()->grid_handle();const auto scope=grid->align_handle()->scope();
    const auto telescope_path=checked_file(cfg["telescope"]);
    const std::string telescope_identity="sha256:"+citlali::utils::sha256_file(telescope_path);
    const std::string apt_identity="sha256:"+citlali::utils::sha256_file(checked_file(cfg["manifest"]));
    require(scope.observation==relation.observation().observation &&
            scope.subobservation==relation.observation().subobservation && scope.scan==relation.observation().scan,
            "CAL selected APT belongs to another observation");
    for(const auto &[name,unit]:std::array<std::pair<std::string,std::string>,3>{{
        {"flxscale","mJy/beam/xs"},{"x_t","arcsec"},{"y_t","arcsec"}}}) {
        const auto it=std::find_if(verified.fields.begin(),verified.fields.end(),[&](const auto &f){return f.name==name;});
        require(it!=verified.fields.end() && it->unit==unit,"CAL/AST selected APT field convention differs: "+name);
    }
    std::vector<CalDetectorFactor> factors;std::vector<AstRtcDetectorGeometry> geometry;
    for(std::size_t d=0;d<grid->detectors().size();++d) {
        const auto &g=grid->detectors()[d];const auto channel=channels.at(d);
        const auto &binding=grid->align_handle()->paired_handle()->network(g.network).detector(g.detector);
        const auto &mapping=grid->align_handle()->paired_handle()->network(g.network).mapping_authority();
        require(mapping.x.meaning_id=="kidscpp:gainlintrend:x:meaning" && mapping.x.unit_or_scale_id=="dimensionless" &&
            mapping.producer_interface_id==native_paired_xr_producer_interface_id,
            "CAL ordinary-xs admission does not support this producer's measured-channel convention");
        const auto edge=std::find_if(relation.bindings().begin(),relation.bindings().end(),[&](const auto &b){return b.network==g.network && b.channel==channel;});
        require(edge!=relation.bindings().end(),"CAL target-to-source relation absent");
        const auto row=std::find_if(verified.apt.rows.begin(),verified.apt.rows.end(),[&](const auto &r){return r.uid==edge->output_uid;});
        require(row!=verified.apt.rows.end() && row->network==g.network && row->channel==channel && row->array==edge->array,
                "CAL selected child row identity differs from RTC acquisition");
        const auto row_identity=apt_identity+":row="+std::to_string(row->uid);
        const double f=field(*row,"flxscale");
        factors.push_back({binding,row_identity,static_cast<int>(row->array),edge->disposition==apt::RelationDisposition::matched,
                           std::isfinite(f)?std::optional<double>{f}:std::nullopt});
        geometry.push_back({binding,row_identity,field(*row,"x_t"),field(*row,"y_t"),static_cast<int>(row->array)});
    }
    netCDF::NcFile tel(telescope_path.string(),netCDF::NcFile::read);
    require(read_netcdf_scalar<int>(tel,"Header.Dcs.ObsNum")==scope.observation &&
        read_netcdf_scalar<int>(tel,"Header.Dcs.SubObsNum")==scope.subobservation &&
        read_netcdf_scalar<int>(tel,"Header.Dcs.ScanNum")==scope.scan,
        "AST/CAL telescope observation mismatch");
    require(read_netcdf_scalar<double>(tel,"Header.Source.Epoch")==2000 &&
        read_netcdf_scalar<int>(tel,"Header.Source.CoordSys")==0,
        "AST bounded V2 adapter requires declared J2000 radec source frame");
    auto series=[&](const std::string &name,const std::string &unit) {
        const auto var=tel.getVar(name);require(!var.isNull() && var.getDimCount()==1,"AST telescope series absent: "+name);
        std::string actual;const auto attr=var.getAtt("units");require(!attr.isNull(),"AST telescope series unit absent: "+name);attr.getValues(actual);
        require(actual==unit,"AST telescope series unit differs: "+name);
        Eigen::VectorXd values(var.getDim(0).getSize());var.getVar(values.data());return values;
    };
    NativeTelescopeData data;
    for(const auto &[source_name,key]:std::array<std::pair<std::string,std::string>,7>{{
        {"TelTime","TelTime"},{"SourceRaAct","TelRa"},{"SourceDecAct","TelDec"},
        {"TelAzAct","TelAzAct"},{"TelElAct","TelElAct"},{"TelElCor","TelElCor"},{"ActParAng","ActParAng"}}})
        data[key]=series("Data.TelescopeBackend."+source_name,key=="TelTime"?"sec":"rad");
    auto center=[&](const std::string &name) {auto v=series(name,"rad");require(v.size()>0 && v.array().isFinite().all() &&
        (v.array()==v[0]).all(),"AST bounded V2 adapter requires a fixed source center");return v[0];};
    const double ra0=center("Header.Source.Ra"),dec0=center("Header.Source.Dec");
    const auto effective_path=checked_file(cfg["effective_config"]);const auto effective=YAML::LoadFile(effective_path.string());
    YAML::Node selected;
    const std::string input_name=std::to_string(scope.observation)+"_"+std::to_string(scope.subobservation)+"_"+std::to_string(scope.scan);
    unsigned matches=0;
    for(const auto &input:effective["inputs"])if(input["meta"]["name"].as<std::string>()==input_name) {
        for(const auto &cal:input["cal_items"])if(cal["type"].as<std::string>()=="astrometry") {selected=YAML::Clone(cal);++matches;}
    }
    require(matches==1,"AST pointing requires one exact observation's astrometry item");
    NativePointingOffsetsArcsec offset_values;Eigen::VectorXd offset_times;unsigned time_records=0;
    for(const auto &entry:selected["pointing_offsets"]) {
        if(entry["modified_julian_date"]) {
            const auto v=entry["modified_julian_date"].as<std::vector<double>>();require(v.size()==2,"AST offset time support must have two endpoints");
            offset_times.resize(2);for(int i=0;i<2;++i)offset_times[i]=(v[i]-40587.)*86400.;++time_records;
        }else {
            const auto axis=entry["axes_name"].as<std::string>();const auto values=entry["value_arcsec"].as<std::vector<double>>();
            require((axis=="az" || axis=="alt") && !offset_values.contains(axis),"AST offset axis duplicate or unknown");
            offset_values[axis]=Eigen::Map<const Eigen::VectorXd>(values.data(),values.size());
        }
    }
    require(time_records==1,"AST pointing offset time support absent or duplicated");
    auto offsets=std::make_shared<const NativePointingOffsetModel>(std::move(offset_values),std::move(offset_times));
    auto ast=AstRtcCoordinates::realize_v2(grid,scope,telescope_identity,
        std::make_shared<const RawTelescopeTrajectory>(std::move(data)),ra0,dec0,
        "sha256:"+citlali::utils::sha256_file(effective_path)+":inputs:"+input_name+":astrometry",offsets,std::move(geometry));
    // Owner 2026-09-18: one observation-associated reading is constant over
    // that observation. Preserve the header's actual update text; do not
    // manufacture a sampled series or assert an unverified time conversion.
    const auto opacity_header=citlali::cli::detail::read_cal_opacity_header(tel);
    double observation_first=std::numeric_limits<double>::infinity(),observation_last=-observation_first;
    for(auto network:grid->align_handle()->participant_network_ids()) {
        const auto &axis=grid->align_handle()->paired_handle()->network(network).occurrence_axis();
        observation_first=std::min(observation_first,grid->align_handle()->occurrence_assignment(network,axis.first_native_row()).assigned_time_unix_sec);
        observation_last=std::max(observation_last,grid->align_handle()->occurrence_assignment(network,axis.past_last_native_row()-1).assigned_time_unix_sec);
    }
    std::vector<CalWvrRecord> opacity_records;
    if(opacity_header.tau225)
        opacity_records.push_back({telescope_identity+":Header.Radiometer.Tau",std::nullopt,*opacity_header.tau225,true,NAN,NAN});
    auto wvr=CalWvrEvidence::learn(scope,telescope_identity,
        "ALIGN:exact-RTC-occurrence-integration-midpoint:Unix-seconds",std::move(opacity_records),
        CalWvrObservationInterval{observation_first,observation_last});
    auto evidence=CalEvidence::learn(source,ast,wvr,CalAtmosphereSurface::frozen(),apt_identity,std::move(factors));
    const auto learned=std::chrono::steady_clock::now();
    auto plan=CalPlan::consider(evidence,source.val_snapshot_handle(),1);
    const auto considered=std::chrono::steady_clock::now();
    auto signal=CalAppliedSignal::apply(plan,source,source.val_snapshot_handle());
    auto val=ValSnapshot::commit_cal_output(source.val_snapshot_handle(),ValCalOutputFacts::preserve(signal));
    const auto applied=std::chrono::steady_clock::now();
    const auto destination=output/"cal";fs::create_directory(destination);
    YAML::Node record;record["schema"]="citlali-cal-output-v1";
    record["state"]=signal->available_count()?"calibrated-development-signal":"no-calibrated-output";
    record["available"]=signal->available_count();record["scheduled"]=source.rtc_terminal_handle()->finalization().scheduled_slots;
    record["input_RTC_attempt"]=grid->applied_handle()->plan_handle()->attempt();record["CAL_plan"]=plan->instance();
    record["input_VAL_generation"]=source.val_snapshot_handle()->generation().value;record["output_VAL_generation"]=val->generation().value;
    record["unit"]=std::string(signal->unit);record["observable"]=std::string(signal->observable);
    record["reference_frequency_GHz"]=std::vector<int>{272,214,150};record["reference_alpha"]=0;record["X_ref"]=0;
    record["selected_child_APT"]=apt_identity;record["factor_application"]="selected-child-flxscale-once;no-parent-or-embodied-rescale-reapplication";
    record["application_counts_on_available_samples"]["selected_flxscale"]=1;
    record["application_counts_on_available_samples"]["target_atmosphere"]=1;
    record["application_counts_on_available_samples"]["target_unit_identity"]=1;
    record["application_counts_on_available_samples"]["parent_factors_or_embodied_rescales"]=0;
    record["atmosphere_operator"]=std::string(CalAtmosphereSurface::operator_id);
    record["atmosphere_contract_sha256"]=std::string(CalAtmosphereSurface::contract_sha256);
    record["atmosphere_nodes_sha256"]=std::string(CalAtmosphereSurface::nodes_sha256);
    record["passband_sha256"]=std::string(CalAtmosphereSurface::passband_sha256);
    record["WVR_source"]=wvr->source_identity();record["WVR_method"]=std::string(wvr->method_id());
    record["WVR_policy"]=std::string(wvr->policy);record["WVR_records"]=wvr->records().size();
    record["WVR_cause"]=std::string(cal_wvr_cause_name(wvr->at(observation_first).cause));
    record["WVR_single_reading_constant"]=wvr->single_reading();
    record["WVR_observation_first_unix_sec"]=observation_first;record["WVR_observation_last_unix_sec"]=observation_last;
    record["WVR_source_update_text"]=opacity_header.update_text;
    record["WVR_source_time_mapping"]=wvr->single_reading()?"not-required-for-singleton;raw-header-update-preserved":"unavailable-no-reading";
    record["WVR_variability_measured"]=false;
    record["WVR_input_limitation"]=wvr->single_reading()?
        "one observation-associated opacity reading; within-observation variability is unmeasured; atmosphere correction still uses each sample's elevation":
        "no opacity reading supplied; CAL remains unavailable";
    if(wvr->single_reading()) {
        record["WVR_record_identity"]=wvr->records().front().identity;
        record["WVR_constant_tau225"]=wvr->records().front().tau225;
        record["WVR_admission"]="owner-approved-observation-header;no-producer-invalid-flag-supplied;numeric-validity-checked-by-CAL";
    }
    record["opacity_quality_method"]=std::string(wvr->quality_method);
    record["opacity_quality"]=std::string(cal_opacity_quality_name(evidence->opacity_quality().classification));
    record["quality_window_first_unix_sec"]=evidence->opacity_quality().first;record["quality_window_last_unix_sec"]=evidence->opacity_quality().last;
    record["quality_cause"]=evidence->opacity_quality().cause;
    record["quality_summary_available"]=evidence->opacity_quality().summary_available;
    if(evidence->opacity_quality().summary_available) {
        record["quality_mean_tau225"]=evidence->opacity_quality().mean;
        record["quality_min_tau225"]=evidence->opacity_quality().minimum;
        record["quality_max_tau225"]=evidence->opacity_quality().maximum;
    }
    record["AST_role"]=std::string(ast->role);record["AST_method"]=std::string(ast->method);
    record["AST_telescope"]=ast->telescope_identity();record["AST_pointing_offset"]=ast->offset_identity();
    record["AST_coordinate_unit"]="degree";record["AST_frame"]="J2000-radec-gnomonic-tangent";
    record["AST_center_ra_rad"]=ra0;record["AST_center_dec_rad"]=dec0;
    record["AST_angle_filtering_applied"]=false;record["r_calibrated"]=false;
    record["conditional_and_total_uncertainty"]=std::string(signal->uncertainty);
    record["literal_peak_response_qualified"]=false;record["science_qualified"]=false;
    record["PTC_executed"]=false;record["MAP_executed"]=false;
    record["numerical_layout"]="supported slots only: slot.i64 and value.f64; causes.u16 covers every original RTC slot, little endian";
    record["cause_bits"]["rtc_unavailable"]=1;record["cause_bits"]["direct_replacement_or_exclusion"]=2;
    record["cause_bits"]["invalid_factor"]=4;record["cause_bits"]["outside_supported_calibration"]=8;
    record["cause_bits"]["invalid_atmosphere"]=16;record["cause_bits"]["pointing_unavailable"]=32;record["cause_bits"]["numeric_failure"]=64;
    for(std::size_t d=0;d<grid->detectors().size();++d) {
        const auto stem="channel-"+std::to_string(channels[d]);
        std::ofstream values(destination/(stem+"-value.f64"),std::ios::binary),slots(destination/(stem+"-slot.i64"),std::ios::binary),causes(destination/(stem+"-causes.u16"),std::ios::binary);
        YAML::Node detector;detector["channel"]=channels[d];detector["selected_row"]=evidence->factors()[d].selected_row_identity;
        detector["array"]=evidence->factors()[d].array;detector["uniquely_matched"]=evidence->factors()[d].uniquely_matched;
        if(evidence->factors()[d].flxscale_mJy_beam_per_x)detector["selected_flxscale"]=*evidence->factors()[d].flxscale_mJy_beam_per_x;
        std::map<std::uint16_t,std::size_t> counts;std::size_t available=0;
        for(std::size_t s=0;s<grid->detectors()[d].scheduled_count;++s) {
            const auto cause=val->committed_cal_output_facts_handle()->at(signal,d,s);++counts[cause];
            causes.write(reinterpret_cast<const char*>(&cause),sizeof(cause));
            if(const auto value=signal->value(d,s)){const auto slot=static_cast<std::int64_t>(s);slots.write(reinterpret_cast<const char*>(&slot),8);values.write(reinterpret_cast<const char*>(&*value),8);++available;}
        }
        values.close();slots.close();causes.close();require(bool(values)&&bool(slots)&&bool(causes),"required CAL output failed");
        detector["available"]=available;for(const auto &[cause,count]:counts)detector["joint_cause_counts"][std::to_string(cause)]=count;
        for(const auto *suffix:{"-value.f64","-slot.i64","-causes.u16"})detector["files"][stem+suffix]=citlali::utils::sha256_file(destination/(stem+suffix));
        record["detectors"].push_back(detector);
    }
    record["Learn_seconds"]=std::chrono::duration<double>(learned-started).count();
    record["Consider_seconds"]=std::chrono::duration<double>(considered-learned).count();
    record["Apply_VAL_seconds"]=std::chrono::duration<double>(applied-considered).count();
    write_yaml(destination/"receipt.yaml",record);return record;
}
} // namespace
