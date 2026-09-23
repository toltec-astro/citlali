// Private successor boundary: exact saved-input parsing and publication.
// Model arithmetic/lifecycle, PTC, AST and MAP remain in their typed owners.
#include <citlali/core/pipeline/fruit_cycle.h>
namespace {
int execute_saved_fruit(const fs::path &input,const fs::path &output) {
    const auto began=std::chrono::steady_clock::now();const auto cpu=std::clock();
    auto elapsed=[](auto t){return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();};
    require(output.is_absolute() && !fs::exists(output),"FRUIT requires a fresh absolute output");
    MapFileLedger ledger;const auto request_sha=ledger.bind(input);const auto request=YAML::LoadFile(input.string());
    const std::set<std::string> fields{"schema","method","mode","array","map_request","parent_plan","candidate","owner_binding"};
    for(const auto &e:request)require(fields.contains(e.first.as<std::string>()),"FRUIT unknown request field");
    require(request["schema"].as<std::string>()=="citlali-point-fruit-cycle-v1" && request["method"].as<std::string>()=="point-152389-direct-one-cycle-v1","FRUIT unsupported method");
    const auto mode=request["mode"].as<std::string>();require(mode=="zero" || mode=="reference","FRUIT only zero control or first reference cycle");
    const auto owner_path=checked_file(request["owner_binding"]);const auto owner_sha=ledger.bind(owner_path);
    const auto owner=YAML::LoadFile(owner_path.string());
    require(owner["identity"].as<std::string>()=="POINT-152389-FIRST-FRUIT-OWNER-2026-09-23" &&
        owner["outside_disk_no_feedback"].as<bool>() && owner["direct_identical_operand_rejoin"].as<bool>() &&
        owner["gain"].as<double>()==1 && owner["nonzero_cycles"].as<int>()==1,"FRUIT owner choices absent");
    const auto map_path=checked_file(request["map_request"]);ledger.bind(map_path);const auto mr=YAML::LoadFile(map_path.string());
    const auto pp=checked_file(request["parent_plan"]);ledger.bind(pp);const auto parent=YAML::LoadFile(pp.string());
    const auto array=request["array"].as<std::string>();const std::array<std::string,3> arrays{"a1100","a1400","a2000"};
    const auto ai=std::find(arrays.begin(),arrays.end(),array)-arrays.begin();require(ai<3 && parent["array"].as<std::string>()==array && parent["observation"].as<std::string>()=="obs:152389_0_2","FRUIT foreign array/observation");
    require(mr["schema"].as<std::string>()=="citlali-map-saved-ptc-v1" && mr["coverage_cut"].as<double>()==parent["coverage_cut"].as<double>() && mr["pixel_size_arcsec"].as<double>()==2,"FRUIT MAP policy differs");
    const auto mo=checked_file(mr["owner_binding"]);ledger.bind(mo);require(parent["owner_binding"].as<std::string>()=="sha256:"+citlali::utils::sha256_file(mo),"FRUIT MAP owner differs");
    const auto candidate=checked_file(request["candidate"]);const auto candidate_sha=ledger.bind(candidate);
    const std::array<std::string,3> selected_hashes{
        "8edea26bb5c5b5aca065a88c93a0a30fe5064f3cc3388f9423725ed6a63bc640",
        "a6e5a34d6251d3bb7b11b0bc4217fdd5a221e85d48fe08ec2fa5f4b82018d9ff",
        "aafe2afb4965bcd98bac1c1adf9a9dd889b606077af43b1486b2462c8f15d9ac"};
    require(candidate_sha==selected_hashes[ai],"FRUIT requires the exact reviewed threshold-3 candidate");
    fitsfile *f=nullptr;int status=0,any=0;fits_open_file(&f,candidate.c_str(),READONLY,&status);
    long axes[2];int ndim=0;fits_get_img_dim(f,&ndim,&status);fits_get_img_size(f,2,axes,&status);require(!status && ndim==2 && axes[0]==41 && axes[1]==41,"FRUIT candidate geometry");
    auto key=[&](const char *name){double v=NAN;fits_read_key(f,TDOUBLE,name,&v,nullptr,&status);return v;};
    const double cx=key("CRPIX1"),cy=key("CRPIX2"),dx=key("CDELT1"),dy=key("CDELT2"),ra=key("CRVAL1"),dec=key("CRVAL2"),equinox=key("EQUINOX");
    char frame[FLEN_VALUE],unit[FLEN_VALUE];fits_read_key(f,TSTRING,"RADESYS",frame,nullptr,&status);fits_read_key(f,TSTRING,"BUNIT",unit,nullptr,&status);
    auto same=[](double a,double b){return std::abs(a-b)<=1e-13*std::max(1.,std::abs(b));};
    require(!status && std::string(frame)=="FK5" && equinox==2000 && std::string(unit)=="mJy/nominal beam" && same(dx,parent["WCS"]["CDELT1"].as<double>()) && same(dy,parent["WCS"]["CDELT2"].as<double>()) &&
        same(ra,parent["WCS"]["CRVAL1"].as<double>()) && same(dec,parent["WCS"]["CRVAL2"].as<double>()),"FRUIT candidate frame/unit/scale mismatch");
    std::vector<double> brightness(axes[0]*axes[1]);std::vector<std::uint8_t> valid(brightness.size());
    fits_read_img(f,TDOUBLE,1,brightness.size(),nullptr,brightness.data(),&any,&status);
    fits_movnam_hdu(f,IMAGE_HDU,const_cast<char*>("VALID"),0,&status);fits_read_img(f,TBYTE,1,valid.size(),nullptr,valid.data(),&any,&status);fits_close_file(f,&status);require(!status,"FRUIT candidate read failed");
    FruitPointModel model("sha256:"+candidate_sha,{std::size_t(axes[1]),std::size_t(axes[0]),"candidate-grid"},cx,cy,dx,dy,std::move(brightness),std::move(valid));
    fs::create_directories(output);YAML::Node record;record["schema"]="citlali-point-fruit-cycle-output-v1";record["outcome"]="incomplete";
    record["request_sha256"]=request_sha;record["owner_sha256"]=owner_sha;record["source_revision"]=std::string(CITLALI_GIT_VERSION);record["mode"]=mode;record["array"]=array;
    record["candidate_identity"]=model.identity();record["selected_model"]=mode=="zero"?"zero integration control":model.identity();record["unit"]="mJy/nominal-beam";
    record["reference"]="source-relative model; fitted constant excluded; newly learned detector means not restored; rejoined model modes are model-supported, not recovered measured modes";
    record["response"]="empirical post-RTC/CAL direct containing-pixel convention; no complete upstream response inferred";
    record["r"]="unchanged diagnostic inert";record["outside_disk"]="no feedback; original data and eligibility retained; no zero-sky inference";
    record["inside_missing"]="unavailable projected model excludes affected input; no zero fill";
    write_yaml(output/"receipt.yaml",record);
    std::vector<double> totals,residuals,operands;struct Coord{std::size_t row;double lon,lat;};std::vector<Coord> coords;
    double extent_x=0,extent_y=0,ra0=NAN,dec0=NAN;std::set<int> seen;std::size_t expected_networks=parent["sources"].size();
    for(const auto &nr:mr["networks"]) {
        // Verify exact source receipt before selecting this array, then load its complete domain.
        ledger.bind(fs::path(nr["root"].as<std::string>())/"donor-continuity/ptc/receipt.yaml",nr["ptc_receipt_sha256"].as<std::string>());
        const auto cal=YAML::LoadFile((fs::path(nr["root"].as<std::string>())/"donor-continuity/cal/receipt.yaml").string());
        if(cal["detectors"][0]["array"].as<int>()!=int(ai))continue;
        auto n=load_saved_map_network(nr,ledger);require(n.obsnum==152389 && seen.insert(n.network).second,"FRUIT network selection");
        if(std::isfinite(ra0))require(ra0==n.observation.ra0 && dec0==n.observation.dec0,"FRUIT AST center changed");
        else {ra0=n.observation.ra0;dec0=n.observation.dec0;}
        bool bound=false;for(const auto &p:parent["sources"])if(p["network"].as<int>()==n.network && p["PTC_receipt_sha256"].as<std::string>()==n.ptc_sha)bound=true;
        require(bound,"FRUIT foreign network parent");
        PtcSolverRequest solver;solver.rank=n.ptc["rank"].as<int>();solver.iteration_limit=n.ptc["iteration_limit"].as<int>();
        solver.relative_objective_tolerance=n.ptc["relative_objective_tolerance"].as<double>();solver.relative_rank_tolerance=n.ptc["relative_rank_tolerance"].as<double>();
        require(solver.rank==10 && n.ptc["method"].as<std::string>()=="observed-als","FRUIT changed PTC recipe");
        for(std::size_t si=0;si<n.ptc["segments"].size();++si) {
            const auto ts=std::chrono::steady_clock::now();const auto old=load_map_segment(n,si,ledger);const auto s=n.ptc["segments"][si];
            const auto nt=s["scheduled_times"].as<std::size_t>(),nd=s["detectors"].as<std::size_t>(),no=s["output_detectors"].as<std::size_t>();
            const auto columns=s["fit_columns_in_output"].as<std::vector<std::size_t>>(),detectors=s["output_detector_indices"].as<std::vector<std::size_t>>();
            require(nd==columns.size(),"FRUIT compact fit domain mismatch");
            const auto stem="segment-"+std::to_string(si),name="network-"+std::to_string(n.network)+"-"+stem;
            const auto dir=output/name;fs::create_directory(dir);
            const auto data=map_read_array<double>(n.ptcroot/(stem+"-input.f64"),nt*nd);const auto mask=map_read_array<std::uint8_t>(n.ptcroot/(stem+"-eligible.u8"),nt*nd);
            const PtcMatrix y=Eigen::Map<const PtcMatrix>(data.data(),nt,nd);const PtcMask eligible=Eigen::Map<const PtcMask>(mask.data(),nt,nd);
            Eigen::VectorXd times(nt);for(std::size_t t=0;t<nt;++t)times[t]=n.native_times[n.first+old->identity().slots[t]*n.factor];
            const auto schedule=evaluate_ast_rtc_schedule(*n.observation.trajectory,*n.observation.offsets,n.observation.ra0,n.observation.dec0,times);
            PtcMatrix xy(nt,2*no);for(std::size_t d=0;d<no;++d){const auto [x,z]=n.geometry[detectors[d]];const auto v=schedule.detector_tangent_deg(x,z);xy.col(2*d)=v.col(0);xy.col(2*d+1)=v.col(1);}
            PtcMatrix operand=PtcMatrix::Zero(nt,nd);PtcMask supported=PtcMask::Ones(nt,nd),classes=PtcMask::Zero(nt,nd);std::array<std::size_t,4> counts{};
            for(std::size_t t=0;t<nt;++t)for(std::size_t d=0;d<nd;++d)if(eligible(t,d)){
                const auto v=model.at(xy(t,2*columns[d]),xy(t,2*columns[d]+1));classes(t,d)=std::uint8_t(v.support);++counts[classes(t,d)];
                if(mode!="zero"){operand(t,d)=v.value;supported(t,d)=v.support!=FruitModelSupport::unavailable;}
            }
            write_matrix(dir/"operand.f64",operand);map_write_array<std::uint8_t>(dir/"model-support.u8",{classes.data(),std::size_t(classes.size())});
            const auto residual_id="sha256:"+request_sha+":"+name+":y="+s["files"][stem+"-input.f64"].as<std::string>()+":operand="+citlali::utils::sha256_file(dir/"operand.f64")+":support="+citlali::utils::sha256_file(dir/"model-support.u8");
            const auto residual=FruitResidual::subtract(residual_id,y,eligible,std::move(operand),std::move(supported));
            YAML::Node seg;seg["source_root"]=n.ptcroot.string();seg["source_receipt_sha256"]=n.ptc_sha;seg["source_segment"]=s;seg["residual_identity"]=residual_id;
            seg["loading_projection_serialization_seconds"]=elapsed(ts);seg["projection_support_counts"]=counts;
            const auto learned=FruitLearned::learn(residual,solver);const auto applied=learned.apply(residual);const auto &fit=learned.fit();
            PtcMatrix clean=PtcMatrix::Zero(nt,no),total=clean,restore=clean;PtcMask causes=PtcMask::Constant(nt,no,17);
            for(std::size_t d=0;d<nd;++d){clean.col(columns[d])=applied.cleaned.values.col(d);total.col(columns[d])=applied.total.col(d);restore.col(columns[d])=residual->operand().col(d);causes.col(columns[d])=applied.cleaned.causes.col(d);}
            seg["converged"]=fit.converged;seg["stopping_reason"]=fit.stopping_reason;seg["rank"]=solver.rank;seg["iterations"]=fit.iterations;seg["objective"]=fit.objective;
            seg["preparation_seconds"]=learned.prepared().preparation_seconds;seg["learning_seconds"]=fit.fit_seconds;seg["application_seconds"]=applied.cleaned.seconds;
            seg["retained"]=applied.cleaned.retained;seg["application_failed_times"]=applied.cleaned.failed_times;
            seg["input_eligibility_changed"]=std::size_t((residual->eligible()!=eligible).count());seg["output_cause_changed"]=std::size_t((causes!=old->causes()).count());
            seg["zero_replay_maximum_cleaned_difference"]=(clean-old->values()).cwiseAbs().maxCoeff();
            const auto oldbasis=map_read_array<double>(n.ptcroot/(stem+"-basis.f64"),s["basis_rows"].as<std::size_t>()*s["basis_columns"].as<std::size_t>());
            // write_matrix serializes rows explicitly, regardless of Eigen's
            // in-memory basis layout. This is a diagnostic, never learned state.
            const Eigen::Map<const PtcMatrix> b(oldbasis.data(),s["basis_rows"].as<int>(),s["basis_columns"].as<int>());
            if(b.rows()==fit.basis.rows() && b.cols()==fit.basis.cols() && b.size())seg["basis_projector_Frobenius_difference"]=(b*b.transpose()-fit.basis*fit.basis.transpose()).norm();
            write_matrix(dir/"mean.f64",learned.prepared().mean);write_matrix(dir/"basis.f64",fit.basis);write_matrix(dir/"cleaned.f64",clean);
            map_write_array<std::uint8_t>(dir/"causes.u8",{causes.data(),std::size_t(causes.size())});
            for(const auto *file:{"operand.f64","model-support.u8","mean.f64","basis.f64","cleaned.f64","causes.u8"})seg["files"][file]=citlali::utils::sha256_file(dir/file);
            auto id=old->identity();id.application=residual_id+":source="+std::string(CITLALI_GIT_VERSION);
            id.publication=id.application+":cleaned="+seg["files"]["cleaned.f64"].as<std::string>()+":basis="+seg["files"]["basis.f64"].as<std::string>()+":mean="+seg["files"]["mean.f64"].as<std::string>();
            id.support_reference=id.publication+":causes="+seg["files"]["causes.u8"].as<std::string>();
            const auto q=PtcMapPublication::bind_verified_segment(id,std::move(clean),causes,fit.converged);
            seg["PTC_publication"]=id.publication;seg["PTC_support"]=id.support_reference;
            if(q->retained()){
                const auto g=PtcUniformCoefficients::publish(q,{"sha256:"+owner_sha,std::string(PtcUniformCoefficients::family)},id.publication+":uniform",id.support_reference);
                const auto admission=ValMapAdmission::bind(q,g,{id.publication,id.cal_parent,id.rtc_grid,id.detectors,id.slots});
                seg["MAP_first_input_row"]=totals.size();std::size_t admitted=0;
                for(std::size_t t=0;t<nt;++t)for(std::size_t d=0;d<no;++d)if(q->member(t,d)){
                    const auto row=totals.size();totals.push_back(total(t,d));residuals.push_back(q->values()(t,d));operands.push_back(restore(t,d));
                    const auto decision=admission.at(t,d,n.cal_causes[detectors[d]][id.slots[t]]);if(!decision.pass())continue;
                    const auto gamma=g->value(decision.coefficient_qc,t,d);require(gamma && *gamma==1,"FRUIT uniform coefficient mismatch");
                    const auto lon=xy(t,2*d),lat=xy(t,2*d+1);if(!std::isfinite(lon) || !std::isfinite(lat))continue;
                    coords.push_back({row,lon,lat});extent_x=std::max(extent_x,std::abs(lon));extent_y=std::max(extent_y,std::abs(lat));++admitted;
                }seg["MAP_admitted"]=admitted;
            }
            seg["wall_seconds"]=elapsed(ts);write_yaml(dir/"receipt.yaml",seg);record["segments"].push_back(seg);
        }
    }
    require(seen.size()==expected_networks && !totals.empty(),"FRUIT incomplete requested observation array");
    const auto mapping=std::chrono::steady_clock::now();const auto wcs=AstRtcTanGrid::centered(ra0,dec0,2,extent_x,extent_y);
    const std::string domain="sha256:"+request_sha+":new-ordered-PTC-retained";mapmaking::ForwardMapGrid grid{wcs.rows,wcs.columns,domain+":grid"};
    std::vector<mapmaking::ForwardMapMember> members;members.reserve(coords.size());
    for(const auto &c:coords){const auto [x,z]=wcs.continuous_pixel(c.lon,c.lat);const auto p=mapmaking::containing_pixel(grid,x,z);require(bool(p),"FRUIT resolved grid failed to enclose admitted sample");members.push_back({c.row,*p,1});}
    const auto op=mapmaking::ForwardMapOperator::resolve(grid,domain,totals.size(),std::move(members),{mr["coverage_cut"].as<double>(),{}});
    const auto T=op.apply(domain,totals),R=op.apply(domain,residuals),A=op.apply(domain,operands);double error=0;
    for(std::size_t i=0;i<T.signal.size();++i)error=std::max(error,std::abs(T.signal[i]-R.signal[i]-A.signal[i]));
    record["decomposition_max_abs_mJy_per_nominal_beam"]=error;record["MAP_seconds"]=elapsed(mapping);
    require(error<1e-8,"FRUIT decomposition accounting failed");const auto serial=std::chrono::steady_clock::now();
    const double beam=std::array<double,3>{5,7.5,10}[ai];map_fits(output/"total.fits",wcs,op,T,array,beam);map_fits(output/"residual.fits",wcs,op,R,array,beam);map_fits(output/"restored.fits",wcs,op,A,array,beam);
    std::vector<std::int32_t> pixels(totals.size(),-1);for(const auto &m:op.members())pixels[m.input_row]=m.pixel;
    map_write_array<std::int32_t>(output/"operator-pixel.i32",pixels);map_write_array<std::size_t>(output/"output-rows.u64",op.output_rows());
    ledger.verify();YAML::Node manifest;for(const auto &[p,h]:ledger.files)manifest[p]=h;write_yaml(output/"input-manifest.yaml",manifest);
    for(const auto *file:{"total.fits","residual.fits","restored.fits","operator-pixel.i32","output-rows.u64","input-manifest.yaml"})record["files"][file]=citlali::utils::sha256_file(output/file);
    record["MAP_input_rows"]=totals.size();record["MAP_contributions"]=op.members().size();record["science_pixels"]=op.output_rows().size();record["coverage_cut"]=mr["coverage_cut"];record["reference_normalization"]=op.reference_normalization();
    record["MAP_policy"]="unchanged one-hot uniform coefficients and new eligibility; one operator for T1/R1/A1";
    record["model_only_output"]=false;record["uncertainty"]="unavailable for this generation";record["outcome"]="success";
    record["serialization_verification_seconds"]=elapsed(serial);record["wall_seconds"]=elapsed(began);record["cpu_seconds"]=double(std::clock()-cpu)/CLOCKS_PER_SEC;
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);record["peak_rss_native_units"]=usage.ru_maxrss;record["peak_rss_unit"]=
#ifdef __APPLE__
    "bytes";
#else
    "KiB";
#endif
    write_yaml(output/"receipt.yaml",record);return 0;
}
}
