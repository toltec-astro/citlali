// Private successor boundary: exact saved-input parsing and publication.
// Model arithmetic/lifecycle, PTC, AST and MAP remain in their typed owners.
#include <citlali/core/pipeline/fruit_cycle.h>
#include <future>
#include <cstring>
#include <deque>
#include <functional>
#include <optional>
namespace {
int execute_saved_fruit(const fs::path &input,const fs::path &output,bool validate_only) {
    const auto began=std::chrono::steady_clock::now();const auto cpu=std::clock();
    auto elapsed=[](auto t){return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();};
    require(output.is_absolute() && !fs::exists(output),"FRUIT requires a fresh absolute output");
    MapFileLedger ledger;const auto request_sha=ledger.bind(input);const auto request=YAML::LoadFile(input.string());
    const std::set<std::string> fields{"schema","method","mode","array","map_request","parent_plan","candidate","owner_binding","cycle_index","previous_state","branch","injection"};
    for(const auto &e:request)require(fields.contains(e.first.as<std::string>()),"FRUIT unknown request field");
    const bool refseq=request["schema"].as<std::string>()=="citlali-point-fruit-refseq-v1";
    const bool corr=request["schema"].as<std::string>()=="citlali-point-fruit-corr-neighbor-v1";
    const bool reference=request["schema"].as<std::string>()=="citlali-bright-point-reference-v1";
    const bool speed_experiment=request["schema"].as<std::string>()=="citlali-point-speed-pca-v1";
    const bool rank_experiment=speed_experiment || request["schema"].as<std::string>()=="citlali-point-ptc-rank-v1";
    const bool independent=rank_experiment || reference || request["schema"].as<std::string>()=="citlali-point-fruit-independent-v1";
    const bool guarded=independent || request["schema"].as<std::string>()=="citlali-point-fruit-guarded-v1";
    const bool campaign=guarded || corr || refseq || request["schema"].as<std::string>()=="citlali-point-fruit-cd-v1";
    const bool continuation=request["schema"].as<std::string>()=="citlali-point-fruit-continuation-v1";
    require((guarded && request["method"].as<std::string>()==(speed_experiment?"point-speed-pca-v1":rank_experiment?"point-ptc-rank-v1":reference?"bright-point-base-v1":independent?"point-152430-no-upper-coverage-veto-v1":"point-152389-guarded-feedback-v1")) || (corr && request["method"].as<std::string>()=="point-152389-corr-neighbor-v1") || (refseq && request["method"].as<std::string>()=="point-152389-a2000-ref-basis-sequence-v1") || (campaign && !refseq && !corr && !guarded && request["method"].as<std::string>()=="point-152389-response-robustness-cd-v1") || (continuation && request["method"].as<std::string>()=="point-152389-direct-ten-cycle-v1") ||
        (!continuation && request["schema"].as<std::string>()=="citlali-point-fruit-cycle-v1" && request["method"].as<std::string>()=="point-152389-direct-one-cycle-v1"),"FRUIT unsupported method");
    const auto mode=request["mode"].as<std::string>();require(mode=="zero" || mode=="reference","FRUIT unknown mode");
    const int cycle=(continuation || campaign)?request["cycle_index"].as<int>():(mode=="zero"?0:1);
    require(!continuation || ((mode=="zero" && cycle==0) || (mode=="reference" && cycle>=2 && cycle<=10)),"FRUIT continuation limited to cycles 2 through 10");
    const std::string branch=campaign?request["branch"].as<std::string>():"";
    const std::set<std::string> branches{"REF","C-core-plus","C-core-minus","C-shoulder","D-seed-low","D-seed-high","D-seed-shoulder","D-threshold-5"};
    require(!campaign || ((refseq?(branch=="REFSEQ-REF" || branch=="REFSEQ-SHOULDER"):branches.contains(branch)) && cycle>=0 && cycle<=10 &&
        ((cycle==0 && mode=="zero" && branch.rfind("D-",0)!=0) || (cycle>0 && mode=="reference"))),"FRUIT C+D branch/index/mode outside bound");
    require(campaign || (!request["branch"] && !request["injection"]),"FRUIT foreign campaign fields");
    const auto owner_path=checked_file(request["owner_binding"]);const auto owner_sha=ledger.bind(owner_path);
    const auto owner=YAML::LoadFile(owner_path.string());
    bool step_experiment=false,four_network_rtc=false;
    if(speed_experiment){
      const auto p=YAML::LoadFile(checked_file(owner["prospective"]).string());
      step_experiment=p["campaign_scope"] && p["campaign_scope"].as<std::string>()=="rtc-step-point-v1";
      // One interpretation feeds every array/network/cycle gate below.
      const auto scope=p["campaign_scope"]?p["campaign_scope"].as<std::string>():"";
      four_network_rtc=scope=="rtc-step-consequences-v1" || scope=="rtc-retained-residuals-v1";
      if(four_network_rtc)require(p["observation"].as<int>()==152430 && owner["observation"].as<int>()==152430 &&
        p["array"].as<std::string>()=="a1100" && p["networks"].as<std::vector<int>>()==std::vector<int>({1,2,3,4}) &&
        p["ptc"]["rank"].as<int>()==10 && p["experimental_method"].as<std::string>()=="warm-observed-als" &&
        p["feedback_count"].as<int>()==5,"FRUIT four-network step consequence scope");
      if(step_experiment)require((p["observation"].as<int>()==152430 || p["observation"].as<int>()==152420) &&
        p["observation"].as<int>()==owner["observation"].as<int>() && p["ptc"]["rank"].as<int>()==10 &&
        p["experimental_method"].as<std::string>()=="warm-observed-als" && p["feedback_count"].as<int>()==5,
        "FRUIT step campaign scope");
    }
    require(owner["identity"].as<std::string>()==(speed_experiment?"POINT-SPEED-PCA-v1":rank_experiment?"POINT-PTC-RANK-v1":reference?"BRIGHT-POINT-BASE-v1":independent?"POINT-152430-NO-UPPER-COVERAGE-VETO-2026-09-26":guarded?"POINT-152389-GUARDED-OWNER-2026-09-25":corr?"POINT-152389-CORR-NEIGHBOR-OWNER-2026-09-24":refseq?"POINT-152389-A2000-REFSEQ-OWNER-2026-09-24":campaign?"POINT-152389-CD-OWNER-2026-09-23":continuation?"POINT-152389-TEN-FRUIT-OWNER-2026-09-23":"POINT-152389-FIRST-FRUIT-OWNER-2026-09-23") &&
        owner["outside_disk_no_feedback"].as<bool>() && owner["direct_identical_operand_rejoin"].as<bool>() &&
        owner["gain"].as<double>()==1 && (speed_experiment?(owner["nonzero_cycles"].as<int>()>=0 && owner["nonzero_cycles"].as<int>()<=10 && cycle<=owner["nonzero_cycles"].as<int>()):owner["nonzero_cycles"].as<int>()==((continuation || campaign)?10:1)),"FRUIT owner choices absent");
    const auto map_path=checked_file(request["map_request"]);ledger.bind(map_path);const auto mr=YAML::LoadFile(map_path.string());
    const auto pp=checked_file(request["parent_plan"]);ledger.bind(pp);const auto parent=YAML::LoadFile(pp.string());
    const auto array=request["array"].as<std::string>();const std::array<std::string,3> arrays{"a1100","a1400","a2000"};
    require(!four_network_rtc || array=="a1100","FRUIT step consequence request array");
    const auto ai=std::find(arrays.begin(),arrays.end(),array)-arrays.begin();require(ai<3 && parent["array"].as<std::string>()==array && parent["observation"].as<std::string>()==((reference || step_experiment)?("obs:"+std::to_string(owner["observation"].as<int>())+"_0_2"):independent?"obs:152430_0_2":"obs:152389_0_2") && (reference || four_network_rtc || !independent || array=="a2000"),"FRUIT foreign array/observation");
    require(!refseq || (array=="a2000" && owner["workers"].as<int>()==3 && owner["basis_schedule"].size()==11),"FRUIT reference sequence scope/resources");
    require(mr["schema"].as<std::string>()=="citlali-map-saved-ptc-v1" && mr["coverage_cut"].as<double>()==parent["coverage_cut"].as<double>() && mr["pixel_size_arcsec"].as<double>()==2,"FRUIT MAP policy differs");
    const auto mo=checked_file(mr["owner_binding"]);ledger.bind(mo);require(parent["owner_binding"].as<std::string>()=="sha256:"+citlali::utils::sha256_file(mo),"FRUIT MAP owner differs");
    if(corr) {
        const auto policy=owner["policy"].as<std::string>();
        require(policy=="BASE" || policy=="CORR" || policy=="DIAG","FRUIT unknown neighborhood policy");
        require(branch=="REF" || (policy!="BASE" && (branch=="C-core-plus" || branch=="C-core-minus" || branch=="C-shoulder" || branch=="D-seed-shoulder")),"FRUIT neighborhood branch outside matrix");
        require(cycle>=1,"FRUIT neighborhood reuses authenticated M0");
        ledger.bind(checked_file(owner["prospective"]));
        if(policy!="BASE") {
            const auto gate_path=checked_file(owner["launch_gate"]);ledger.bind(gate_path);
            const auto gate=YAML::LoadFile(gate_path.string());
            require(gate["policies"][array][policy]["pass"].as<bool>(),"FRUIT neighborhood launch gate failed");
        }
    }
    if(guarded) {
        const auto policy=owner["policy"].as<std::string>();
        require(policy=="BASE-POINT" || policy=="CORR" || policy=="DIAG","FRUIT unknown guarded policy");
        require((cycle>=1 || independent) && (branch=="REF" || branch=="C-core-plus" || branch=="C-core-minus" || branch=="C-shoulder" || branch=="D-seed-shoulder"),"FRUIT guarded matrix outside bound");
        require(owner["source_method"].as<std::string>()=="POINT-EXPECTED-v1","FRUIT guarded source method differs");
        ledger.bind(checked_file(owner["prospective"]));
        if(!independent) {
        const auto checks_path=checked_file(owner["source_checks"]);ledger.bind(checks_path);
        const auto checks=YAML::LoadFile(checks_path.string());
        require(checks["repair"][array]["usable"].as<bool>() && checks["repair"][array]["source_retention_check_pass"].as<bool>(),"FRUIT guarded source checks failed");
        if(policy!="BASE-POINT") {
            const auto gate_path=checked_file(owner["launch_gate"]);ledger.bind(gate_path);
            const auto gate=YAML::LoadFile(gate_path.string());
            require(gate["policies"][array][policy]["comparison_usable"].as<bool>(),"FRUIT guarded launch gate failed");
            const auto thresholds_path=checked_file(owner["thresholds"]);ledger.bind(thresholds_path);
            const auto thresholds=YAML::LoadFile(thresholds_path.string());
            require(owner["threshold"].as<double>()==thresholds["thresholds"][array][policy].as<double>() &&
                    owner["threshold"].as<double>()==gate["policies"][array][policy]["threshold"].as<double>(),"FRUIT guarded frozen threshold differs");
        }
        } else {
            const auto prospective=YAML::LoadFile(checked_file(owner["prospective"]).string());
            if(rank_experiment) {
                const int rank=prospective["ptc"]["rank"].as<int>();
                require(prospective["schema"].as<std::string>()==(speed_experiment?"citlali-point-speed-pca-v1":"citlali-point-ptc-rank-v1") &&
                    prospective["profile"].as<std::string>()==(speed_experiment?"POINT-SPEED-PCA-v1":"point-152430-ptc-rank-"+std::to_string(rank)) &&
                    (speed_experiment?(rank==10 || rank==15):(rank==5 || rank==10 || rank==15)) && (!speed_experiment || (prospective["feedback_count"].as<int>()==owner["nonzero_cycles"].as<int>() && (prospective["experimental_method"].as<std::string>()=="observed-als" || prospective["experimental_method"].as<std::string>()=="warm-observed-als" || prospective["experimental_method"].as<std::string>()=="masked-covariance-count-v1"))) && (prospective["ptc"]["method"].as<std::string>()=="observed-als" || (speed_experiment && prospective["experimental_method"].as<std::string>()=="masked-covariance-count-v1" && prospective["ptc"]["method"].as<std::string>()=="masked-covariance-count-v1")) &&
                    (step_experiment || prospective["observation"].as<int>()==152430) && prospective["subobservation"].as<int>()==0 && prospective["scan"].as<int>()==2 &&
                    (four_network_rtc?(prospective["array"].as<std::string>()=="a1100" && prospective["networks"].as<std::vector<int>>()==std::vector<int>({1,2,3,4})):(prospective["array"].as<std::string>()=="a2000" && prospective["networks"].as<std::vector<int>>()==std::vector<int>({11,12}))) &&
                    prospective["coverage_policy"].as<std::string>()=="high-coverage-informational" && prospective["cycles"].as<int>()==((prospective["campaign_scope"] && (prospective["campaign_scope"].as<std::string>()=="rtc-cost-response-v1" || prospective["campaign_scope"].as<std::string>()=="rtc-fixed11-v1" || prospective["campaign_scope"].as<std::string>()=="rtc-step-point-v1" || four_network_rtc))?5:10) &&
                    policy=="BASE-POINT" && owner["threshold"].as<double>()==3 && branch.rfind("D-",0)!=0,
                    "FRUIT bounded rank experiment scope");
            } else if(reference) {
                require(prospective["schema"].as<std::string>()=="citlali-bright-point-reference-v1" && prospective["observation"].as<int>()==owner["observation"].as<int>() && prospective["subobservation"].as<int>()==0 && prospective["scan"].as<int>()==2 && prospective["array"].as<std::string>()==array && prospective["coverage_policy"].as<std::string>()=="high-coverage-informational" && prospective["cycles"].as<int>()==10 && policy=="BASE-POINT" && owner["threshold"].as<double>()==3 && branch=="REF" && cycle>=1,"FRUIT frozen bright-POINT scope");
            } else require(prospective["observation"].as<int>()==152430 && prospective["coverage_policy"].as<std::string>()=="high-coverage-informational" && prospective["maximum_feedback_applications"].as<int>()==150 && prospective["maximum_trajectories"].as<int>()==15 && prospective["maximum_initial_physical_inputs"].as<int>()==4 && prospective["cycles"].as<int>()==10 && prospective["array"].as<std::string>()=="a2000" && (cycle>0 || branch.rfind("C-",0)==0), "FRUIT independent prospective scope");
            if(policy!="BASE-POINT") {
                const auto tp=checked_file(owner["thresholds"]);ledger.bind(tp);
                const auto thresholds=YAML::LoadFile(tp.string());
                require(owner["threshold"].as<double>()==thresholds["thresholds"][array][policy].as<double>(),"FRUIT independent frozen threshold differs");
            }
        }
    }
    // Resolve the declared four-network domain before loading detector arrays.
    // These checks run identically in execution and contract-only validation.
    if(four_network_rtc) {
        std::set<int> declared,parents;
        for(const auto &source:parent["sources"])
            require(parents.insert(source["network"].as<int>()).second,"FRUIT duplicate parent network");
        for(const auto &nr:mr["networks"]) {
            const auto cfg_path=checked_file(nr["input"]);ledger.bind(cfg_path);
            const auto cfg=YAML::LoadFile(cfg_path.string());const int nw=cfg["network"].as<int>();
            require(cfg["observation"].as<int>()==152430 && declared.insert(nw).second,"FRUIT four-network input identity");
            const auto root=fs::path(nr["root"].as<std::string>());
            const auto rtc_path=root/"receipt.yaml",ptc_path=root/"donor-continuity/ptc/receipt.yaml";
            ledger.bind(rtc_path,nr["receipt_sha256"].as<std::string>());
            ledger.bind(ptc_path,nr["ptc_receipt_sha256"].as<std::string>());
            const auto rtc=YAML::LoadFile(rtc_path.string()),ptc=YAML::LoadFile(ptc_path.string());
            require(rtc["observation"].as<int>()==152430 && rtc["network"].as<int>()==nw,"FRUIT four-network receipt identity");
            bool linked=false;
            for(const auto &source:parent["sources"])if(source["network"].as<int>()==nw)
                linked=source["PTC_receipt_sha256"].as<std::string>()==nr["ptc_receipt_sha256"].as<std::string>();
            require(linked && ptc["segments"].size()>0,"FRUIT four-network parent relation");
            for(const auto &segment:ptc["segments"])
                require(segment["network"].as<int>()==nw,"FRUIT four-network segment identity");
        }
        require(declared==std::set<int>({1,2,3,4}) && parents==declared,"FRUIT four-network declared networks");
    }
    const auto candidate=checked_file(request["candidate"]);const auto candidate_sha=ledger.bind(candidate);
    const std::array<std::string,3> selected_hashes{
        "8edea26bb5c5b5aca065a88c93a0a30fe5064f3cc3388f9423725ed6a63bc640",
        "a6e5a34d6251d3bb7b11b0bc4217fdd5a221e85d48fe08ec2fa5f4b82018d9ff",
        "aafe2afb4965bcd98bac1c1adf9a9dd889b606077af43b1486b2462c8f15d9ac"};
    if(!continuation && !campaign)require(candidate_sha==selected_hashes[ai],"FRUIT requires the exact reviewed threshold-3 candidate");
    else if(!campaign || cycle>0) {
        const auto previous_path=checked_file(request["previous_state"]);ledger.bind(previous_path);
        const auto previous=YAML::LoadFile(previous_path.string());
        require(previous["schema"].as<std::string>()=="point-fruit-ready-state-v1" && previous["array"].as<std::string>()==array &&
            previous["cycle_index"].as<int>()==(mode=="zero"?1:cycle-1) && previous["ready"].as<bool>(),"FRUIT predecessor unavailable or index mismatch");
        require(previous["candidate"]["sha256"].as<std::string>()==candidate_sha && checked_file(previous["candidate"])==candidate,
            "FRUIT predecessor candidate differs");
        require(previous["original_plan_sha256"].as<std::string>()==citlali::utils::sha256_file(pp) &&
            previous["original_map_request_sha256"].as<std::string>()==citlali::utils::sha256_file(map_path),"FRUIT original parent changed");
        if(campaign)require(previous["branch"].as<std::string>()==branch && previous["owner_sha256"].as<std::string>()==owner_sha,
            "FRUIT C+D predecessor branch/campaign mismatch");
        ledger.bind(checked_file(previous["total"]));ledger.bind(checked_file(previous["receipt"]));
        if(!campaign && cycle<=2)require(owner["seed_candidates"][array].as<std::string>()==candidate_sha,"FRUIT saved m1 differs");
    }
    if(campaign){
        require(owner["parents"][array]["map_request"]["sha256"].as<std::string>()==citlali::utils::sha256_file(map_path) &&
            owner["parents"][array]["parent_plan"]["sha256"].as<std::string>()==citlali::utils::sha256_file(pp),"FRUIT C+D foreign parent");
        if(cycle==0 && !independent)require(candidate_sha==selected_hashes[ai],"FRUIT M0 requires original candidate as unused grid control");
        if(cycle==1 && branch.rfind("D-",0)==0)require(candidate_sha==owner["seeds"][array][branch]["sha256"].as<std::string>(),"FRUIT C+D first seed differs");
    }
    auto read_model=[&](const fs::path &path,const std::string &sha){
    fitsfile *f=nullptr;int status=0,any=0;fits_open_file(&f,path.c_str(),READONLY,&status);
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
    return FruitPointModel("sha256:"+sha,{std::size_t(axes[1]),std::size_t(axes[0]),"candidate-grid"},cx,cy,dx,dy,std::move(brightness),std::move(valid));
    };
    auto model=read_model(candidate,candidate_sha);
    std::unique_ptr<FruitPointModel> injection;
    if(campaign && (branch.rfind("C-",0)==0 || branch=="REFSEQ-SHOULDER")){
        const auto ip=checked_file(request["injection"]);const auto ih=ledger.bind(ip);
        require(ih==owner["injections"][array][refseq?"C-shoulder":branch]["sha256"].as<std::string>(),"FRUIT C+D foreign injection");
        injection=std::make_unique<FruitPointModel>(read_model(ip,ih));
    }else require(!request["injection"],"FRUIT injection forbidden on REF/D");
    if(validate_only) {
        ledger.verify();fs::create_directories(output);YAML::Node validation;
        validation["schema"]="citlali-point-fruit-contract-validation-v1";
        validation["outcome"]="request-contract-accepted";
        validation["request_sha256"]=request_sha;validation["source_revision"]=std::string(CITLALI_GIT_VERSION);
        validation["cycle_index"]=cycle;validation["array"]=array;
        validation["four_network_rtc"]=four_network_rtc;
        validation["detector_arrays_loaded"]=false;validation["scientific_processing_executed"]=false;
        validation["scope"]="native request, owner, prospective scope, bound parent/predecessor, candidate model and declared network metadata; not numerical processing or realized-output validation";
        YAML::Node inputs;for(const auto &[p,h]:ledger.files)inputs[p]=h;validation["input_manifest"]=inputs;
        write_yaml(output/"validation.yaml",validation);return 0;
    }
    fs::create_directories(output);YAML::Node record;record["schema"]="citlali-point-fruit-cycle-output-v1";record["outcome"]="incomplete";
    record["request_sha256"]=request_sha;record["owner_sha256"]=owner_sha;record["source_revision"]=std::string(CITLALI_GIT_VERSION);record["mode"]=mode;record["array"]=array;
    record["cycle_index"]=cycle;if(continuation || (campaign && cycle>0))record["previous_state"]=request["previous_state"];
    if(campaign){record["branch"]=branch;record["injection_identity"]=injection?injection->identity():"none";
        record["injection_boundary"]="immutable post-RTC/CAL pre-PTC y; all eligible entries; never rejoined";}
    record["candidate_identity"]=model.identity();record["selected_model"]=mode=="zero"?"zero integration control":model.identity();record["unit"]="mJy/nominal-beam";
    if(refseq){record["basis_mode"]="externally supplied stage-matched original REF basis; no new learned state";record["learning_performed"]=false;record["workers"]=3;record["library_threads"]=1;}
    record["reference"]="source-relative model; fitted constant excluded; current-residual detector means not restored; rejoined model modes are model-supported, not recovered measured modes";
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
        auto n=load_saved_map_network(nr,ledger);require(n.obsnum==((reference || step_experiment)?owner["observation"].as<int>():independent?152430:152389) && seen.insert(n.network).second,"FRUIT network selection");
        if(std::isfinite(ra0))require(ra0==n.observation.ra0 && dec0==n.observation.dec0,"FRUIT AST center changed");
        else {ra0=n.observation.ra0;dec0=n.observation.dec0;}
        bool bound=false;for(const auto &p:parent["sources"])if(p["network"].as<int>()==n.network && p["PTC_receipt_sha256"].as<std::string>()==n.ptc_sha)bound=true;
        require(bound,"FRUIT foreign network parent");
        const auto speed_config=speed_experiment?YAML::LoadFile(checked_file(owner["prospective"]).string()):YAML::Node();
        const auto experimental_method=speed_experiment?speed_config["experimental_method"].as<std::string>():"observed-als";
        PtcSolverRequest solver;solver.rank=n.ptc["rank"].as<int>();solver.iteration_limit=n.ptc["iteration_limit"].as<int>();
        solver.relative_objective_tolerance=n.ptc["relative_objective_tolerance"].as<double>();solver.relative_rank_tolerance=n.ptc["relative_rank_tolerance"].as<double>();
        // The shared post-CAL carrier was produced with rank 10. Its saved basis
        // is diagnostic only: every experiment M0/residual learns from original y.
        if(rank_experiment) {
            const auto config=YAML::LoadFile(checked_file(owner["prospective"]).string());
            require((speed_experiment?solver.rank==config["ptc"]["rank"].as<int>():solver.rank==10) && solver.iteration_limit==100 && solver.relative_objective_tolerance==1e-5 && solver.relative_rank_tolerance==1e-10,
                "FRUIT rank experiment changed parent solver recipe");
            solver.rank=config["ptc"]["rank"].as<int>();
        }
        if(experimental_method=="masked-covariance-count-v1")solver.method=PtcMethod::masked_covariance_count;
        require((rank_experiment || (reference?solver.rank==YAML::LoadFile(checked_file(owner["prospective"]).string())["ptc"]["rank"].as<int>():solver.rank==10)) && n.ptc["method"].as<std::string>()==(experimental_method=="masked-covariance-count-v1"?"masked-covariance-count-v1":"observed-als"),"FRUIT changed PTC recipe");
        std::deque<std::function<void()>> pending;
        for(std::size_t si=0;si<n.ptc["segments"].size();++si) {
            const auto ts=std::chrono::steady_clock::now();const auto old=load_map_segment(n,si,ledger);const auto s=n.ptc["segments"][si];
            const auto nt=s["scheduled_times"].as<std::size_t>(),nd=s["detectors"].as<std::size_t>(),no=s["output_detectors"].as<std::size_t>();
            const auto columns=s["fit_columns_in_output"].as<std::vector<std::size_t>>(),detectors=s["output_detector_indices"].as<std::vector<std::size_t>>();
            require(nd==columns.size(),"FRUIT compact fit domain mismatch");
            const auto stem="segment-"+std::to_string(si),name="network-"+std::to_string(n.network)+"-"+stem;
            const auto dir=output/name;fs::create_directory(dir);
            const auto data=map_read_array<double>(n.ptcroot/(stem+"-input.f64"),nt*nd);const auto mask=map_read_array<std::uint8_t>(n.ptcroot/(stem+"-eligible.u8"),nt*nd);
            const PtcMatrix y=Eigen::Map<const PtcMatrix>(data.data(),nt,nd);const PtcMask eligible=Eigen::Map<const PtcMask>(mask.data(),nt,nd);
            const auto projection_started=std::chrono::steady_clock::now();
            const double loading_seconds=elapsed(ts);
            Eigen::VectorXd times(nt);for(std::size_t t=0;t<nt;++t)times[t]=n.native_times[n.first+old->identity().slots[t]*n.factor];
            const auto schedule=evaluate_ast_rtc_schedule(*n.observation.trajectory,*n.observation.offsets,n.observation.ra0,n.observation.dec0,times);
            PtcMatrix xy(nt,2*no);for(std::size_t d=0;d<no;++d){const auto [x,z]=n.geometry[detectors[d]];const auto v=schedule.detector_tangent_deg(x,z);xy.col(2*d)=v.col(0);xy.col(2*d+1)=v.col(1);}
            PtcMatrix operand=PtcMatrix::Zero(nt,nd);PtcMask supported=PtcMask::Ones(nt,nd),classes=PtcMask::Zero(nt,nd);std::array<std::size_t,4> counts{};
            for(std::size_t t=0;t<nt;++t)for(std::size_t d=0;d<nd;++d)if(eligible(t,d)){
                const auto v=model.at(xy(t,2*columns[d]),xy(t,2*columns[d]+1));classes(t,d)=std::uint8_t(v.support);++counts[classes(t,d)];
                if(mode!="zero"){operand(t,d)=v.value;supported(t,d)=v.support!=FruitModelSupport::unavailable;}
            }
            const double projection_seconds=elapsed(projection_started);
            const auto input_serialization_started=std::chrono::steady_clock::now();
            write_matrix(dir/"operand.f64",operand);map_write_array<std::uint8_t>(dir/"model-support.u8",{classes.data(),std::size_t(classes.size())});
            const auto residual_id="sha256:"+request_sha+":"+name+":y="+s["files"][stem+"-input.f64"].as<std::string>()+":operand="+citlali::utils::sha256_file(dir/"operand.f64")+":support="+citlali::utils::sha256_file(dir/"model-support.u8");
            PtcMatrix branch_y=y;double injection_error=0;std::array<std::size_t,4> injection_counts{};
            if(injection){
                PtcMatrix pi=PtcMatrix::Zero(nt,nd);PtcMask ps=PtcMask::Ones(nt,nd);
                for(std::size_t t=0;t<nt;++t)for(std::size_t d=0;d<nd;++d)if(eligible(t,d)){
                    const auto v=injection->at(xy(t,2*columns[d]),xy(t,2*columns[d]+1));
                    ++injection_counts[std::uint8_t(v.support)];pi(t,d)=v.value;ps(t,d)=v.support!=FruitModelSupport::unavailable;
                }
                if(refseq || (guarded && (!independent || cycle>0))){
                    const auto saved_pi=checked_file(owner["injection_operands"][name]);ledger.bind(saved_pi);
                    const auto exact_pi=map_read_array<double>(saved_pi,nt*nd);
                    require(std::equal(exact_pi.begin(),exact_pi.end(),pi.data()),"FRUIT saved injection operand differs");
                    const auto parent_path=checked_file(owner["injected_parents"][name]);ledger.bind(parent_path);
                    const auto parent_values=map_read_array<double>(parent_path,nt*nd);
                    branch_y=Eigen::Map<const PtcMatrix>(parent_values.data(),nt,nd);
                    if(guarded)branch_y=fruit_verified_injected_parent(y,eligible,pi,ps,
                        Eigen::Map<const PtcMatrix>(exact_pi.data(),nt,nd),branch_y);
                    else {
                        const auto expected=fruit_injected_parent(y,eligible,pi,ps);
                        require(std::memcmp(branch_y.data(),expected.data(),std::size_t(expected.size())*sizeof(double))==0,"FRUIT immutable shoulder parent differs");
                    }
                }else branch_y=fruit_injected_parent(y,eligible,pi,ps);
                for(Eigen::Index k=0;k<y.size();++k)if(eligible.data()[k]){
                    const double err=std::abs((branch_y.data()[k]-y.data()[k])-pi.data()[k]);
                    injection_error=std::max(injection_error,err);
                    require(err<=8*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(y.data()[k]),std::abs(pi.data()[k])}),"FRUIT injection identity failed");
                }
                write_matrix(dir/"injection.f64",pi);
                if(independent && cycle==0)write_matrix(dir/"injected-parent.f64",branch_y);
            }
            const double input_serialization_seconds=elapsed(input_serialization_started);
            const auto subtraction_started=std::chrono::steady_clock::now();
            const auto residual=FruitResidual::subtract(residual_id,branch_y,eligible,std::move(operand),std::move(supported));
            const double subtraction_seconds=elapsed(subtraction_started);
            YAML::Node seg;seg["loading_seconds"]=loading_seconds;seg["projection_seconds"]=projection_seconds;seg["input_serialization_and_injection_seconds"]=input_serialization_seconds;seg["subtraction_seconds"]=subtraction_seconds;seg["source_root"]=n.ptcroot.string();seg["source_receipt_sha256"]=n.ptc_sha;seg["source_segment"]=s;seg["residual_identity"]=residual_id;
            if(campaign){seg["injection_identity_max_abs"]=injection_error;seg["injection_projection_support_counts"]=injection_counts;
                seg["immutable_parent_reloaded"]=true;seg["invalid_parent_entries_unchanged"]=true;
                if(injection)seg["files"]["injection.f64"]=citlali::utils::sha256_file(dir/"injection.f64");}
            seg["loading_projection_serialization_seconds"]=elapsed(ts);seg["projection_support_counts"]=counts;
            // Scope includes immutable source/detector/support identities, not residual
            // values or means. A predecessor only initializes a fully new solve.
            const std::string warm_scope=speed_experiment?(citlali::utils::sha256_file(checked_file(owner["prospective"]))+":"+branch+":"+name+":"+n.ptc_sha+":"+s["files"][stem+"-eligible.u8"].as<std::string>()):"";
            std::optional<PtcWarmStart> warm;
            const auto warm_at=std::chrono::steady_clock::now();
            if(experimental_method=="warm-observed-als" && cycle>0) {
                PtcFit previous_fit;previous_fit.request=solver;fs::path basis_path;std::string basis_hash;
                if(cycle==1 && branch=="REF") {
                    // The branch's own freshly generated cold M0, never another run.
                    previous_fit.converged=s["converged"].as<bool>();
                    require(s["application_failed_times"].as<std::size_t>()==0,"warm M0 application failed");
                    basis_path=n.ptcroot/(stem+"-basis.f64");basis_hash=s["files"][stem+"-basis.f64"].as<std::string>();
                    seg["warm_predecessor"]="own-raw-derived-cold-M0";
                } else {
                    const auto prior=YAML::LoadFile(checked_file(request["previous_state"]).string());
                    const auto receipt_path=checked_file(prior["receipt"]);ledger.bind(receipt_path);
                    const auto receipt=YAML::LoadFile(receipt_path.string());
                    require(receipt["cycle_index"].as<int>()==cycle-1 && receipt["branch"].as<std::string>()==branch,"warm predecessor trajectory/cycle mismatch");
                    YAML::Node previous(YAML::NodeType::Undefined);
                    for(const auto &entry:receipt["segments"])if(entry["warm_scope"] && entry["warm_scope"].as<std::string>()==warm_scope){require(!previous.IsDefined(),"duplicate warm scope");previous=YAML::Clone(entry);}
                    require(previous.IsDefined() && previous["source_receipt_sha256"].as<std::string>()==n.ptc_sha &&
                        previous["ptc_method"].as<std::string>()==experimental_method && previous["rank"].as<int>()==solver.rank &&
                        previous["realized_rank"].as<int>()==solver.rank && previous["input_eligibility_changed"].as<std::size_t>()==0 &&
                        previous["application_failed_times"].as<std::size_t>()==0,"warm predecessor identity/support mismatch");
                    previous_fit.converged=previous["converged"].as<bool>();
                    basis_path=receipt_path.parent_path()/name/"basis.f64";basis_hash=previous["files"]["basis.f64"].as<std::string>();
                    seg["warm_predecessor"]=prior["receipt"];
                }
                ledger.bind(basis_path,basis_hash);const auto values=map_read_array<double>(basis_path,nd*solver.rank);
                previous_fit.basis=Eigen::Map<const PtcMatrix>(values.data(),nd,solver.rank);
                warm=PtcWarmStart::from_converged(previous_fit,eligible,warm_scope);
            }
            seg["warm_scope"]=warm_scope;seg["ptc_method"]=experimental_method;
            seg["warm_initializer_used"]=bool(warm);seg["warm_validation_loading_seconds"]=elapsed(warm_at);
            // Only preparation/Apply runs concurrently; all ledger, publication and MAP
            // accumulation remain in original segment order on the calling thread.
            struct Stage {PtcPrepared prepared;PtcFit fit;FruitLearned::Result applied;double apply_restore_seconds=0;};
            Eigen::MatrixXd imported_basis;
            if(refseq){
                const auto binding=owner["basis_schedule"][cycle][name];
                require(binding["stage"].as<int>()==cycle && binding["rank"].as<int>()==solver.rank &&
                    binding["relative_rank_tolerance"].as<double>()==solver.relative_rank_tolerance &&
                    binding["parent_input_sha256"].as<std::string>()==s["files"][stem+"-input.f64"].as<std::string>() &&
                    binding["eligible_sha256"].as<std::string>()==s["files"][stem+"-eligible.u8"].as<std::string>(),"FRUIT foreign reference basis design/stage");
                const auto bp=checked_file(binding["basis"]);const auto bh=ledger.bind(bp);
                const auto bv=map_read_array<double>(bp,nd*solver.rank);
                imported_basis=Eigen::Map<const PtcMatrix>(bv.data(),nd,solver.rank);
                seg["basis_source"]=binding;seg["basis_identity"]="sha256:"+bh;
                seg["learning_performed"]=false;seg["basis_available"]=true;
            }
            const std::string basis_identity=refseq?seg["basis_identity"].as<std::string>():"";
            auto task=[residual,solver,imported_basis,refseq,basis_identity,warm,warm_scope]() {
                Stage v;
                if(refseq){
                    auto x=fruit_apply_reference_basis(residual,solver,imported_basis,basis_identity);
                    v.prepared=std::move(x.prepared);v.applied={std::move(x.cleaned),std::move(x.total)};
                    v.fit.request=solver;v.fit.basis=imported_basis;
                }else{
                    const auto learned=FruitLearned::learn(residual,solver,warm?&*warm:nullptr,warm_scope);
                    v.prepared=learned.prepared();v.fit=learned.fit();const auto at=std::chrono::steady_clock::now();v.applied=learned.apply(residual);v.apply_restore_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-at).count();
                }return v;
            };
            auto future=std::make_shared<std::future<Stage>>(std::async(refseq?std::launch::async:std::launch::deferred,std::move(task)));
            auto publish=[future,seg,nt,nd,no,columns,detectors,dir,residual,residual_id,old,s,stem,xy,ts,eligible,refseq,
                &n,&solver,&record,&totals,&residuals,&operands,&coords,&extent_x,&extent_y,&owner_sha,&elapsed]() mutable {
            auto stage=future->get();const auto &applied=stage.applied;const auto &fit=stage.fit;
            const bool state_available=refseq?true:fit.converged;
            PtcMatrix clean=PtcMatrix::Zero(nt,no),total=clean,restore=clean;PtcMask causes=PtcMask::Constant(nt,no,17);
            for(std::size_t d=0;d<nd;++d){clean.col(columns[d])=applied.cleaned.values.col(d);total.col(columns[d])=applied.total.col(d);restore.col(columns[d])=residual->operand().col(d);causes.col(columns[d])=applied.cleaned.causes.col(d);}
            if(!refseq){seg["converged"]=fit.converged;seg["stopping_reason"]=fit.stopping_reason;seg["iterations"]=fit.iterations;seg["objective"]=fit.objective;}
            else {seg["application_outcome"]=applied.cleaned.failed_times?"failed-times":"success";seg["application_factorizations"]=applied.cleaned.factorizations;seg["application_factor_reuses"]=applied.cleaned.factor_reuses;}
            seg["rank"]=solver.rank;
            seg["realized_rank"]=fit.basis.cols();seg["learning_performed"]=!refseq;
            seg["apply_restore_seconds"]=stage.apply_restore_seconds;
            seg["preparation_seconds"]=stage.prepared.preparation_seconds;seg["learning_seconds"]=fit.fit_seconds;seg["application_seconds"]=applied.cleaned.seconds;
            seg["initialization_seconds"]=fit.initialization_seconds;seg["covariance_seconds"]=fit.covariance_seconds;
            seg["decomposition_seconds"]=fit.decomposition_seconds;seg["coefficient_seconds"]=fit.coefficient_seconds;
            seg["basis_seconds"]=fit.basis_seconds;seg["check_seconds"]=fit.check_seconds;
            seg["initialization"]=fit.initialization;
            seg["signed_spectrum"]=fit.signed_spectrum;seg["overlap_counts"]=fit.overlap_counts;seg["overlap_frequencies"]=fit.overlap_frequencies;
            seg["eigen_relative_residual"]=fit.eigen_relative_residual;seg["eigen_residual_tolerance"]=fit.eigen_residual_tolerance;
            if(std::isfinite(fit.negative_to_positive_power))seg["negative_to_positive_power"]=fit.negative_to_positive_power;else seg["negative_to_positive_power"]=YAML::Null;if(std::isfinite(fit.most_negative_to_largest_positive))seg["most_negative_to_largest_positive"]=fit.most_negative_to_largest_positive;else seg["most_negative_to_largest_positive"]=YAML::Null;
            seg["retained"]=applied.cleaned.retained;seg["application_failed_times"]=applied.cleaned.failed_times;
            seg["input_eligibility_changed"]=std::size_t((residual->eligible()!=eligible).count());seg["output_cause_changed"]=std::size_t((causes!=old->causes()).count());
            seg["zero_replay_maximum_cleaned_difference"]=(clean-old->values()).cwiseAbs().maxCoeff();
            const auto oldbasis=map_read_array<double>(n.ptcroot/(stem+"-basis.f64"),s["basis_rows"].as<std::size_t>()*s["basis_columns"].as<std::size_t>());
            // write_matrix serializes rows explicitly, regardless of Eigen's
            // in-memory basis layout. This is a diagnostic, never learned state.
            const Eigen::Map<const PtcMatrix> b(oldbasis.data(),s["basis_rows"].as<int>(),s["basis_columns"].as<int>());
            if(b.rows()==fit.basis.rows() && b.cols()==fit.basis.cols() && b.size())seg["basis_projector_Frobenius_difference"]=(b*b.transpose()-fit.basis*fit.basis.transpose()).norm();
            write_matrix(dir/"mean.f64",stage.prepared.mean);write_matrix(dir/"basis.f64",fit.basis);write_matrix(dir/"cleaned.f64",clean);
            map_write_array<std::uint8_t>(dir/"causes.u8",{causes.data(),std::size_t(causes.size())});
            for(const auto *file:{"operand.f64","model-support.u8","mean.f64","basis.f64","cleaned.f64","causes.u8"})seg["files"][file]=citlali::utils::sha256_file(dir/file);
            auto id=old->identity();id.application=residual_id+":source="+std::string(CITLALI_GIT_VERSION);
            id.publication=id.application+":cleaned="+seg["files"]["cleaned.f64"].as<std::string>()+":basis="+seg["files"]["basis.f64"].as<std::string>()+":mean="+seg["files"]["mean.f64"].as<std::string>();
            id.support_reference=id.publication+":causes="+seg["files"]["causes.u8"].as<std::string>();
            const auto q=PtcMapPublication::bind_verified_segment(id,std::move(clean),causes,state_available);
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
            };
            pending.push_back(std::move(publish));
            if(pending.size()>=(refseq?3:1)){pending.front()();pending.pop_front();}
        }
        while(!pending.empty()){pending.front()();pending.pop_front();}
    }
    require(!four_network_rtc || seen==std::set<int>({1,2,3,4}),"FRUIT step consequence realized networks");
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
    record["serialization_seconds"]=elapsed(serial);const auto verification=std::chrono::steady_clock::now();
    ledger.verify();record["verification_seconds"]=elapsed(verification);YAML::Node manifest;for(const auto &[p,h]:ledger.files)manifest[p]=h;write_yaml(output/"input-manifest.yaml",manifest);
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
