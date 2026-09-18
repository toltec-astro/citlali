// Bounded solver cost replay. It consumes a checksummed preserved PTC input,
// not a substitute CAL producer; output is diagnostic, never production VAL.
#include <citlali/core/pipeline/timestream_ptc_numerics.h>
#include <citlali/core/utils/sha256.h>
#include <citlali_config/gitversion.h>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sys/resource.h>
using namespace citlali::pipeline;
namespace fs=std::filesystem;
int main(int argc,char **argv) {
 try {
    if(argc!=7)throw std::invalid_argument("input-ptc-dir|synthetic-overlap|synthetic-staggered segment rank observed-als|pairwise-covariance tolerance NEW-output-dir");
    const auto began=std::chrono::steady_clock::now();const std::string input=argv[1],method=argv[4];const int segment=std::stoi(argv[2]);
    const fs::path output=argv[6];if(fs::exists(output))throw std::invalid_argument("output exists");
    PtcMatrix values;PtcMask mask;YAML::Node record;
    if(input=="synthetic-overlap" || input=="synthetic-staggered") {
        values.resize(610,400);mask=PtcMask::Ones(610,400);std::mt19937 random(20260918);std::normal_distribution<double> normal;
        Eigen::MatrixXd modes(610,18),loadings(400,18);
        for(int i=0;i<modes.rows();++i)for(int k=0;k<18;++k)modes(i,k)=normal(random);
        for(int j=0;j<400;++j)for(int k=0;k<18;++k)loadings(j,k)=normal(random)/(1.+k);
        values=modes*loadings.transpose();
        for(int t=0;t<610;++t)for(int d=0;d<400;++d)values(t,d)+=d*.01+.2*normal(random);
        for(int d=0;d<100;++d){const int first=input=="synthetic-overlap"?250:100+(d*3)%300;mask.block(first,d,60,1).setZero();}
        record["source"]="synthetic-independent-400-detector-population-seed-20260918;610-post-F2-samples-about-10s";
    } else {
        const fs::path root=input;const auto receipt=YAML::LoadFile((root/"receipt.yaml").string());
        if(receipt["schema"].as<std::string>()!="citlali-ptc-output-v1")throw std::invalid_argument("unsupported preserved input");
        const auto g=receipt["segments"][segment];const auto n=g["scheduled_times"].as<int>(),d=g["detectors"].as<int>();
        values.resize(n,d);mask.resize(n,d);const auto prefix="segment-"+std::to_string(segment);
        auto read=[&](const std::string &suffix,auto *data,std::size_t size){const auto name=prefix+suffix;
            if(citlali::utils::sha256_file(root/name)!=g["files"][name].as<std::string>())throw std::invalid_argument("preserved input checksum mismatch");
            if(fs::file_size(root/name)!=size)throw std::invalid_argument("preserved input shape mismatch");
            std::ifstream file(root/name,std::ios::binary);file.read(reinterpret_cast<char*>(data),size);if(!file)throw std::runtime_error("input read failed");};
        read("-input.f64",values.data(),values.size()*8);read("-eligible.u8",mask.data(),mask.size());
        for(int t=0;t<n;++t)for(int j=0;j<d;++j)if(!mask(t,j))values(t,j)=NAN;
        record["source_receipt_sha256"]=citlali::utils::sha256_file(root/"receipt.yaml");record["source_CAL_receipt_sha256"]=receipt["source_CAL_receipt_sha256"];
        record["source"]=root.string();record["segment"]=segment;record["network"]=g["network"];record["scan"]=g["scan"];
    }
    PtcSolverRequest request;request.rank=std::stoi(argv[3]);request.relative_objective_tolerance=std::stod(argv[5]);
    if(method!="observed-als" && method!="pairwise-covariance")throw std::invalid_argument("unknown method");
    request.method=method=="observed-als"?PtcMethod::observed_als:PtcMethod::pairwise_covariance;
    const auto prepared=PtcPrepared::prepare(values,mask);const auto fit=ptc_learn(prepared,request);const auto applied=ptc_apply(prepared,fit);
    PtcMatrix response(values.rows(),values.cols());
    for(int t=0;t<response.rows();++t)for(int d=0;d<response.cols();++d)response(t,d)=std::exp(-.5*std::pow((t-280.-d*.05)/2.,2));
    const auto propagated=ptc_response(prepared,fit,response);
    record["schema"]="ptc-cost-probe-v1";record["method"]=method;record["rank"]=request.rank;record["relative_objective_tolerance"]=request.relative_objective_tolerance;
    record["times"]=values.rows();record["detectors"]=values.cols();record["eligible"]=prepared.eligible_count;record["complete_time_fraction"]=double(prepared.complete_times)/values.rows();
    record["time_mask_patterns"]=prepared.time_patterns.size();record["detector_mask_patterns"]=prepared.detector_patterns.size();
    record["converged"]=fit.converged;record["stopping_reason"]=fit.stopping_reason;record["iterations"]=fit.iterations;record["objective"]=fit.objective;
    record["preparation_seconds"]=prepared.preparation_seconds;record["initialization"]=fit.initialization;
    record["initialization_seconds"]=fit.initialization_seconds;record["fit_seconds"]=fit.fit_seconds;
    record["covariance_seconds"]=fit.covariance_seconds;record["decomposition_seconds"]=fit.decomposition_seconds;
    record["coefficient_seconds"]=fit.coefficient_seconds;record["basis_seconds"]=fit.basis_seconds;record["check_seconds"]=fit.check_seconds;
    record["coefficient_factorizations"]=fit.coefficient_factorizations;record["coefficient_factor_reuses"]=fit.coefficient_factor_reuses;
    record["apply_seconds"]=applied.seconds;record["response_seconds"]=propagated.seconds;record["response_scope"]="controlled CAL-grid probe, not complete astronomical response";
    record["retained"]=applied.retained;record["application_failed_times"]=applied.failed_times;record["apply_factorizations"]=applied.factorizations;record["apply_factor_reuses"]=applied.factor_reuses;
    record["threads"]=Eigen::nbThreads();record["build_identity"]=CITLALI_GIT_VERSION;record["run_reuse"]="preserved-CAL-input-only;cold-preparation-and-independent-rank-fit;no-warm-start";
    record["scientific_rank_selection"]=false;record["heldout_probe_executed"]=false;
    fs::create_directories(output);const auto write_at=std::chrono::steady_clock::now();
    auto write=[&](const std::string &name,const auto &matrix){std::ofstream stream(output/name,std::ios::binary);
        for(int i=0;i<matrix.rows();++i)for(int j=0;j<matrix.cols();++j){const double v=matrix(i,j);stream.write(reinterpret_cast<const char*>(&v),8);}stream.close();if(!stream)throw std::runtime_error("required probe output failed");};
    write("basis.f64",fit.basis);write("cleaned.f64",applied.values);
    record["output_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-write_at).count();
    record["wall_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
    struct rusage usage{};if(getrusage(RUSAGE_SELF,&usage)==0) {
#ifdef __APPLE__
        record["peak_rss_bytes"]=usage.ru_maxrss;
#else
        record["peak_rss_bytes"]=usage.ru_maxrss*1024;
#endif
    }
    record["peak_memory_scope"]="one cost-probe process including input, preparation, fit, apply and response;excludes-RTC-CAL";
    YAML::Emitter yaml;yaml.SetDoublePrecision(17);yaml<<record;std::ofstream stream(output/"result.yaml");stream<<yaml.c_str()<<'\n';stream.close();if(!stream)throw std::runtime_error("required receipt failed");
    std::cout<<method<<" rank="<<request.rank<<" "<<fit.stopping_reason<<" fit="<<fit.fit_seconds<<" apply="<<applied.seconds<<'\n';return 0;
 }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
