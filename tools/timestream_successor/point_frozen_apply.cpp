// Bounded saved-state diagnostic. This harness never calls ptc_learn or admission.
#include <citlali/core/pipeline/timestream_ptc_numerics.h>
#include <citlali/core/utils/sha256.h>
#include <citlali_config/gitversion.h>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sys/resource.h>
using namespace citlali::pipeline;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
void check(bool ok,const std::string &s){if(!ok)throw std::runtime_error(s);}
PtcMatrix matrix(const YAML::Node &n,int rows,int cols) {
    fs::path p=n["path"].as<std::string>();check(citlali::utils::sha256_file(p)==n["sha256"].as<std::string>(),"input hash: "+p.string());
    check(fs::file_size(p)==std::size_t(rows)*cols*8,"matrix shape");PtcMatrix a(rows,cols);
    std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(a.data()),a.size()*8);check(bool(f),"matrix read");return a;
}
template<class T>std::vector<T> array(const YAML::Node &n,std::size_t count) {
    fs::path p=n["path"].as<std::string>();check(citlali::utils::sha256_file(p)==n["sha256"].as<std::string>(),"array hash");
    check(fs::file_size(p)==count*sizeof(T),"array shape");std::vector<T> v(count);std::ifstream f(p,std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()),count*sizeof(T));check(bool(f),"array read");return v;
}
int main(int argc,char **argv) {try {
    if(argc==2 && std::string(argv[1])=="--version"){std::cout<<CITLALI_GIT_VERSION<<'\n';return 0;}
    check(argc==3,"point_frozen_apply request.yaml NEW-output");const auto start=Clock::now();const auto cpu=std::clock();
    YAML::Node q=YAML::LoadFile(argv[1]),record;check(q["schema"].as<std::string>()=="point-frozen-apply-v1","request schema");
    fs::path out=argv[2];check(!fs::exists(out),"output exists");fs::create_directories(out);
    const int nt=q["nt"].as<int>(),nd=q["nd"].as<int>(),no=q["no"].as<int>(),np=q["pixels"].as<int>(),rank=q["rank"].as<int>();
    check(nt>0&&nd>0&&no>=nd&&np>0&&rank>0,"dimensions");const auto columns=q["columns"].as<std::vector<int>>();check(int(columns.size())==nd,"columns");
    const auto inputs=q["inputs"];auto y=matrix(inputs["y"],nt,nd),u=matrix(inputs["u"],nt,nd);
    auto vr=matrix(inputs["vr"],nt,nd),vh=matrix(inputs["vh"],nt,nd),mr=matrix(inputs["mr"],nd,1),mh=matrix(inputs["mh"],nd,1);
    auto cr=matrix(inputs["cleanr"],nt,no),ch=matrix(inputs["cleanh"],nt,no);
    auto maskdata=array<std::uint8_t>(inputs["mask"],nt*nd),rc=array<std::uint8_t>(inputs["causer"],nt*no),hc=array<std::uint8_t>(inputs["causeh"],nt*no);
    PtcMask mask=Eigen::Map<const PtcMask>(maskdata.data(),nt,nd);check(rc==hc,"branch causes differ");
    auto pixels=array<std::int32_t>(inputs["pixel"],nt*nd);
    PtcFit R,H;for(auto *f:{&R,&H}){f->request.rank=rank;f->request.relative_rank_tolerance=q["rank_tolerance"].as<double>();}
    R.converged=q["converged_R"].as<bool>();H.converged=q["converged_H"].as<bool>();
    R.basis=matrix(inputs["br"],nd,rank);H.basis=matrix(inputs["bh"],nd,rank);check(R.converged==H.converged,"convergence mismatch");
    record["loading_verification_seconds"]=seconds(start);double mean_seconds=0,apply_seconds=0,map_seconds=0;int calls=0;
    const auto at=Clock::now();const auto prepared=PtcPrepared::prepare(y,mask);mean_seconds+=seconds(at);
    auto J=[&](const PtcMatrix &z){const auto t=Clock::now();PtcMatrix result= PtcPrepared::prepare(z,mask).mean;mean_seconds+=seconds(t);return result;};
    auto center=[&](const PtcMatrix &z,const PtcMatrix &m){PtcMatrix a=PtcMatrix::Zero(nt,nd);for(int t=0;t<nt;++t)for(int d=0;d<nd;++d)if(mask(t,d))a(t,d)=z(t,d)-m(d,0);return a;};
    auto broadcast=[&](const PtcMatrix &m){PtcMatrix a=PtcMatrix::Zero(nt,nd);for(int t=0;t<nt;++t)for(int d=0;d<nd;++d)if(mask(t,d))a(t,d)=m(d,0);return a;};
    auto close=[&](const std::string &name,const PtcMatrix &a,const PtcMatrix &b,double atol=1e-8){
        check(a.rows()==b.rows()&&a.cols()==b.cols(),name+" shape");double mx=0,relative=0;bool pass=true;
        for(Eigen::Index i=0;i<a.size();++i){const double e=std::abs(a.data()[i]-b.data()[i]),scale=std::max(std::abs(a.data()[i]),std::abs(b.data()[i]));mx=std::max(mx,e);relative=std::max(relative,e/std::max(1.,scale));pass &= std::isfinite(e)&&e<=atol+1e-11*scale;}
        record["checks"][name]["max_abs"]=mx;record["checks"][name]["max_relative_floor_1"]=relative;check(pass,name+" tolerance");};
    auto Q=[&](const PtcFit &f,const PtcMatrix &z){auto v=ptc_response(prepared,f,z);apply_seconds+=v.seconds;++calls;
        for(int t=0;t<nt;++t)for(int d=0;d<nd;++d)check(v.causes(t,d)==rc[t*no+columns[d]],"Apply cause changed");return v.values;};
    PtcMatrix xr=PtcMatrix::Zero(nt,nd),xh=xr;
    double injection_error=0;for(int t=0;t<nt;++t)for(int d=0;d<nd;++d)if(mask(t,d)){
        xr(t,d)=y(t,d)-vr(t,d);xh(t,d)=(y(t,d)+u(t,d))-vh(t,d);
        const double e=std::abs(((y(t,d)+u(t,d))-y(t,d))-u(t,d));injection_error=std::max(injection_error,e);
        check(e<=8*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(y(t,d)),std::abs(u(t,d))}),"injection identity");
    }
    const PtcMatrix d=vh-vr;close("residual_identity",xh-xr,u-d);close("saved_mean_R",J(xr),mr,1e-9);close("saved_mean_H",J(xh),mh,1e-9);
    const auto ju=J(u),jd=J(d);close("mean_difference",mh-mr,ju-jd);
    auto ar=Q(R,center(xr,mr)),ah=Q(H,center(xh,mh));PtcMatrix expectedr(nt,nd),expectedh(nt,nd);
    for(int j=0;j<nd;++j){expectedr.col(j)=cr.col(columns[j]);expectedh.col(j)=ch.col(columns[j]);}
    close("self_R",ar,expectedr,1e-9);close("self_H",ah,expectedh,1e-9);
    auto arplus=Q(R,center(xr+u,mr)),arh=Q(R,center(xh,mr)),cross=Q(R,center(xh,mh));
    auto direct=Q(R,u),feedback=PtcMatrix(d-Q(R,d)),cmu=PtcMatrix(-Q(R,broadcast(mh-mr))),cq=PtcMatrix(ah-cross),state=PtcMatrix(ah-arh);
    close("D_direct",direct,arplus-ar);close("F_direct",feedback,arh-arplus+d);close("Cmu_direct",cmu,cross-arh);close("state_split",state,cmu+cq);
    close("four_term_timestream",direct+feedback+cmu+cq,(ah+vh)-(ar+vr));
    const auto dc=Q(R,center(u,ju));PtcMatrix fc=d-Q(R,center(d,jd));close("centered_regroup",dc+fc+cq,direct+feedback+cmu+cq);
    std::map<std::string,Eigen::VectorXd> maps;
    auto mapped=[&](const std::string &name,const PtcMatrix &z){const auto t=Clock::now();Eigen::VectorXd v=Eigen::VectorXd::Zero(np);
        for(Eigen::Index i=0;i<z.size();++i)if(pixels[i]>=0){check(pixels[i]<np,"MAP pixel");check(rc[(i/nd)*no+columns[i%nd]]==0,"MAP invalid entry");v[pixels[i]]+=z.data()[i];}
        maps[name]=v;map_seconds+=seconds(t);};
    mapped("SELF_R",ar+vr);mapped("SELF_H",ah+vh);mapped("IDEAL",u);mapped("D",direct);mapped("F",feedback);mapped("S",state);mapped("CMU",cmu);mapped("CQ",cq);
    mapped("DC",dc);mapped("FC",fc);mapped("CENTER",-broadcast(ju));mapped("PROJECTION",dc-center(u,ju));
    PtcMatrix partition=PtcMatrix::Zero(nt,nd),fsum=partition,fcsum=partition;
    for(const auto &name:{"RETAIN","ADD","REMOVE"}){PtcMatrix part=PtcMatrix::Zero(nt,nd);
        for(Eigen::Index i=0;i<d.size();++i){check(vr.data()[i]>=0&&vh.data()[i]>=0,"nonpositive selected candidate semantics");const bool r=vr.data()[i]!=0,h=vh.data()[i]!=0;
            if((std::string(name)=="RETAIN"&&r&&h)||(std::string(name)=="ADD"&&!r&&h)||(std::string(name)=="REMOVE"&&r&&!h))part.data()[i]=d.data()[i];}
        partition+=part;PtcMatrix fp=part-Q(R,part),fcp=part-Q(R,center(part,J(part)));fsum+=fp;fcsum+=fcp;
        mapped(std::string("F_")+name,fp);mapped(std::string("FC_")+name,fcp);
    }
    close("candidate_partition",partition,d,0);close("F_partition",fsum,feedback);close("FC_partition",fcsum,fc);
    if(q["reverse"].as<bool>()){
        auto dp=Q(H,u);PtcMatrix fp=d-Q(H,d),mp=-Q(H,broadcast(mh-mr)),qp=Q(H,center(xr,mr))-ar;
        close("reverse_closure",dp+fp+mp+qp,(ah+vh)-(ar+vr));mapped("REV_D",dp);mapped("REV_F",fp);mapped("REV_CMU",mp);mapped("REV_CQ",qp);
    }
    if(inputs["oldvr"] || inputs["oldvh"]){
        check(inputs["oldvr"] && inputs["oldvh"],"both old operand branches required");
        const auto oldr=matrix(inputs["oldvr"],nt,nd),oldh=matrix(inputs["oldvh"],nt,nd);
        const PtcMatrix change=(vh-vr)-(oldh-oldr);
        const PtcMatrix history=change-Q(R,center(change,J(change)));mapped("HISTORY",history);
        PtcMatrix partition=PtcMatrix::Zero(nt,nd);
        for(int branch=0;branch<2;++branch){const auto &now=branch?vh:vr;const auto &old=branch?oldh:oldr;
            for(const auto &name:{"RETAIN","ADD","REMOVE"}){PtcMatrix piece=PtcMatrix::Zero(nt,nd);
                for(Eigen::Index i=0;i<piece.size();++i){check(old.data()[i]>=0&&now.data()[i]>=0,"history candidate positive semantics");const bool a=old.data()[i]!=0,b=now.data()[i]!=0;
                    if((std::string(name)=="RETAIN"&&a&&b)||(std::string(name)=="ADD"&&!a&&b)||(std::string(name)=="REMOVE"&&a&&!b))piece.data()[i]=now.data()[i]-old.data()[i];}
                const PtcMatrix effect=piece-Q(R,center(piece,J(piece)));
                mapped(std::string("HISTORY_")+(branch?"H_":"R_")+name,effect);
                partition+=(branch?1.:-1.)*effect;
            }
        }close("paired_history_partition",partition,history);
    }
    record["injection_identity_max_abs"]=injection_error;record["native_learn_calls"]=0;record["admission_updates"]=0;record["apply_calls"]=calls;
    record["source_revision"]=std::string(CITLALI_GIT_VERSION);record["request_sha256"]=citlali::utils::sha256_file(argv[1]);
    const auto serial=Clock::now();std::ofstream file(out/"numerators.f64",std::ios::binary);
    for(const auto &[name,v]:maps){record["components"].push_back(name);file.write(reinterpret_cast<const char*>(v.data()),v.size()*8);}file.close();check(bool(file),"output write");
    record["serialization_seconds"]=seconds(serial);record["mean_seconds"]=mean_seconds;record["Apply_seconds"]=apply_seconds;record["MAP_seconds"]=map_seconds;
    record["wall_seconds"]=seconds(start);record["cpu_seconds"]=double(std::clock()-cpu)/CLOCKS_PER_SEC;struct rusage usage{};getrusage(RUSAGE_SELF,&usage);record["peak_rss_native_units"]=usage.ru_maxrss;
#ifdef __APPLE__
    record["peak_rss_unit"]="bytes";
#else
    record["peak_rss_unit"]="KiB";
#endif
    record["status"]="pass";std::ofstream receipt(out/"receipt.yaml");receipt<<record;check(bool(receipt),"receipt write");return 0;
 }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
