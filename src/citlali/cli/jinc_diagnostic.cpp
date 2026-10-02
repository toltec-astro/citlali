// Bounded diagnostic application boundary; all deposition/certification is owned
// by mapmaking. This is not a production reduction or source fitter.
#include <citlali/core/mapmaking/jinc_discrete.h>
#include <yaml-cpp/yaml.h>
#include <citlali/core/utils/sha256.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
#include <limits>
#include <bit>
#include <ctime>
#include <numbers>
namespace fs=std::filesystem;
using namespace citlali::mapmaking;
void require(bool v,const char*m){if(!v)throw std::runtime_error(m);}
template<class T>std::vector<T> read(const YAML::Node&ref,std::size_t n){fs::path p=ref["path"].as<std::string>();require(citlali::utils::sha256_file(p)==ref["sha256"].as<std::string>(),"input hash mismatch");require(fs::file_size(p)==n*sizeof(T),"input shape mismatch");std::vector<T> v(n);std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));require(bool(f),"read failed");return v;}
template<class T>void write(const fs::path&p,const std::vector<T>&v,YAML::Node&receipt){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));f.close();require(bool(f),"write failed");receipt["files"][p.filename().string()]=citlali::utils::sha256_file(p);}
int main(int argc,char**argv){try{
 require(argc==3,"usage: citlali_jinc_diagnostic request.json fresh-output");const fs::path request=argv[1],out=argv[2];require(!fs::exists(out),"fresh output required");require(std::endian::native==std::endian::little,"little-endian required");auto q=YAML::LoadFile(request.string());const bool altaz=q["schema"].as<std::string>()=="oof-a2000-altaz-jinc-diagnostic-v1";const bool oof=altaz || q["schema"].as<std::string>()=="oof-a2000-jinc-diagnostic-v1";require(oof || q["schema"].as<std::string>()=="point-152389-jinc-diagnostic-v1","unknown diagnostic request");
 if(oof)require(q["observation"].as<int>()>=152385 && q["observation"].as<int>()<=152387 && q["subobservation"].as<int>()==0 && q["scan"].as<int>()==1 && q["array"].as<std::string>()=="a2000" && q["frame"].as<std::string>()==(altaz?"delta-source-AltAz":"FK5"),"OOF JINC acquisition/frame differs");
 const auto began=std::chrono::steady_clock::now();const auto cpu=std::clock();auto seconds=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();};
 const auto s=q["spec"];JincDiscreteSpec spec{s["rows"].as<std::size_t>(),s["columns"].as<std::size_t>(),s["phases"].as<std::size_t>(),{},s["scale_rad"].as<double>(),s["a"].as<double>(),s["b"].as<double>(),s["c"].as<double>(),s["rmax"].as<double>()};
 const auto m=s["pixel_matrix_rad"].as<std::vector<double>>();require(m.size()==4,"matrix shape");std::copy(m.begin(),m.end(),spec.pixel_matrix_rad.begin());if(oof)require(spec.phases==1 && spec.a==1.1 && spec.b==3.17 && spec.c==2. && spec.rmax==1.5 && spec.scale_rad==(2.0/1000)/45 && m[1]==0 && m[2]==0 && m[0]==-std::numbers::pi/180/3600 && m[3]==std::numbers::pi/180/3600,"OOF JINC selected parameters differ");JincDiscreteKernel kernel(spec);
 fs::create_directories(out);YAML::Node r;r["schema"]=oof?"oof-a2000-jinc-diagnostic-output-v1":"point-152389-jinc-diagnostic-output-v1";
 if(oof){r["observation"]="obs:"+std::to_string(q["observation"].as<int>())+"_0_1";r["array"]="a2000";r["frame"]=altaz?"delta-source-AltAz":"FK5";}r["request_sha256"]=citlali::utils::sha256_file(request);r["spec"]=s;r["double_digits"]=std::numeric_limits<double>::digits;r["long_double_digits"]=std::numeric_limits<long double>::digits;r["FP_policy"]="IEEE binary64 nearest; no fast-math, no contraction; subnormal/range paths unavailable";
 const auto h=kernel.half_width(),side=2*h+1;std::vector<double> cache;
 for(std::size_t y=0;y<spec.phases;++y)for(std::size_t x=0;x<spec.phases;++x)for(int dy=-int(h);dy<=int(h);++dy)for(int dx=-int(h);dx<=int(h);++dx)cache.push_back(kernel.coefficient(x,y,dx,dy));
 write(out/"cache.f64",cache,r);r["half_width"]=h;r["cache_seconds"]=seconds();
 if(q["mode"].as<std::string>()=="accumulate"){
  require(q["profile"].as<std::string>()==(oof?"OOF-A2000-JINC-NUMERICAL-ADEQUACY-v1":"POINT-152389-JINC-NUMERICAL-ADEQUACY-v1"),"unknown profile");
  require(q["cache_sha256"].as<std::string>()==r["files"]["cache.f64"].as<std::string>(),"production cache changed");
  const auto ek=read<double>(q["coefficient_errors"],cache.size());const auto count=q["samples"].as<std::size_t>();
  // Rows are x,y,t,omega,fs; all rows are explicitly admitted saved occurrences.
  std::vector<JincDiscreteSample> samples(count);
  if(q["geometry_payload"]){
   const auto geometry=read<double>(q["geometry_payload"],count*4),values=read<double>(q["value_payload"],count);
   for(std::size_t i=0;i<count;++i)samples[i]={geometry[4*i],geometry[4*i+1],values[i],geometry[4*i+2],geometry[4*i+3],true};
  }else{
   const auto data=read<double>(q["sample_payload"],count*5);
   for(std::size_t i=0;i<count;++i)samples[i]={data[5*i],data[5*i+1],data[5*i+2],data[5*i+3],data[5*i+4],true};
  }
  r["loaded_seconds"]=seconds();const auto result=kernel.accumulate_certified(samples,ek,q["workers"].as<std::size_t>(),q["frozen_peak"].as<double>());r["accumulated_seconds"]=seconds();
  const auto &v=result.sums;write(out/"N.f64",v.numerator,r);write(out/"C.f64",v.normalization,r);write(out/"Q.f64",v.quadratic,r);write(out/"T.f64",v.coefficient_squared_time,r);write(out/"absolute-C.f64",v.absolute_normalization,r);write(out/"members.u64",v.members,r);write(out/"m.f64",v.algebraic_value,r);write(out/"causes.u8",result.causes,r);write(out/"tolerance-lower.f64",result.tolerance_lower,r);
  for(std::size_t i=0;i<5;++i)write(out/("error-"+std::to_string(i)+".f64"),result.error[i],r);
  r["samples"]=count;r["workers"]=q["workers"];r["reduction"]="contiguous sample blocks, serial pixel sums, ascending worker merge; K=actual pixel memberships+workers+6";
  std::size_t good=0;for(auto c:result.causes)good+=c==0;r["accepted_pixels"]=good;
  // Verify immutable payloads again before accepting publication.
  for(const auto*key:{"sample_payload","geometry_payload","value_payload","coefficient_errors"})if(q[key])require(citlali::utils::sha256_file(q[key]["path"].as<std::string>())==q[key]["sha256"].as<std::string>(),"input changed during accumulation");
 }else require(q["mode"].as<std::string>()=="cache","unknown diagnostic mode");
 r["wall_seconds"]=seconds();r["cpu_seconds"]=double(std::clock()-cpu)/CLOCKS_PER_SEC;YAML::Emitter e;e.SetDoublePrecision(17);e<<r;std::ofstream f(out/"receipt.yaml");f<<e.c_str()<<'\n';f.close();require(bool(f),"receipt write failed");return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
