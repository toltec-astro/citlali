#include <citlali/core/pipeline/rtc_point_run_selection.h>
#include "../../src/citlali/cli/rtc_point_selection_record.h"
#include <citlali/core/utils/sha256.h>
#include <filesystem>
#include <fstream>
#include <iostream>
namespace fs=std::filesystem;
using namespace citlali::pipeline;
template<class T>std::vector<T> plane(const fs::path &p,std::size_t n) {
 if(fs::file_size(p)!=n*sizeof(T))throw std::runtime_error("selector plane shape");
 std::vector<T> v(n);std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));if(!f)throw std::runtime_error("selector plane read");return v;
}
int main(int argc,char **argv)try {
 if(argc!=3 || fs::exists(argv[2]))throw std::runtime_error("usage: rtc_point_select B-network-root NEW-selection.yaml");
 const fs::path root=argv[1],p=root/"source-probe";
 const auto r=YAML::LoadFile((root/"receipt.yaml").string()),s=YAML::LoadFile((p/"receipt.yaml").string());
 if(s["branch"].as<std::string>()!="REF" || !s["r_unchanged"].as<bool>() || !s["input_validity_unchanged"].as<bool>())throw std::runtime_error("saved selector needs original REF");
 for(auto f:s["files"])if(citlali::utils::sha256_file(p/f.first.as<std::string>())!=f.second.as<std::string>())throw std::runtime_error("saved selector input digest");
 const auto rows=s["native_rows"].as<std::size_t>(),nd=s["detectors"].as<std::size_t>(),cells=rows*nd;
 auto x=plane<double>(p/"original-x.f64",cells),v=plane<double>(p/"original-r.f64",cells),t=plane<double>(p/"assigned-time.f64",rows),lon=plane<double>(p/"longitude-deg.f64",cells),lat=plane<double>(p/"latitude-deg.f64",cells);
 auto vx=plane<std::uint8_t>(p/"input-valid.u8",cells),vr=plane<std::uint8_t>(p/"input-r-valid.u8",cells),map=plane<std::uint8_t>(p/"mapping-available.u8",cells);
 auto channels=s["channels"].as<std::vector<int>>();std::vector<RtcPointRun> runs;for(auto q:r["native_runs"])runs.push_back({q[0].as<std::int64_t>(),q[1].as<std::int64_t>()});
 const auto plan=RtcPointRunSelectionPlan::consider(RtcPointRunSelectionEvidence::learn({rows,nd,x,v,t,lon,lat,vx,vr,map,channels,runs,t.size()>1?t[1]-t[0]:0.,r["original_parent"].as<std::string>(),"saved-original-AST-probe:"+citlali::utils::sha256_file(p/"receipt.yaml")}));
 YAML::Emitter serialized;serialized.SetDoublePrecision(17);serialized<<citlali::cli::detail::rtc_point_selection_record(plan);
 if(!serialized.good())throw std::runtime_error("selection serialization");
 std::ofstream f(argv[2]);f<<serialized.c_str()<<'\n';f.close();if(!f)throw std::runtime_error("selection publication");return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
