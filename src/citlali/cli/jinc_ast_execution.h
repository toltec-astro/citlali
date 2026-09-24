// Diagnostic export of native same-schedule AST coordinates only. Never Learn,
// Apply, candidate generation or map accumulation. Reuses saved input adapters.
namespace {
int execute_jinc_ast(const fs::path &input,const fs::path &output){
 const auto began=std::chrono::steady_clock::now();require(!fs::exists(output),"fresh JINC AST output required");
 const auto request=YAML::LoadFile(input.string());require(request["schema"].as<std::string>()=="point-152389-jinc-ast-v1","unknown AST diagnostic");
 MapFileLedger ledger;const auto mrp=checked_file(request["map_request"]);ledger.bind(mrp);const auto mr=YAML::LoadFile(mrp.string());
 const auto pp=checked_file(request["parent_plan"]);ledger.bind(pp);const auto plan=YAML::LoadFile(pp.string());const auto expected=YAML::LoadFile(checked_file(request["archived_input_manifest"]).string());
 const int ai=request["array_index"].as<int>();require(ai>=0&&ai<3&&plan["observation"].as<std::string>()=="obs:152389_0_2","JINC AST observation/array");
 const auto header=request["WCS"];const double cx=header["CRPIX1"].as<double>(),cy=header["CRPIX2"].as<double>(),dx=header["CDELT1"].as<double>(),dy=header["CDELT2"].as<double>();
 require(header["RADESYS"].as<std::string>()=="FK5"&&header["EQUINOX"].as<double>()==2000,"JINC AST frame");
 for(const auto *key:{"CRPIX1","CRPIX2","CDELT1","CDELT2","CRVAL1","CRVAL2"})require(header[key].as<double>()==plan["WCS"][key].as<double>(),"JINC full target WCS differs from original plan");
 std::size_t selected_networks=0;
 fs::create_directories(output);YAML::Node result;result["schema"]="point-152389-native-ast-v1";result["source_revision"]=std::string(CITLALI_GIT_VERSION);result["request_sha256"]=citlali::utils::sha256_file(input);result["WCS"]=header;
 for(const auto &nr:mr["networks"]){const fs::path root=nr["root"].as<std::string>();const auto cal=YAML::LoadFile((root/"donor-continuity/cal/receipt.yaml").string());if(cal["detectors"][0]["array"].as<int>()!=ai)continue;
  ++selected_networks;auto n=load_saved_map_network(nr,ledger);require(n.obsnum==152389&&n.array==ai,"JINC AST wrong array");
  require(n.observation.ra0*180/M_PI==header["CRVAL1"].as<double>()||std::abs(n.observation.ra0*180/M_PI-header["CRVAL1"].as<double>())<1e-10,"JINC AST reference differs");
  const double cadence=n.learn["cadence_interval_seconds"].as<double>(),frequency=1/(cadence*n.factor);require(std::isfinite(frequency)&&frequency>0,"JINC processed cadence unavailable");
  for(std::size_t si=0;si<n.ptc["segments"].size();++si){const auto old=load_map_segment(n,si,ledger);const auto s=n.ptc["segments"][si];const auto nt=s["scheduled_times"].as<std::size_t>(),no=s["output_detectors"].as<std::size_t>();const auto detectors=s["output_detector_indices"].as<std::vector<std::size_t>>();
   Eigen::VectorXd times(nt);for(std::size_t t=0;t<nt;++t)times[t]=n.native_times[n.first+old->identity().slots[t]*n.factor];
   const auto schedule=evaluate_ast_rtc_schedule(*n.observation.trajectory,*n.observation.offsets,n.observation.ra0,n.observation.dec0,times);
   std::vector<double> coords(nt*no*4);std::vector<std::uint16_t> causes(nt*no);
   for(std::size_t d=0;d<no;++d){const auto [gx,gy]=n.geometry[detectors[d]];const auto v=schedule.detector_tangent_deg(gx,gy);
    for(std::size_t t=0;t<nt;++t){const auto i=t*no+d;coords[4*i]=v(t,0)/dx+cx;coords[4*i+1]=v(t,1)/dy+cy;coords[4*i+2]=v(t,0);coords[4*i+3]=v(t,1);causes[i]=n.cal_causes[detectors[d]][old->identity().slots[t]];}
   }
   const auto stem="network-"+std::to_string(n.network)+"-segment-"+std::to_string(si);const auto cp=output/(stem+".f64"),vp=output/(stem+"-cal.u16");map_write_array<double>(cp,coords);map_write_array<std::uint16_t>(vp,causes);
   YAML::Node rec;rec["name"]=stem;rec["source_segment"]=s;rec["detector_occurrences"]=old->identity().detectors;rec["slots"]=old->identity().slots;rec["native_phase"]=n.first;rec["factor"]=n.factor;rec["cadence_interval_seconds"]=cadence;rec["frequency_hz"]=frequency;rec["frequency_source"]="saved native cadence_interval_seconds and realized RTC factor; fixed binary64 processed sample frequency";rec["coordinates"]["path"]=cp.string();rec["coordinates"]["sha256"]=citlali::utils::sha256_file(cp);rec["CAL_causes"]["path"]=vp.string();rec["CAL_causes"]["sha256"]=citlali::utils::sha256_file(vp);result["segments"].push_back(rec);
  }
 }
 require(selected_networks==plan["sources"].size(),"JINC AST selected network inventory differs");
 // Old lineage supplies expectations; newly computed hashes alone cannot admit
 // an input. Parent plan itself is an explicit current request binding.
 for(const auto &[p,h]:ledger.files)if(p!=fs::absolute(pp).string())require(expected[p]&&expected[p].as<std::string>()==h,"AST input absent or changed in archived ledger: "+p);
 ledger.verify();YAML::Node refs;for(const auto &[p,h]:ledger.files)refs[p]=h;write_yaml(output/"input-manifest.yaml",refs);
 result["columns"]="row-major scheduled-slot/output-detector: FITS x,y,tangent east deg,north deg";result["wall_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();write_yaml(output/"receipt.yaml",result);return 0;
}
}
