// Private RTC publication helpers. Keep the existing YAML data model while
// streaming large sequences; do not retain a YAML node graph per window.
namespace {
void rtc_yaml_node(std::ostream &out,const YAML::Node &node,std::size_t indent=0) {
    YAML::Emitter emitted;emitted.SetDoublePrecision(17);emitted<<node;
    require(emitted.good(),"RTC receipt serialization failed");
    std::istringstream lines(emitted.c_str());std::string line;
    while(std::getline(lines,line))out<<std::string(indent,' ')<<line<<'\n';
}
void rtc_yaml_file(std::ostream &out,const fs::path &path,std::size_t indent) {
    std::ifstream input(path);require(bool(input),"RTC receipt fragment missing");
    std::string line;while(std::getline(input,line))out<<std::string(indent,' ')<<line<<'\n';
    require(input.eof() && bool(out),"RTC receipt fragment publication failed");
}
void rtc_yaml_map(const fs::path &path,const YAML::Node &node,
    const std::vector<std::pair<std::string,fs::path>> &sections) {
    std::ofstream out(path);
    for(const auto &entry:node) {
        const auto key=entry.first.as<std::string>();
        if(std::any_of(sections.begin(),sections.end(),[&](const auto &s){return s.first==key;}))continue;
        YAML::Node one;one[key]=entry.second;rtc_yaml_node(out,one);
    }
    for(const auto &[key,file]:sections) {out<<key<<":\n";rtc_yaml_file(out,file,2);}
    out.close();require(bool(out),"required RTC receipt output failed: "+path.string());
}
void rtc_spectrum_record(std::ostream &out,const RtcNativeSpectrum &s,int channel,bool conditioned) {
    out<<std::setprecision(17)<<"- channel: "<<channel<<"\n  coordinate: "<<static_cast<int>(s.coordinate)
       <<"\n  available: "<<(s.available()?"true":"false")<<"\n  cause: "<<static_cast<int>(s.cause)
       <<"\n  bins: "<<s.psd.size()<<'\n';
    if(s.windows.empty())return; // Preserve the existing absent-key convention.
    out<<"  windows:\n";
    for(const auto &w:s.windows) {
        if(!conditioned)out<<"    - ["<<w.rows.first<<", "<<w.rows.past_last<<"]\n";
        else out<<"    - {rows: ["<<w.rows.first<<", "<<w.rows.past_last
            <<"], representative_replacements: "<<w.representative_replacements
            <<", replacement_influenced_samples: "<<w.replacement_influenced_samples
            <<", unrepaired_influenced_samples: "<<w.unrepaired_influenced_samples
            <<", representative_exclusions: "<<w.representative_exclusions<<"}\n";
    }
}
}
