#pragma once

#include <citlali/core/utils/sha256.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

namespace citlali::cli::detail {

struct PointObservationPass {
    std::filesystem::path development, request, output;
    std::string request_sha;
};

inline bool is_point_digest(const std::string &value) {
    return value.size()==64 && std::all_of(value.begin(),value.end(),[](char c) {
        return (c>='0' && c<='9') || (c>='a' && c<='f');
    });
}

// Only accept the next ordinary MAP request beneath the bound observation.
// Hash the same bytes that are parsed. A command contains no caller-selected path.
inline PointObservationPass read_point_observation_pass(
        const std::filesystem::path &work,unsigned index,unsigned workers,
        const std::string &digest,const std::string &initial_health_sha) {
    if (!is_point_digest(digest)) throw std::invalid_argument("invalid POINT pass command");
    std::ostringstream name;
    if (index) name << "weighted-" << std::setfill('0') << std::setw(2) << index-1;
    else name << "health";
    const auto stage=work/name.str();
    PointObservationPass pass{stage/"development.yaml",work/(name.str()+"-request.json"),stage/"products",{}};
    std::ifstream stream(pass.development,std::ios::binary);
    if (!stream) throw std::runtime_error("missing POINT pass descriptor: "+pass.development.string());
    const std::string bytes{std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
    if (stream.bad() || citlali::utils::sha256(bytes)!=digest)
        throw std::runtime_error("POINT pass descriptor digest differs");
    const auto config=YAML::Load(bytes);
    const auto bound=config["input"];
    if (!config.IsMap() || config.size()!=5 || config["schema"].as<std::string>()!="citlali-development-v1" ||
        config["terminal"].as<std::string>()!="map" || config["network_workers"].as<unsigned>()!=workers ||
        config["output"].as<std::string>()!=pass.output.string() || !bound.IsMap() || bound.size()!=2 ||
        bound["path"].as<std::string>()!=pass.request.string())
        throw std::invalid_argument("POINT pass descriptor route or binding differs");
    pass.request_sha=bound["sha256"].as<std::string>();
    if (!is_point_digest(pass.request_sha) || (!index && pass.request_sha!=initial_health_sha))
        throw std::invalid_argument("POINT pass request generation differs");
    return pass;
}

} // namespace citlali::cli::detail
