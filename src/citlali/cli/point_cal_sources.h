#pragma once
#include "map_file_ledger.h"
#include "point_saved_apt.h"
#include <yaml-cpp/yaml.h>

namespace citlali::cli::detail {
// Original-only ledger. A pass request, model or generated product must
// never extend this ledger or change the original observation identity.
class PointCalSources {
public:
    MapFileLedger files;
    PointStageApt apt;

    void bind_domain(const YAML::Node &request,const std::string &config_sha) {
        if (!config_sha_.empty()) throw std::logic_error("POINT CAL domain already bound");
        network_binding_=YAML::Dump(request["networks"]);
        config_binding_=YAML::Dump(request["effective_config"]);
        config_sha_=config_sha;
    }
    void verify_reuse(const YAML::Node &request,const std::string &config_sha) {
        if (config_sha_.empty() || config_sha!=config_sha_ ||
            YAML::Dump(request["effective_config"])!=config_binding_ ||
            YAML::Dump(request["networks"])!=network_binding_)
            throw std::invalid_argument("weighted POINT prepared CAL network/config generation differs");
        apt.verify_retained_generations();
        files.verify();
    }
    void verify_publication(bool reused) {
        if(reused) apt.verify_retained_generations();
        else apt.verify_before_publication();
        files.verify();
    }
private:
    std::string network_binding_,config_binding_,config_sha_;
};
} // namespace citlali::cli::detail
