#pragma once

#include <citlali/core/cli/config_loading.h>
#include <citlali/core/cli/successor_development_execution.h>
#include <citlali/core/utils/sha256.h>
#include <set>
#include <stdexcept>

namespace citlali::cli {

template<class RuntimeConfig, class Logger>
int load_and_run_successor_development(const RuntimeConfig &runtime,
                                      const Logger &logger) {
    logger->info("development route: Timestream Successor; legacy fallback disabled");
    const auto loaded = load_merged_yaml_config_files(runtime, logger);
    const YAML::Node config = YAML::Load(loaded.config.to_str());
    if (!config.IsMap() || !config["schema"] ||
        config["schema"].as<std::string>() != "citlali-development-v1") {
        throw std::invalid_argument(
            "successor.input_binding_unavailable: the development default requires "
            "a citlali-development-v1 request with exact native x/r input and plan "
            "bindings; conversion from legacy reduction/data_items configuration "
            "is not implemented. Use --dump_config for the current input contract. "
            "The legacy reduction was not run.");
    }
    const std::set<std::string> keys{"schema", "input", "output", "terminal", "ptc"};
    for (const auto &entry : config) {
        if (!entry.first.IsScalar() || !keys.contains(entry.first.as<std::string>()))
            throw std::invalid_argument("successor.unsupported_configuration: unknown development request field");
    }
    const auto terminal = config["terminal"] ? config["terminal"].as<std::string>() : "ptc";
    if (terminal != "rtc-only" && terminal != "cal" && terminal != "ptc")
        throw std::invalid_argument(
            "successor.endpoint_not_implemented: current development endpoint is PTC/VAL; "
            "MAP/FRUIT execution is unavailable, with no legacy substitution");
    DevelopmentPtcRequest ptc;
    if(config["ptc"]) {
        const auto p=config["ptc"];
        if(terminal!="ptc" || !p.IsMap())throw std::invalid_argument("successor.PTC_request_invalid");
        for(const auto &entry:p) {
            const auto key=entry.first.as<std::string>();
            if(key!="method" && key!="rank")throw std::invalid_argument("successor.PTC_request_unknown_field");
        }
        if(p["method"])ptc.method=p["method"].as<std::string>();
        if(p["rank"]){ptc.rank=p["rank"].as<int>();if(ptc.rank<=0)throw std::invalid_argument("successor.PTC_rank_must_be_positive");}
        if(ptc.method!="observed-als" && ptc.method!="pairwise-covariance")throw std::invalid_argument("successor.PTC_method_unknown");
    }
    const YAML::Node input = config["input"];
    if (!input || !input.IsMap() || input.size() != 2 ||
        !input["path"] || !input["sha256"] || !config["output"])
        throw std::invalid_argument("successor.input_binding_missing: input.path, input.sha256 and output are required");
    const std::filesystem::path path{input["path"].as<std::string>()};
    const std::filesystem::path output{config["output"].as<std::string>()};
    if (!path.is_absolute() || !output.is_absolute())
        throw std::invalid_argument("successor.path_binding: input and output must be absolute paths");
    if (citlali::utils::sha256_file(path) != input["sha256"].as<std::string>())
        throw std::invalid_argument("successor.input_digest_mismatch: input record has changed");
    return run_successor_rtc(path, output, RtcInvocation::development_default, {},
        terminal == "ptc" ? DevelopmentTerminal::ptc : terminal == "cal" ? DevelopmentTerminal::cal : DevelopmentTerminal::rtc_only,ptc);
}

} // namespace citlali::cli
