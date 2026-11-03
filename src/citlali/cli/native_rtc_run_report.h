#pragma once

#include "native_ptc_setup.h"
#include "native_rtc_output_layout.h"
#include <iomanip>
#include <sstream>
#include <string>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

namespace citlali::cli {
inline void log_native_run_summary(const std::string &summary) {
    // The ordinary CLI suppresses info messages on the default console sink.
    // Reuse its run-owned logger; do not reconfigure either sink or its level.
    const auto logger = spdlog::get("citlali_logger");
    if (logger) logger->info("{}", summary);
    else spdlog::info("{}", summary);
}

inline std::string native_terminal_name(DevelopmentTerminal terminal) {
    switch (terminal) {
    case DevelopmentTerminal::rtc_only: return "rtc-only";
    case DevelopmentTerminal::cal: return "cal";
    case DevelopmentTerminal::ptc: return "ptc";
    case DevelopmentTerminal::prepare: return "prepare";
    }
    throw std::invalid_argument("unsupported native terminal");
}

inline YAML::Node native_setup_record(const YAML::Node &cfg, DevelopmentTerminal terminal,
        const NativeRtcOutputLayout::NetworkPaths &output, const NativePtcSetup *ptc) {
    YAML::Node record;
    record["schema"] = "citlali-native-setup-v1";
    record["observation"] = cfg["observation"];
    record["network"] = cfg["network"];
    record["requested_terminal"] = native_terminal_name(terminal);
    record["output"] = output.root.string();
    record["state"] = "setup-only;execution-not-yet-observed";
    if (cfg["effective_config"]) record["effective_config"] = cfg["effective_config"];
    if (ptc) {
        auto p = record["PTC"];
        p["array"] = ptc->array;
        p["method"] = ptc_method_name(ptc->solver.method);
        p["configured_rank"] = ptc->configured_rank;
        if (ptc->requested_rank) p["requested_rank_override"] = ptc->requested_rank;
        else p["requested_rank_override"] = YAML::Null;
        p["effective_rank"] = ptc->solver.rank;
        p["rank_source"] = ptc->requested_rank ? "explicit-request-override" : "bound-effective-config";
        p["iteration_limit"] = ptc->solver.iteration_limit;
        p["relative_rank_tolerance"] = ptc->solver.relative_rank_tolerance;
        p["relative_objective_tolerance"] = ptc->solver.relative_objective_tolerance;
    }
    return record;
}

inline std::string native_setup_summary(const YAML::Node &record) {
    std::ostringstream text;
    text << "Native setup: network=" << record["network"].as<int>()
         << " terminal=" << record["requested_terminal"].as<std::string>()
         << " output=" << std::quoted(record["output"].as<std::string>());
    if (const auto p = record["PTC"]) {
        text << " PTC method=" << p["method"].as<std::string>()
             << " configured_rank=" << p["configured_rank"].as<int>()
             << " effective_rank=" << p["effective_rank"].as<int>()
             << " rank_source=" << p["rank_source"].as<std::string>()
             << " iteration_limit=" << p["iteration_limit"].as<int>();
    }
    text << "; execution not yet observed";
    return text.str();
}

// Read actual products only. In particular, effective rank is a fit request,
// not proof of convergence or output availability. Detailed decisions remain
// in the existing receipts; this is a concise view of those same facts.
inline std::string native_result_summary(int network, const YAML::Node &receipt) {
    std::ostringstream text;
    const auto rtc = receipt["donor_continuity"]["terminal"];
    text << "Native result: network=" << network << " status=" << receipt["status"].as<std::string>()
         << " RTC x_available=" << rtc["x_available"].as<std::size_t>()
         << "/" << rtc["scheduled_slots"].as<std::size_t>()
         << " directly_excluded=" << rtc["directly_excluded"].as<std::size_t>()
         << " replacement_influenced=" << rtc["replacement_influenced"].as<std::size_t>();
    if (const auto cal = receipt["CAL"])
        text << " CAL available=" << cal["available"].as<std::size_t>();
    if (const auto ptc = receipt["PTC"])
        text << " PTC effective_rank=" << ptc["rank"].as<int>()
             << " available=" << ptc["available"].as<std::size_t>()
             << " failed_fits=" << ptc["failed_fits"].as<std::size_t>();
    text << "; scientific qualification unresolved";
    return text.str();
}
} // namespace citlali::cli
