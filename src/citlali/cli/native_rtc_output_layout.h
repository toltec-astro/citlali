#pragma once

#include <citlali/core/cli/successor_development_execution.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::cli {

// Application-lifetime ownership, not scientific state. All structural paths
// are fixed and exclusively created before any input preparation or workers.
// Failure deliberately retains the claimed tree; destruction never deletes it.
class NativeRtcOutputLayout {
public:
    struct TreatmentPaths {
        std::filesystem::path root, cal, ptc;
    };
    struct NetworkPaths {
        std::filesystem::path root;
        std::vector<TreatmentPaths> treatments;
        const TreatmentPaths &treatment(bool continuity) const {
            return continuity ? treatments.back() : treatments.front();
        }
    };

    NativeRtcOutputLayout(const std::filesystem::path &output,
                          const std::vector<int> &networks,
                          bool automatic, bool comparison,
                          DevelopmentTerminal terminal) {
        if (output.empty()) throw std::invalid_argument("empty native output path");
        // Keep symlink/.. interpretation with the filesystem. Lexical
        // normalization can redirect the claim to a different physical parent.
        root_ = std::filesystem::absolute(output);
        if (root_.filename().empty() && root_ != root_.root_path())
            root_ = root_.parent_path();
        if (root_.filename().empty())
            throw std::invalid_argument("native output must name a new directory");
        const auto add_network = [&](const std::filesystem::path &root) {
            NetworkPaths paths{root, {}};
            const auto add_treatment = [&](const std::filesystem::path &arm) {
                paths.treatments.push_back({arm, arm / "cal", arm / "ptc"});
            };
            if (automatic) {
                if (comparison) add_treatment(root / "exclusion-control");
                add_treatment(root / "donor-continuity");
            } else add_treatment(root);
            networks_.push_back(std::move(paths));
        };
        if (networks.empty()) add_network(root_);
        else {
            int previous = -1;
            for (int network : networks) {
                if (network < 0 || network > 12 || network <= previous)
                    throw std::invalid_argument("native output network inventory invalid");
                previous = network;
                add_network(root_ / ("network" + std::to_string(network)));
            }
        }
        std::filesystem::create_directories(root_.parent_path());
        claim(root_);
        for (const auto &network : networks_) {
            if (network.root != root_) claim(network.root);
            if (terminal == DevelopmentTerminal::prepare) continue;
            for (const auto &arm : network.treatments) {
                if (arm.root != network.root) claim(arm.root);
                if (terminal == DevelopmentTerminal::cal || terminal == DevelopmentTerminal::ptc)
                    claim(arm.cal);
                if (terminal == DevelopmentTerminal::ptc) claim(arm.ptc);
            }
        }
    }

    NativeRtcOutputLayout(const NativeRtcOutputLayout &) = delete;
    NativeRtcOutputLayout &operator=(const NativeRtcOutputLayout &) = delete;
    const std::filesystem::path &root() const noexcept { return root_; }
    const NetworkPaths &network(std::size_t index) const { return networks_.at(index); }

private:
    static void claim(const std::filesystem::path &path) {
        // create_directory is the claim; existing directories are never reused.
        if (!std::filesystem::create_directory(path))
            throw std::runtime_error("native output already claimed; preserve previous output: " + path.string());
    }
    std::filesystem::path root_;
    std::vector<NetworkPaths> networks_;
};

} // namespace citlali::cli
