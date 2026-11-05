#pragma once

#include <citlali/core/utils/sha256.h>
#include <yaml-cpp/yaml.h>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::cli::detail {

// Same little-endian payloads declared by the saved CAL receipt. Slots index
// the scheduled RTC grid. Values (mJy/nominal beam) exist only on supported
// slots; each scheduled slot has its CAL/VAL cause bits.
struct PointCalChannel {
    int channel;
    std::vector<double> values;
    std::vector<std::int64_t> slots;
    std::vector<std::uint16_t> causes;

    template<class T>
    static std::string digest(const std::vector<T> &v) {
        if constexpr (std::endian::native != std::endian::little)
            throw std::runtime_error("POINT CAL transport requires little-endian payloads");
        citlali::utils::Sha256 hash;
        if (!v.empty()) hash.update(reinterpret_cast<const std::uint8_t *>(v.data()),v.size()*sizeof(T));
        return hash.finish();
    }

    YAML::Node declarations() const {
        const auto stem="channel-"+std::to_string(channel);
        YAML::Node files;
        files[stem+"-value.f64"]=digest(values);
        files[stem+"-slot.i64"]=digest(slots);
        files[stem+"-causes.u16"]=digest(causes);
        return files;
    }
};

// A worker fills only its own network slot. All workers join before Python
// can request health. This storage is consumed once, never selected by path.
struct PointCalNetwork {
    std::filesystem::path root,input_path;
    std::string input_sha,receipt_sha;
    std::vector<PointCalChannel> channels;

    void require_binding(const std::filesystem::path &expected_root,const std::filesystem::path &expected_input,
                         const std::string &input,
                         const std::string &receipt,std::size_t detectors) const {
        if (root!=expected_root || input_path!=expected_input || input_sha!=input || receipt_sha!=receipt ||
            channels.size()!=detectors || channels.empty())
            throw std::runtime_error("POINT in-memory CAL network binding differs");
    }

    const PointCalChannel &admit(std::size_t detector,int channel,std::size_t scheduled,
                                 const YAML::Node &declaration) const {
        const auto &product=channels.at(detector);
        if (product.channel!=channel || product.causes.size()!=scheduled ||
            product.values.size()!=product.slots.size() ||
            product.values.size()!=declaration["available"].as<std::size_t>())
            throw std::runtime_error("POINT in-memory CAL detector domain differs");
        const auto expected=product.declarations();
        const auto files=declaration["files"];
        if (!files.IsMap() || files.size()!=expected.size())
            throw std::runtime_error("POINT in-memory CAL payload inventory differs");
        for(const auto &entry:expected) {
            const auto key=entry.first.as<std::string>();
            if(!files[key] || files[key].as<std::string>()!=entry.second.as<std::string>())
                throw std::runtime_error("POINT in-memory CAL payload differs: "+key);
        }
        return product;
    }
};
using PointCalHandoff = std::map<int,PointCalNetwork>;
} // namespace citlali::cli::detail
