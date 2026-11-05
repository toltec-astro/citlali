#pragma once
#include <citlali/core/utils/sha256.h>
#include <yaml-cpp/yaml.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace citlali::cli::detail {
struct MapFileLedger {
    std::map<std::string,std::string> files;
    std::string bind(const std::filesystem::path &path,
                     const std::optional<std::string> &expected=std::nullopt) {
        return remember(path,citlali::utils::sha256_file(path),expected);
    }
    std::pair<std::string,std::string> read_bytes(const std::filesystem::path &path,
                          const std::optional<std::string> &expected=std::nullopt) {
        std::ifstream stream(path,std::ios::binary);
        if (!stream) throw std::runtime_error("MAP source open failed: "+path.string());
        std::string bytes;
        std::array<char,64*1024> block;
        while(stream.read(block.data(),block.size()) || stream.gcount()) {
            bytes.append(block.data(),static_cast<std::size_t>(stream.gcount()));
        }
        if (!stream.eof() || stream.bad()) throw std::runtime_error("MAP source read failed: "+path.string());
        auto digest=remember(path,citlali::utils::sha256(bytes),expected);
        return {std::move(digest),std::move(bytes)};
    }
    std::pair<std::string,YAML::Node> read_yaml(const std::filesystem::path &path,
                          const std::optional<std::string> &expected=std::nullopt) {
        const auto [digest,bytes]=read_bytes(path,expected);
        return {digest,YAML::Load(bytes)};
    }
    void verify() const {
        for(const auto &[path,digest]:files)
            if(citlali::utils::sha256_file(path)!=digest)
                throw std::runtime_error("MAP source changed during application: "+path);
    }
private:
    std::string remember(const std::filesystem::path &path,const std::string &digest,
                         const std::optional<std::string> &expected) {
        if (expected && digest!=*expected)
            throw std::runtime_error("MAP source digest mismatch: "+path.string());
        const auto [it,inserted]=files.emplace(std::filesystem::absolute(path).string(),digest);
        if (!inserted && it->second!=digest) throw std::runtime_error("MAP source changed during read");
        return digest;
    }
};
} // namespace citlali::cli::detail
