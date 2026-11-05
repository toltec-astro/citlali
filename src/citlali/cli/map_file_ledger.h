#pragma once
#include <citlali/core/utils/sha256.h>
#include <yaml-cpp/yaml.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace citlali::cli::detail {
struct FileHashIo {
    std::uintmax_t files=0,bytes=0;
    void add(std::uintmax_t size) { ++files; bytes+=size; }
};
struct MapFileLedger {
    std::map<std::string,std::string> files;
    FileHashIo io,channel_io;
    void bind_channel(const std::filesystem::path &path,const std::string &expected) {
        channel_paths_.insert(std::filesystem::absolute(path).string());
        bind(path,expected);
    }
    std::string bind(const std::filesystem::path &path,
                     const std::optional<std::string> &expected=std::nullopt) {
        return remember(path,hash(path),expected);
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
        io.add(bytes.size());
        return {std::move(digest),std::move(bytes)};
    }
    std::pair<std::string,YAML::Node> read_yaml(const std::filesystem::path &path,
                          const std::optional<std::string> &expected=std::nullopt) {
        const auto [digest,bytes]=read_bytes(path,expected);
        return {digest,YAML::Load(bytes)};
    }
    void verify() {
        for(const auto &[path,digest]:files)
            if(hash(path)!=digest)
                throw std::runtime_error("MAP source changed during application: "+path);
    }
private:
    std::set<std::string> channel_paths_;
    std::string hash(const std::filesystem::path &path) {
        std::uintmax_t size{};
        auto digest=citlali::utils::sha256_file_with_size(path,size);
        io.add(size);
        if(!channel_paths_.empty() && channel_paths_.contains(std::filesystem::absolute(path).string()))
            channel_io.add(size);
        return digest;
    }
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
