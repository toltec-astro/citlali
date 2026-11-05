#pragma once
#include <citlali/core/utils/sha256.h>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>

namespace citlali::cli::detail {
struct MapFileLedger {
    std::map<std::string,std::string> files;
    std::string bind(const std::filesystem::path &path,const std::string &expected={}) {
        const auto digest=citlali::utils::sha256_file(path);
        if (!expected.empty() && digest!=expected)
            throw std::runtime_error("MAP source digest mismatch: "+path.string());
        const auto [it,inserted]=files.emplace(std::filesystem::absolute(path).string(),digest);
        if (!inserted && it->second!=digest) throw std::runtime_error("MAP source changed during read");
        return digest;
    }
    void verify() const {
        for(const auto &[path,digest]:files)
            if(citlali::utils::sha256_file(path)!=digest)
                throw std::runtime_error("MAP source changed during application: "+path);
    }
};
} // namespace citlali::cli::detail
