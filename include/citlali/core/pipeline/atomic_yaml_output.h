#pragma once

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <system_error>

namespace citlali::pipeline {

inline void write_yaml_file_atomic(const std::filesystem::path &output_path,
                                   const YAML::Node &node) {
    auto temporary_path = output_path;
    temporary_path += ".tmp";

    try {
        std::ofstream stream(temporary_path, std::ios::out | std::ios::trunc);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        stream << node;
        stream.flush();
        stream.close();
        std::filesystem::rename(temporary_path, output_path);
    }
    catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary_path, ignored);
        throw;
    }
}

}  // namespace citlali::pipeline
