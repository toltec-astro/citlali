#pragma once

#include <yaml-cpp/yaml.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace citlali::pipeline {

namespace atomic_output {

inline void synchronize_file(const std::filesystem::path &path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        throw std::ios_base::failure(
            "unable to open staged output for synchronization " +
            path.string() + ": " + std::strerror(errno));
    }
    if (::fsync(descriptor) != 0) {
        const auto detail = std::string{std::strerror(errno)};
        ::close(descriptor);
        throw std::ios_base::failure(
            "unable to synchronize staged output " + path.string() +
            ": " + detail);
    }
    if (::close(descriptor) != 0) {
        throw std::ios_base::failure(
            "unable to close synchronized staged output " + path.string() +
            ": " + std::strerror(errno));
    }
}

inline void synchronize_parent_directory(
    const std::filesystem::path &path) {
    const auto parent = path.has_parent_path()
        ? path.parent_path() : std::filesystem::path{"."};
    const int descriptor = ::open(parent.c_str(), O_RDONLY);
    if (descriptor < 0) {
        throw std::ios_base::failure(
            "unable to open output directory for synchronization " +
            parent.string() + ": " + std::strerror(errno));
    }
    if (::fsync(descriptor) != 0) {
        const auto detail = std::string{std::strerror(errno)};
        ::close(descriptor);
        throw std::ios_base::failure(
            "unable to synchronize output directory " + parent.string() +
            ": " + detail);
    }
    if (::close(descriptor) != 0) {
        throw std::ios_base::failure(
            "unable to close synchronized output directory " +
            parent.string() + ": " + std::strerror(errno));
    }
}

inline void replace_atomically(const std::filesystem::path &staged_path,
                               const std::filesystem::path &output_path) {
    auto backup_path = output_path;
    backup_path += ".replace-backup";
    std::error_code error;
    std::filesystem::remove(backup_path, error);
    error.clear();
    const bool had_existing_output = std::filesystem::exists(output_path);
    if (had_existing_output) {
        std::filesystem::create_hard_link(
            output_path, backup_path, error);
        if (error) {
            error.clear();
            std::filesystem::copy_file(
                output_path, backup_path,
                std::filesystem::copy_options::overwrite_existing, error);
            if (error) {
                throw std::ios_base::failure(
                    "unable to preserve existing required output " +
                    output_path.string() + " before atomic replacement: " +
                    error.message());
            }
            synchronize_file(backup_path);
        }
    }

    std::filesystem::rename(staged_path, output_path, error);
    if (error) {
        const auto detail = error.message();
        std::error_code ignored;
        std::filesystem::remove(backup_path, ignored);
        throw std::ios_base::failure(
            "unable to atomically replace required output " +
            output_path.string() + " from " + staged_path.string() +
            ": " + detail);
    }
    try {
        synchronize_parent_directory(output_path);
    }
    catch (...) {
        std::error_code restore_error;
        if (had_existing_output) {
            std::filesystem::rename(
                backup_path, output_path, restore_error);
        }
        else {
            std::filesystem::remove(output_path, restore_error);
        }
        try {
            synchronize_parent_directory(output_path);
        }
        catch (...) {
        }
        throw;
    }
    std::filesystem::remove(backup_path, error);
}

inline bool yaml_nodes_equivalent(const YAML::Node &expected,
                                  const YAML::Node &actual) {
    if (expected.Type() != actual.Type()) {
        return false;
    }
    switch (expected.Type()) {
        case YAML::NodeType::Undefined:
        case YAML::NodeType::Null:
            return true;
        case YAML::NodeType::Scalar:
            return expected.Scalar() == actual.Scalar();
        case YAML::NodeType::Sequence:
            if (expected.size() != actual.size()) {
                return false;
            }
            for (std::size_t index = 0; index < expected.size(); ++index) {
                if (!yaml_nodes_equivalent(expected[index], actual[index])) {
                    return false;
                }
            }
            return true;
        case YAML::NodeType::Map:
            if (expected.size() != actual.size()) {
                return false;
            }
            for (const auto &entry : expected) {
                if (!entry.first.IsScalar()) {
                    return false;
                }
                const auto value = actual[entry.first.Scalar()];
                if (!value || !yaml_nodes_equivalent(entry.second, value)) {
                    return false;
                }
            }
            return true;
    }
    return false;
}

}  // namespace atomic_output

template <class Validator>
inline void write_yaml_file_atomic_validated(
    const std::filesystem::path &output_path, const YAML::Node &node,
    Validator &&validator) {
    auto temporary_path = output_path;
    temporary_path += ".tmp";
    std::error_code ignored;
    std::filesystem::remove(temporary_path, ignored);

    try {
        std::ofstream stream(temporary_path, std::ios::out | std::ios::trunc);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        stream << node;
        stream.flush();
        atomic_output::synchronize_file(temporary_path);
        stream.close();

        const auto reopened = YAML::LoadFile(temporary_path.string());
        if (!atomic_output::yaml_nodes_equivalent(node, reopened)) {
            throw std::runtime_error(
                "reopened YAML content does not match the staged document");
        }
        std::invoke(std::forward<Validator>(validator), reopened);
        atomic_output::replace_atomically(temporary_path, output_path);
    }
    catch (...) {
        std::filesystem::remove(temporary_path, ignored);
        throw;
    }
}

inline void write_yaml_file_atomic(const std::filesystem::path &output_path,
                                   const YAML::Node &node) {
    write_yaml_file_atomic_validated(
        output_path, node, [](const YAML::Node &) {});
}

}  // namespace citlali::pipeline
