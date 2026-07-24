#pragma once

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#if defined(__linux__)
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace citlali::utils {

struct ProcessResourceSnapshot {
    bool available = false;
    long long rss_kib = -1;
    long long peak_rss_kib = -1;
    long long virtual_kib = -1;
    long long threads = -1;
    long long open_fds = -1;
    long long mappings = 0;
    long long file_mappings = 0;
    unsigned long long executable_inode = 0;
    long long executable_size = -1;
    long long executable_mtime = -1;
    std::string executable_path;
};

inline long long parse_proc_status_value(const std::string &line,
                                         const std::string &key) {
    if (line.compare(0, key.size(), key) != 0) {
        return -1;
    }
    std::istringstream value(line.substr(key.size()));
    long long parsed = -1;
    value >> parsed;
    return value.fail() ? -1 : parsed;
}

inline ProcessResourceSnapshot process_resource_snapshot() {
    ProcessResourceSnapshot result;
#if defined(__linux__)
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (const auto value = parse_proc_status_value(line, "VmRSS:");
            value >= 0) {
            result.rss_kib = value;
        }
        else if (const auto value = parse_proc_status_value(line, "VmHWM:");
                 value >= 0) {
            result.peak_rss_kib = value;
        }
        else if (const auto value = parse_proc_status_value(line, "VmSize:");
                 value >= 0) {
            result.virtual_kib = value;
        }
        else if (const auto value = parse_proc_status_value(line, "Threads:");
                 value >= 0) {
            result.threads = value;
        }
    }

    if (auto *directory = ::opendir("/proc/self/fd"); directory != nullptr) {
        long long count = 0;
        while (const auto *entry = ::readdir(directory)) {
            if (entry->d_name[0] != '.') {
                ++count;
            }
        }
        ::closedir(directory);
        // The directory descriptor used for enumeration is included above.
        result.open_fds = count > 0 ? count - 1 : 0;
    }

    std::ifstream maps("/proc/self/maps");
    while (std::getline(maps, line)) {
        ++result.mappings;
        const auto path_start = line.find('/');
        if (path_start != std::string::npos) {
            ++result.file_mappings;
        }
    }

    char path[4096];
    const auto path_length =
        ::readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (path_length >= 0) {
        path[path_length] = '\0';
        result.executable_path = path;
    }

    struct stat executable_stat {};
    if (::stat("/proc/self/exe", &executable_stat) == 0) {
        result.executable_inode =
            static_cast<unsigned long long>(executable_stat.st_ino);
        result.executable_size =
            static_cast<long long>(executable_stat.st_size);
        result.executable_mtime =
            static_cast<long long>(executable_stat.st_mtime);
    }
    result.available = result.rss_kib >= 0;
#endif
    return result;
}

inline bool process_resource_diagnostics_enabled() {
    const auto *value = std::getenv("CITLALI_PROCESS_RESOURCE_DIAGNOSTICS");
    if (value == nullptr || *value == '\0') {
        return false;
    }
    return std::string(value) != "0" && std::string(value) != "no" &&
           std::string(value) != "off";
}

template <class Logger>
void log_process_resource_snapshot(const Logger &logger,
                                   const std::string &stage) {
    if (!process_resource_diagnostics_enabled()) {
        return;
    }
    const auto snapshot = process_resource_snapshot();
    if (!snapshot.available) {
        logger->debug("process_resources stage='{}' unavailable", stage);
        return;
    }
    constexpr double kib_per_mib = 1024.0;
    logger->debug(
        "process_resources stage='{}' rss_mib={:.3f} peak_rss_mib={:.3f} "
        "virtual_mib={:.3f} threads={} open_fds={} mappings={} "
        "file_mappings={} executable_inode={} executable_size={} "
        "executable_mtime={} executable_path='{}'",
        stage, snapshot.rss_kib / kib_per_mib,
        snapshot.peak_rss_kib / kib_per_mib,
        snapshot.virtual_kib / kib_per_mib, snapshot.threads,
        snapshot.open_fds, snapshot.mappings, snapshot.file_mappings,
        snapshot.executable_inode, snapshot.executable_size,
        snapshot.executable_mtime, snapshot.executable_path);
}

}  // namespace citlali::utils
