#pragma once

#include <citlali/core/pipeline/csv_output.h>

#include <netcdf>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *learning_housekeeping_schema_version =
    "citlali-learning-housekeeping-qa-v1";

struct LearningHousekeepingChannel {
    std::string group;
    std::string id;
    std::string name;
    std::string time_variable;
    std::string value_variable;
    std::string unit;
};

struct LearningHousekeepingMatch {
    std::string status;
    double sample_time_unix_sec = std::numeric_limits<double>::quiet_NaN();
    double sample_offset_sec = std::numeric_limits<double>::quiet_NaN();
    double sample_age_sec = std::numeric_limits<double>::quiet_NaN();
    double value = std::numeric_limits<double>::quiet_NaN();
    double previous_sample_time_unix_sec =
        std::numeric_limits<double>::quiet_NaN();
    double previous_value = std::numeric_limits<double>::quiet_NaN();
    double delta_from_previous = std::numeric_limits<double>::quiet_NaN();
    double next_sample_time_unix_sec =
        std::numeric_limits<double>::quiet_NaN();
    double next_value = std::numeric_limits<double>::quiet_NaN();
    double delta_to_next = std::numeric_limits<double>::quiet_NaN();
    double local_excursion = std::numeric_limits<double>::quiet_NaN();
};

inline const std::vector<LearningHousekeepingChannel> &
learning_housekeeping_channels() {
    static const std::vector<LearningHousekeepingChannel> channels{
        {"toltec_thermometry", "Temperature4", "4K central busbar",
         "Data.ToltecThermetry.Time4",
         "Data.ToltecThermetry.Temperature4", "K"},
        {"toltec_thermometry", "Temperature5", "1.1_1_top",
         "Data.ToltecThermetry.Time5",
         "Data.ToltecThermetry.Temperature5", "K"},
        {"toltec_thermometry", "Temperature6", "2.0_1_foot",
         "Data.ToltecThermetry.Time6",
         "Data.ToltecThermetry.Temperature6", "K"},
        {"toltec_thermometry", "Temperature9", "1.1_0.1_top",
         "Data.ToltecThermetry.Time9",
         "Data.ToltecThermetry.Temperature9", "K"},
        {"toltec_thermometry", "Temperature10", "2.0_0.1_top",
         "Data.ToltecThermetry.Time10",
         "Data.ToltecThermetry.Temperature10", "K"},
        {"toltec_thermometry", "Temperature11", "1.4_0.1_top",
         "Data.ToltecThermetry.Time11",
         "Data.ToltecThermetry.Temperature11", "K"},
        {"toltec_thermometry", "Temperature13", "LS_front",
         "Data.ToltecThermetry.Time13",
         "Data.ToltecThermetry.Temperature13", "K"},
        {"dilution_fridge", "T1", "PT2 Head",
         "Data.ToltecDilutionFridge.SampleTime",
         "Data.ToltecDilutionFridge.StsDevT1TempSigTemp", "K"},
        {"dilution_fridge", "T2", "PT2 Plate",
         "Data.ToltecDilutionFridge.SampleTime",
         "Data.ToltecDilutionFridge.StsDevT2TempSigTemp", "K"},
        {"dilution_fridge", "T3", "Still Plate",
         "Data.ToltecDilutionFridge.SampleTime",
         "Data.ToltecDilutionFridge.StsDevT3TempSigTemp", "K"},
        {"dilution_fridge", "T4", "Cold Plate",
         "Data.ToltecDilutionFridge.SampleTime",
         "Data.ToltecDilutionFridge.StsDevT4TempSigTemp", "K"},
        {"dilution_fridge", "T8", "MC Plate",
         "Data.ToltecDilutionFridge.SampleTime",
         "Data.ToltecDilutionFridge.StsDevT8TempSigTemp", "K"},
        {"dilution_fridge", "T12", "MC Bar",
         "Data.ToltecDilutionFridge.SampleTime",
         "Data.ToltecDilutionFridge.StsDevT12TempSigTemp", "K"},
    };
    return channels;
}

inline std::vector<std::string> learning_housekeeping_csv_header() {
    return {
        "schema_version", "iteration", "obsnum", "scan_zero_based", "nw",
        "array", "score", "pathology_reason", "event_time_unix_sec",
        "event_time_basis", "housekeeping_filepath", "channel_group",
        "channel_id", "channel_name", "unit", "status",
        "sample_time_unix_sec", "sample_offset_sec", "sample_age_sec",
        "value", "previous_sample_time_unix_sec", "previous_value",
        "delta_from_previous", "next_sample_time_unix_sec", "next_value",
        "delta_to_next", "local_excursion"};
}

inline std::string learning_housekeeping_number(double value) {
    if (!std::isfinite(value)) {
        return {};
    }
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

inline std::vector<double> read_learning_housekeeping_variable(
    netCDF::NcFile &file, const std::string &name) {
    const auto variable = file.getVar(name);
    if (variable.isNull() || variable.getDimCount() != 1) {
        return {};
    }
    const auto size = variable.getDim(0).getSize();
    std::vector<double> values(size);
    if (size > 0) {
        variable.getVar(values.data());
    }
    return values;
}

inline bool valid_learning_housekeeping_value(double value) {
    // Current TolTEC HK files use -1 for an unavailable thermometer.  A
    // physical temperature in kelvin must be finite and strictly positive.
    return std::isfinite(value) && value > 0.0;
}

inline LearningHousekeepingMatch match_learning_housekeeping_sample(
    const std::vector<double> &times, const std::vector<double> &values,
    double event_time_unix_sec) {
    LearningHousekeepingMatch match;
    if (!std::isfinite(event_time_unix_sec) || event_time_unix_sec <= 0.0) {
        match.status = "event_time_unavailable";
        return match;
    }
    if (times.empty() || values.empty()) {
        match.status = "channel_missing";
        return match;
    }
    if (times.size() != values.size()) {
        match.status = "channel_length_mismatch";
        return match;
    }

    std::vector<std::size_t> valid_time_indices;
    valid_time_indices.reserve(times.size());
    for (std::size_t i = 0; i < times.size(); ++i) {
        if (std::isfinite(times[i]) && times[i] > 0.0) {
            valid_time_indices.push_back(i);
        }
    }
    if (valid_time_indices.empty()) {
        match.status = "no_valid_timestamps";
        return match;
    }
    std::sort(
        valid_time_indices.begin(), valid_time_indices.end(),
        [&](std::size_t lhs, std::size_t rhs) {
            return times[lhs] < times[rhs];
        });
    if (event_time_unix_sec < times[valid_time_indices.front()] ||
        event_time_unix_sec > times[valid_time_indices.back()]) {
        match.status = "event_outside_housekeeping_range";
        return match;
    }

    const auto nearest_it = std::min_element(
        valid_time_indices.begin(), valid_time_indices.end(),
        [&](std::size_t lhs, std::size_t rhs) {
            return std::abs(times[lhs] - event_time_unix_sec) <
                   std::abs(times[rhs] - event_time_unix_sec);
        });
    const auto nearest_position = static_cast<std::size_t>(
        std::distance(valid_time_indices.begin(), nearest_it));
    const auto nearest = *nearest_it;
    match.sample_time_unix_sec = times[nearest];
    match.sample_offset_sec = times[nearest] - event_time_unix_sec;
    match.sample_age_sec = std::abs(match.sample_offset_sec);
    if (!valid_learning_housekeeping_value(values[nearest])) {
        match.status = "nearest_value_invalid_or_unavailable";
        return match;
    }
    match.status = "matched";
    match.value = values[nearest];

    if (nearest_position > 0) {
        const auto previous = valid_time_indices[nearest_position - 1];
        if (valid_learning_housekeeping_value(values[previous])) {
            match.previous_sample_time_unix_sec = times[previous];
            match.previous_value = values[previous];
        }
    }
    if (std::isfinite(match.previous_value)) {
        match.delta_from_previous = match.value - match.previous_value;
    }
    if (nearest_position + 1 < valid_time_indices.size()) {
        const auto next = valid_time_indices[nearest_position + 1];
        if (valid_learning_housekeeping_value(values[next])) {
            match.next_sample_time_unix_sec = times[next];
            match.next_value = values[next];
        }
    }
    if (std::isfinite(match.next_value)) {
        match.delta_to_next = match.next_value - match.value;
    }
    if (std::isfinite(match.previous_value) &&
        std::isfinite(match.next_value)) {
        match.local_excursion =
            match.value - 0.5 * (match.previous_value + match.next_value);
    }
    return match;
}

inline std::string learning_housekeeping_filename(
    const std::string &reduction_directory, int iteration) {
    return (std::filesystem::path(reduction_directory) /
            ("learning_housekeeping_iter_" + std::to_string(iteration) +
             ".csv"))
        .string();
}

template <class RawObs>
std::vector<std::string> find_learning_housekeeping_files(
    const RawObs &rawobs, const std::string &obsnum) {
    std::vector<std::string> matches;
    for (const auto &item : rawobs.data_items()) {
        if (item.interface() == "toltec_hk" ||
            item.interface() == "housekeeping") {
            matches.push_back(item.filepath());
        }
    }
    if (matches.empty()) {
        std::vector<std::filesystem::path> directories;
        for (const auto &item_ref : rawobs.kidsdata()) {
            directories.push_back(
                std::filesystem::path(item_ref.get().filepath()).parent_path());
        }
        std::sort(directories.begin(), directories.end());
        directories.erase(std::unique(directories.begin(), directories.end()),
                          directories.end());
        const auto obs_token = "_" + obsnum + "_";
        for (const auto &directory : directories) {
            if (directory.empty() || !std::filesystem::is_directory(directory)) {
                continue;
            }
            for (const auto &entry : std::filesystem::directory_iterator(directory)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                const auto filename = entry.path().filename().string();
                if (filename.rfind("toltec_hk_", 0) == 0 &&
                    filename.find(obs_token) != std::string::npos &&
                    entry.path().extension() == ".nc") {
                    matches.push_back(entry.path().string());
                }
            }
        }
    }
    std::sort(matches.begin(), matches.end());
    matches.erase(std::unique(matches.begin(), matches.end()), matches.end());
    return matches;
}

template <class Engine, class RawObs, class Logger>
void write_learning_housekeeping_qa_if_available(
    Engine &engine, const RawObs &rawobs, bool first_observation,
    const Logger &logger) {
    if constexpr (!(requires {
                        engine.learning;
                        engine.output_paths.redu_dir_name;
                        engine.iteration.fruit_iter;
                        engine.observation_identity.obsnum;
                        rawobs.data_items();
                        rawobs.kidsdata();
                    })) {
        return;
    }
    else {
        if (!engine.learning.is_enabled() ||
            !engine.learning.diagnostics_enabled()) {
            return;
        }
        if (engine.output_paths.redu_dir_name.empty()) {
            throw std::runtime_error(
                "learning housekeeping QA is enabled but the reduction output directory is empty");
        }

        const int iteration = engine.iteration.fruit_iter;
        const auto output_path = learning_housekeeping_filename(
            engine.output_paths.redu_dir_name, iteration);
        std::ofstream out(
            output_path,
            first_observation ? std::ios::out : (std::ios::out | std::ios::app));
        if (!out) {
            throw std::runtime_error(
                "failed to open required learning housekeeping QA output " +
                output_path);
        }
        if (first_observation) {
            write_csv_row(out, learning_housekeeping_csv_header());
        }

        using event_t = typename std::decay_t<decltype(engine.learning)>::DetectorPenalty;
        using key_t = std::tuple<std::string, int, int, int, int, std::string>;
        std::map<key_t, event_t> events;
        {
            std::lock_guard<std::mutex> lock(*engine.learning.mutex);
            for (const auto &record : engine.learning.detector_penalty_events) {
                if (record.iter != iteration ||
                    record.obsnum != engine.observation_identity.obsnum ||
                    record.reason != "busy_network_pathology") {
                    continue;
                }
                const key_t key{record.obsnum, record.iter, record.scan,
                                record.nw, record.array, record.reason};
                const auto [it, inserted] = events.emplace(key, record);
                if (!inserted &&
                    (!std::isfinite(it->second.score) ||
                     (std::isfinite(record.score) &&
                      record.score > it->second.score))) {
                    it->second = record;
                }
            }
        }

        std::size_t rows_written = 0;
        std::map<std::string, std::vector<event_t>> events_by_obsnum;
        for (const auto &[key, event] : events) {
            (void)key;
            events_by_obsnum[event.obsnum].push_back(event);
        }
        for (const auto &[obsnum, obs_events] : events_by_obsnum) {
            std::vector<std::string> hk_files;
            std::string discovery_status;
            try {
                hk_files = find_learning_housekeeping_files(rawobs, obsnum);
                discovery_status =
                    hk_files.empty() ? "housekeeping_file_not_found" :
                    (hk_files.size() > 1 ? "housekeeping_file_ambiguous" : "");
            }
            catch (const std::filesystem::filesystem_error &) {
                discovery_status = "housekeeping_discovery_error";
            }
            const std::string hk_path =
                hk_files.size() == 1 ? hk_files.front() : std::string{};

            std::optional<netCDF::NcFile> hk_file;
            std::string file_status = discovery_status;
            if (file_status.empty()) {
                try {
                    hk_file.emplace(hk_path, netCDF::NcFile::read);
                }
                catch (const netCDF::exceptions::NcException &) {
                    file_status = "housekeeping_file_read_error";
                }
            }

            using channel_data_t =
                std::pair<std::vector<double>, std::vector<double>>;
            std::map<std::string, channel_data_t> channel_data;
            std::map<std::string, std::string> channel_read_status;
            if (file_status.empty()) {
                for (const auto &channel : learning_housekeeping_channels()) {
                    try {
                        channel_data.emplace(
                            channel.id,
                            channel_data_t{
                                read_learning_housekeeping_variable(
                                    *hk_file, channel.time_variable),
                                read_learning_housekeeping_variable(
                                    *hk_file, channel.value_variable)});
                    }
                    catch (const netCDF::exceptions::NcException &) {
                        channel_read_status.emplace(channel.id,
                                                    "channel_read_error");
                    }
                }
            }

            for (const auto &event : obs_events) {
                for (const auto &channel : learning_housekeeping_channels()) {
                    LearningHousekeepingMatch match;
                    if (!file_status.empty()) {
                        match.status = file_status;
                    }
                    else if (const auto status =
                                 channel_read_status.find(channel.id);
                             status != channel_read_status.end()) {
                        match.status = status->second;
                    }
                    else {
                        const auto &data = channel_data.at(channel.id);
                        match = match_learning_housekeeping_sample(
                            data.first, data.second,
                            event.event_time_unix_sec);
                    }
                    const auto csv = csv_escaped;
                    write_csv_row(out, {
                        csv(learning_housekeeping_schema_version),
                        std::to_string(iteration), csv(event.obsnum),
                        std::to_string(event.scan), std::to_string(event.nw),
                        std::to_string(event.array),
                        learning_housekeeping_number(event.score),
                        csv(event.reason),
                        learning_housekeeping_number(event.event_time_unix_sec),
                        csv("ptc_tel_time_chunk_midpoint"), csv(hk_path),
                        csv(channel.group), csv(channel.id), csv(channel.name),
                        csv(channel.unit), csv(match.status),
                        learning_housekeeping_number(match.sample_time_unix_sec),
                        learning_housekeeping_number(match.sample_offset_sec),
                        learning_housekeeping_number(match.sample_age_sec),
                        learning_housekeeping_number(match.value),
                        learning_housekeeping_number(
                            match.previous_sample_time_unix_sec),
                        learning_housekeeping_number(match.previous_value),
                        learning_housekeeping_number(match.delta_from_previous),
                        learning_housekeeping_number(match.next_sample_time_unix_sec),
                        learning_housekeeping_number(match.next_value),
                        learning_housekeeping_number(match.delta_to_next),
                        learning_housekeeping_number(match.local_excursion)});
                    ++rows_written;
                }
            }
        }
        out.flush();
        if (!out) {
            throw std::runtime_error(
                "failed to write required learning housekeeping QA output " +
                output_path);
        }
        out.close();
        if (!out) {
            throw std::runtime_error(
                "failed to finalize required learning housekeeping QA output " +
                output_path);
        }
        logger->info(
            "learning housekeeping QA: iteration={} rows={} output={}",
            iteration, rows_written, output_path);
    }
}

}  // namespace citlali::pipeline
