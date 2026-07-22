#include <citlali/core/pipeline/reduction_restart_checkpoint.h>

#include <citlali/core/pipeline/processed_timestream_config_serialization.h>
#include <citlali/core/utils/netcdf_io.h>
#include <citlali_config/gitversion.h>

#include <netcdf>
#include <netcdf.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline {
namespace {

using EffectiveSampleMask = ReductionLearningState::EffectiveSampleMask;
using EffectiveSampleMaskKey =
    ReductionLearningState::EffectiveSampleMaskKey;
using EffectiveDetectorPenaltyKey =
    ReductionLearningState::EffectiveDetectorPenaltyKey;
using DetectorPenalty = ReductionLearningState::DetectorPenalty;

[[noreturn]] void checkpoint_error(const std::filesystem::path &path,
                                   const std::string &message) {
    throw DataIOError("invalid Citlali restart checkpoint " + path.string() +
                      ": " + message);
}

template <class Value>
void write_numeric_vector(netCDF::NcFile &file, const std::string &name,
                          const netCDF::NcType &type,
                          const netCDF::NcDim &dimension,
                          const std::vector<Value> &values) {
    auto var = file.addVar(name, type, dimension);
    if (!values.empty()) {
        var.putVar(values.data());
    }
}

void write_string_vector(netCDF::NcFile &file, const std::string &name,
                         const netCDF::NcDim &dimension,
                         const std::vector<std::string> &values) {
    auto var = file.addVar(name, netCDF::ncString, dimension);
    for (std::size_t i = 0; i < values.size(); ++i) {
        var.putVar(std::vector<std::size_t>{i}, values[i]);
    }
}

template <class Value>
Value read_scalar(netCDF::NcFile &file, const std::filesystem::path &path,
                  const std::string &name) {
    const auto var = file.getVar(name);
    if (var.isNull() || var.getDimCount() != 1 ||
        var.getDim(0).getSize() != 1) {
        checkpoint_error(path, "missing or malformed scalar variable '" +
                                   name + "'");
    }
    Value value{};
    var.getVar(&value);
    return value;
}

std::string read_scalar_string(netCDF::NcFile &file,
                               const std::filesystem::path &path,
                               const std::string &name) {
    const auto var = file.getVar(name);
    if (var.isNull() || var.getDimCount() != 1 ||
        var.getDim(0).getSize() != 1) {
        checkpoint_error(path, "missing or malformed string scalar '" +
                                   name + "'");
    }
    char *raw = nullptr;
    var.getVar(std::vector<std::size_t>{0}, &raw);
    const std::string value = raw == nullptr ? std::string{} : raw;
    if (raw != nullptr) {
        nc_free_string(1, &raw);
    }
    return value;
}

template <class Value>
std::vector<Value> read_numeric_vector(
    netCDF::NcFile &file, const std::filesystem::path &path,
    const std::string &name, std::size_t expected_size) {
    if (expected_size == 0) {
        return {};
    }
    const auto var = file.getVar(name);
    if (var.isNull() || var.getDimCount() != 1 ||
        var.getDim(0).getSize() != expected_size) {
        checkpoint_error(path, "missing or malformed vector variable '" +
                                   name + "'");
    }
    std::vector<Value> values(expected_size);
    var.getVar(values.data());
    return values;
}

std::vector<std::string> read_string_vector(
    netCDF::NcFile &file, const std::filesystem::path &path,
    const std::string &name, std::size_t expected_size) {
    if (expected_size == 0) {
        return {};
    }
    const auto var = file.getVar(name);
    if (var.isNull() || var.getDimCount() != 1 ||
        var.getDim(0).getSize() != expected_size) {
        checkpoint_error(path, "missing or malformed string vector '" +
                                   name + "'");
    }
    std::vector<std::string> values(expected_size);
    for (std::size_t i = 0; i < expected_size; ++i) {
        char *raw = nullptr;
        var.getVar(std::vector<std::size_t>{i}, &raw);
        values[i] = raw == nullptr ? std::string{} : raw;
        if (raw != nullptr) {
            nc_free_string(1, &raw);
        }
    }
    return values;
}

std::map<std::string, int> observation_index(
    const std::vector<std::string> &observation_ids) {
    std::map<std::string, int> result;
    for (std::size_t i = 0; i < observation_ids.size(); ++i) {
        result.try_emplace(observation_ids[i], static_cast<int>(i));
    }
    return result;
}

struct FlatCheckpointState {
    std::vector<int> mask_observation_index;
    std::vector<int> mask_scan;
    std::vector<int> mask_apply_pre_rtc;
    std::vector<int> mask_uid;
    std::vector<int> mask_iteration;
    std::vector<long long> mask_start;
    std::vector<long long> mask_stop;

    std::vector<int> penalty_observation_index;
    std::vector<std::string> penalty_producer;
    std::vector<std::string> penalty_reason;
    std::vector<int> penalty_iteration;
    std::vector<int> penalty_scan;
    std::vector<int> penalty_uid;
    std::vector<int> penalty_network;
    std::vector<int> penalty_array;
    std::vector<double> penalty_factor;
    std::vector<double> penalty_score;
    std::vector<double> penalty_event_time_unix_sec;
    std::vector<int> penalty_scan_local;
};

FlatCheckpointState flatten_effective_state(
    const ReductionLearningState &learning,
    const std::map<std::string, int> &obs_index) {
    FlatCheckpointState flat;
    std::lock_guard<std::mutex> lock(*learning.mutex);

    std::size_t mask_count = 0;
    for (const auto &[key, intervals] : learning.effective_sample_masks) {
        (void) key;
        mask_count += intervals.size();
    }
    flat.mask_observation_index.reserve(mask_count);
    flat.mask_scan.reserve(mask_count);
    flat.mask_apply_pre_rtc.reserve(mask_count);
    flat.mask_uid.reserve(mask_count);
    flat.mask_iteration.reserve(mask_count);
    flat.mask_start.reserve(mask_count);
    flat.mask_stop.reserve(mask_count);
    for (const auto &[key, intervals] : learning.effective_sample_masks) {
        const auto &[obsnum, scan, apply_pre_rtc, uid] = key;
        const auto obs = obs_index.find(obsnum);
        if (obs == obs_index.end()) {
            throw std::invalid_argument(
                "effective sample-mask state names observation " + obsnum +
                " outside the reduction observation set");
        }
        for (const auto &interval : intervals) {
            flat.mask_observation_index.push_back(obs->second);
            flat.mask_scan.push_back(scan);
            flat.mask_apply_pre_rtc.push_back(apply_pre_rtc ? 1 : 0);
            flat.mask_uid.push_back(uid);
            flat.mask_iteration.push_back(interval.iter);
            flat.mask_start.push_back(interval.start);
            flat.mask_stop.push_back(interval.stop);
        }
    }

    const auto penalty_count = learning.effective_detector_penalties.size();
    flat.penalty_observation_index.reserve(penalty_count);
    flat.penalty_producer.reserve(penalty_count);
    flat.penalty_reason.reserve(penalty_count);
    flat.penalty_iteration.reserve(penalty_count);
    flat.penalty_scan.reserve(penalty_count);
    flat.penalty_uid.reserve(penalty_count);
    flat.penalty_network.reserve(penalty_count);
    flat.penalty_array.reserve(penalty_count);
    flat.penalty_factor.reserve(penalty_count);
    flat.penalty_score.reserve(penalty_count);
    flat.penalty_event_time_unix_sec.reserve(penalty_count);
    flat.penalty_scan_local.reserve(penalty_count);
    for (const auto &[key, penalty] :
         learning.effective_detector_penalties) {
        (void) key;
        const auto obs = obs_index.find(penalty.obsnum);
        if (obs == obs_index.end()) {
            throw std::invalid_argument(
                "effective detector-penalty state names observation " +
                penalty.obsnum + " outside the reduction observation set");
        }
        flat.penalty_observation_index.push_back(obs->second);
        flat.penalty_producer.push_back(penalty.producer);
        flat.penalty_reason.push_back(penalty.reason);
        flat.penalty_iteration.push_back(penalty.iter);
        flat.penalty_scan.push_back(penalty.scan);
        flat.penalty_uid.push_back(penalty.uid);
        flat.penalty_network.push_back(penalty.nw);
        flat.penalty_array.push_back(penalty.array);
        flat.penalty_factor.push_back(penalty.factor);
        flat.penalty_score.push_back(penalty.score);
        flat.penalty_event_time_unix_sec.push_back(
            penalty.event_time_unix_sec);
        flat.penalty_scan_local.push_back(penalty.scan_local ? 1 : 0);
    }
    return flat;
}

}  // namespace

std::filesystem::path reduction_restart_checkpoint_path(
    const std::filesystem::path &reduction_dir) {
    return reduction_dir / reduction_restart_checkpoint_filename;
}

std::string learning_restart_policy_snapshot(
    const citlali::config::TimestreamLearningConfig &config) {
    return YAML::Dump(learning_config_node(config));
}

void write_reduction_restart_checkpoint(
    const std::filesystem::path &reduction_dir, int completed_iteration,
    const std::string &fruit_loops_type,
    const std::vector<std::string> &observation_ids,
    const citlali::config::TimestreamLearningConfig &learning_config,
    const ReductionLearningState &learning) {
    if (completed_iteration < 0) {
        throw std::invalid_argument(
            "restart checkpoint completed iteration must be nonnegative");
    }
    if (observation_ids.empty()) {
        throw std::invalid_argument(
            "restart checkpoint requires at least one observation identity");
    }
    const auto obs_index = observation_index(observation_ids);
    const auto flat = flatten_effective_state(learning, obs_index);
    const auto output_path = reduction_restart_checkpoint_path(reduction_dir);

    write_netcdf_atomic(output_path.string(), [&](netCDF::NcFile &file) {
        add_netcdf_var(file, "schema_version",
                       std::string{reduction_restart_checkpoint_schema_version});
        add_netcdf_var(file, "creator_version", std::string{CITLALI_GIT_VERSION});
        add_netcdf_var(file, "completed_iteration", completed_iteration);
        add_netcdf_var(file, "next_iteration", completed_iteration + 1);
        add_netcdf_var(file, "fruit_loops_type", fruit_loops_type);
        add_netcdf_var(file, "learning_policy_yaml",
                       learning_restart_policy_snapshot(learning_config));
        add_netcdf_var(file, "observation_count",
                       static_cast<long long>(observation_ids.size()));
        add_netcdf_var(
            file, "effective_sample_mask_interval_count",
            static_cast<long long>(flat.mask_iteration.size()));
        add_netcdf_var(
            file, "effective_detector_penalty_count",
            static_cast<long long>(flat.penalty_iteration.size()));

        const auto observation_dim =
            file.addDim("observation", observation_ids.size());
        write_string_vector(file, "observation_id", observation_dim,
                            observation_ids);

        if (!flat.mask_iteration.empty()) {
            const auto dim = file.addDim("effective_sample_mask_interval",
                                         flat.mask_iteration.size());
            write_numeric_vector(file, "mask_observation_index", netCDF::ncInt,
                                 dim, flat.mask_observation_index);
            write_numeric_vector(file, "mask_scan", netCDF::ncInt, dim,
                                 flat.mask_scan);
            write_numeric_vector(file, "mask_apply_pre_rtc", netCDF::ncInt,
                                 dim, flat.mask_apply_pre_rtc);
            write_numeric_vector(file, "mask_uid", netCDF::ncInt, dim,
                                 flat.mask_uid);
            write_numeric_vector(file, "mask_iteration", netCDF::ncInt, dim,
                                 flat.mask_iteration);
            write_numeric_vector(file, "mask_start", netCDF::ncInt64, dim,
                                 flat.mask_start);
            write_numeric_vector(file, "mask_stop", netCDF::ncInt64, dim,
                                 flat.mask_stop);
        }

        if (!flat.penalty_iteration.empty()) {
            const auto dim = file.addDim("effective_detector_penalty",
                                         flat.penalty_iteration.size());
            write_numeric_vector(file, "penalty_observation_index",
                                 netCDF::ncInt, dim,
                                 flat.penalty_observation_index);
            write_string_vector(file, "penalty_producer", dim,
                                flat.penalty_producer);
            write_string_vector(file, "penalty_reason", dim,
                                flat.penalty_reason);
            write_numeric_vector(file, "penalty_iteration", netCDF::ncInt,
                                 dim, flat.penalty_iteration);
            write_numeric_vector(file, "penalty_scan", netCDF::ncInt, dim,
                                 flat.penalty_scan);
            write_numeric_vector(file, "penalty_uid", netCDF::ncInt, dim,
                                 flat.penalty_uid);
            write_numeric_vector(file, "penalty_network", netCDF::ncInt, dim,
                                 flat.penalty_network);
            write_numeric_vector(file, "penalty_array", netCDF::ncInt, dim,
                                 flat.penalty_array);
            write_numeric_vector(file, "penalty_factor", netCDF::ncDouble,
                                 dim, flat.penalty_factor);
            write_numeric_vector(file, "penalty_score", netCDF::ncDouble,
                                 dim, flat.penalty_score);
            write_numeric_vector(file, "penalty_event_time_unix_sec",
                                 netCDF::ncDouble, dim,
                                 flat.penalty_event_time_unix_sec);
            write_numeric_vector(file, "penalty_scan_local", netCDF::ncInt,
                                 dim, flat.penalty_scan_local);
        }
    });
}

ReductionRestartCheckpointSummary load_reduction_restart_checkpoint(
    const std::filesystem::path &source_reduction_dir,
    const std::string &expected_fruit_loops_type,
    const std::vector<std::string> &expected_observation_ids,
    const citlali::config::TimestreamLearningConfig &expected_learning_config,
    ReductionLearningState &learning) {
    const auto input_path =
        reduction_restart_checkpoint_path(source_reduction_dir);
    if (!std::filesystem::is_directory(source_reduction_dir)) {
        checkpoint_error(input_path,
                         "restart_path is not an existing reduction directory");
    }
    if (!std::filesystem::is_regular_file(input_path)) {
        checkpoint_error(input_path, "required checkpoint file is missing");
    }

    netCDF::NcFile file(input_path.string(), netCDF::NcFile::read);
    const auto schema = read_scalar_string(file, input_path, "schema_version");
    if (schema != reduction_restart_checkpoint_schema_version) {
        checkpoint_error(input_path, "unsupported schema_version '" + schema +
                                         "'");
    }
    const auto creator_version =
        read_scalar_string(file, input_path, "creator_version");
    const int completed_iteration =
        read_scalar<int>(file, input_path, "completed_iteration");
    const int next_iteration =
        read_scalar<int>(file, input_path, "next_iteration");
    if (completed_iteration < 0 || next_iteration != completed_iteration + 1) {
        checkpoint_error(input_path,
                         "iteration identity is not a completed/next pair");
    }
    const auto fruit_loops_type =
        read_scalar_string(file, input_path, "fruit_loops_type");
    if (fruit_loops_type != expected_fruit_loops_type) {
        checkpoint_error(input_path, "fruit_loops.type mismatch: checkpoint='" +
                                         fruit_loops_type + "' current='" +
                                         expected_fruit_loops_type + "'");
    }
    const auto learning_policy =
        read_scalar_string(file, input_path, "learning_policy_yaml");
    if (learning_policy !=
        learning_restart_policy_snapshot(expected_learning_config)) {
        checkpoint_error(
            input_path,
            "timestream.learning policy differs from the checkpoint; exact restart requires the same learning configuration");
    }

    const auto observation_count_ll =
        read_scalar<long long>(file, input_path, "observation_count");
    const auto mask_count_ll = read_scalar<long long>(
        file, input_path, "effective_sample_mask_interval_count");
    const auto penalty_count_ll = read_scalar<long long>(
        file, input_path, "effective_detector_penalty_count");
    if (observation_count_ll <= 0 || mask_count_ll < 0 ||
        penalty_count_ll < 0) {
        checkpoint_error(input_path, "negative or empty record cardinality");
    }
    const auto observation_count =
        static_cast<std::size_t>(observation_count_ll);
    const auto mask_count = static_cast<std::size_t>(mask_count_ll);
    const auto penalty_count = static_cast<std::size_t>(penalty_count_ll);
    const auto observation_ids = read_string_vector(
        file, input_path, "observation_id", observation_count);
    if (observation_ids != expected_observation_ids) {
        checkpoint_error(
            input_path,
            "ordered observation identities differ from the current reduction");
    }

    const auto mask_obs = read_numeric_vector<int>(
        file, input_path, "mask_observation_index", mask_count);
    const auto mask_scan =
        read_numeric_vector<int>(file, input_path, "mask_scan", mask_count);
    const auto mask_pre_rtc = read_numeric_vector<int>(
        file, input_path, "mask_apply_pre_rtc", mask_count);
    const auto mask_uid =
        read_numeric_vector<int>(file, input_path, "mask_uid", mask_count);
    const auto mask_iter = read_numeric_vector<int>(
        file, input_path, "mask_iteration", mask_count);
    const auto mask_start = read_numeric_vector<long long>(
        file, input_path, "mask_start", mask_count);
    const auto mask_stop = read_numeric_vector<long long>(
        file, input_path, "mask_stop", mask_count);

    std::map<EffectiveSampleMaskKey, std::vector<EffectiveSampleMask>> masks;
    for (std::size_t i = 0; i < mask_count; ++i) {
        if (mask_obs[i] < 0 ||
            static_cast<std::size_t>(mask_obs[i]) >= observation_ids.size() ||
            (mask_pre_rtc[i] != 0 && mask_pre_rtc[i] != 1) ||
            mask_iter[i] < 0 || mask_iter[i] > completed_iteration ||
            mask_scan[i] < 0 || mask_uid[i] < 0 || mask_start[i] < 0 ||
            mask_stop[i] < mask_start[i]) {
            checkpoint_error(input_path,
                             "invalid effective sample-mask record at row " +
                                 std::to_string(i));
        }
        const EffectiveSampleMaskKey key{
            observation_ids[static_cast<std::size_t>(mask_obs[i])],
            mask_scan[i], mask_pre_rtc[i] != 0, mask_uid[i]};
        masks[key].push_back(EffectiveSampleMask{
            mask_iter[i], mask_uid[i], mask_start[i], mask_stop[i]});
    }
    for (auto &[key, intervals] : masks) {
        (void) key;
        std::sort(intervals.begin(), intervals.end(),
                  [](const auto &left, const auto &right) {
                      return std::tie(left.start, left.stop, left.iter) <
                             std::tie(right.start, right.stop, right.iter);
                  });
        for (std::size_t i = 1; i < intervals.size(); ++i) {
            if (intervals[i - 1].stop >= intervals[i].start - 1) {
                checkpoint_error(
                    input_path,
                    "effective sample-mask intervals are not a canonical disjoint union");
            }
        }
    }

    const auto penalty_obs = read_numeric_vector<int>(
        file, input_path, "penalty_observation_index", penalty_count);
    const auto penalty_producer = read_string_vector(
        file, input_path, "penalty_producer", penalty_count);
    const auto penalty_reason = read_string_vector(
        file, input_path, "penalty_reason", penalty_count);
    const auto penalty_iter = read_numeric_vector<int>(
        file, input_path, "penalty_iteration", penalty_count);
    const auto penalty_scan = read_numeric_vector<int>(
        file, input_path, "penalty_scan", penalty_count);
    const auto penalty_uid = read_numeric_vector<int>(
        file, input_path, "penalty_uid", penalty_count);
    const auto penalty_network = read_numeric_vector<int>(
        file, input_path, "penalty_network", penalty_count);
    const auto penalty_array = read_numeric_vector<int>(
        file, input_path, "penalty_array", penalty_count);
    const auto penalty_factor = read_numeric_vector<double>(
        file, input_path, "penalty_factor", penalty_count);
    const auto penalty_score = read_numeric_vector<double>(
        file, input_path, "penalty_score", penalty_count);
    const auto penalty_event_time = read_numeric_vector<double>(
        file, input_path, "penalty_event_time_unix_sec", penalty_count);
    const auto penalty_scan_local = read_numeric_vector<int>(
        file, input_path, "penalty_scan_local", penalty_count);

    std::map<EffectiveDetectorPenaltyKey, DetectorPenalty> penalties;
    for (std::size_t i = 0; i < penalty_count; ++i) {
        if (penalty_obs[i] < 0 ||
            static_cast<std::size_t>(penalty_obs[i]) >=
                observation_ids.size() ||
            penalty_iter[i] < 0 || penalty_iter[i] > completed_iteration ||
            penalty_producer[i].empty() || penalty_reason[i].empty() ||
            (penalty_scan_local[i] != 0 && penalty_scan_local[i] != 1)) {
            checkpoint_error(input_path,
                             "invalid effective detector-penalty record at row " +
                                 std::to_string(i));
        }
        DetectorPenalty penalty;
        penalty.obsnum =
            observation_ids[static_cast<std::size_t>(penalty_obs[i])];
        penalty.producer = penalty_producer[i];
        penalty.reason = penalty_reason[i];
        penalty.iter = penalty_iter[i];
        penalty.scan = penalty_scan[i];
        penalty.uid = penalty_uid[i];
        penalty.nw = penalty_network[i];
        penalty.array = penalty_array[i];
        penalty.factor = penalty_factor[i];
        penalty.score = penalty_score[i];
        penalty.event_time_unix_sec = penalty_event_time[i];
        penalty.scan_local = penalty_scan_local[i] != 0;
        const EffectiveDetectorPenaltyKey key{
            penalty.obsnum, penalty.producer, penalty.reason, penalty.scan,
            penalty.uid, penalty.nw, penalty.array, penalty.scan_local};
        if (!penalties.emplace(key, penalty).second) {
            checkpoint_error(
                input_path,
                "duplicate effective detector-penalty scientific identity");
        }
    }

    {
        std::lock_guard<std::mutex> lock(*learning.mutex);
        learning.effective_sample_masks = std::move(masks);
        learning.effective_detector_penalties = std::move(penalties);
    }

    return ReductionRestartCheckpointSummary{
        input_path,
        source_reduction_dir,
        creator_version,
        fruit_loops_type,
        completed_iteration,
        next_iteration,
        observation_ids,
        mask_count,
        penalty_count,
    };
}

}  // namespace citlali::pipeline
