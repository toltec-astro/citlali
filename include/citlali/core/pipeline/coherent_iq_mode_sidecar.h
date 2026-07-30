#pragma once

#include <citlali/core/error/error.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/coherent_iq_mode_observer.h>
#include <citlali/core/pipeline/rawobs_data_items.h>
#include <citlali/core/pipeline/rawobs_detector_inventory.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <netcdf>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *coherent_iq_sidecar_schema_version =
    "citlali-coherent-iq-mode-sidecar-v1";
inline constexpr const char *coherent_iq_sidecar_filename =
    "coherent_iq_mode_events.yaml";
inline constexpr int coherent_iq_unavailable_sample = -2147483647;

struct CoherentIqCandidatePoint {
    double time_unix_sec = std::numeric_limits<double>::quiet_NaN();
    double score = std::numeric_limits<double>::quiet_NaN();
    std::string kind;
};

struct CoherentIqCandidate {
    int scan_one_based = 0;
    int network = -1;
    double time_unix_sec = std::numeric_limits<double>::quiet_NaN();
    int supporting_detector_events = 0;
    double maximum_rtc_score = std::numeric_limits<double>::quiet_NaN();
    std::string candidate_kinds;
    int seed_network_count = 0;
    std::string seed_networks;
};

struct CoherentIqSidecarRecord {
    CoherentIqCandidate candidate;
    CoherentIqModeScore mode_score;
    int cross_network_coincident_count = 0;
    std::string cross_network_coincident_networks;
};

inline std::vector<CoherentIqCandidate> cluster_coherent_iq_candidates(
    int scan_one_based, int network,
    std::vector<CoherentIqCandidatePoint> points, double tolerance_sec,
    int maximum_candidates) {
    points.erase(
        std::remove_if(
            points.begin(), points.end(),
            [](const auto &point) {
                return !std::isfinite(point.time_unix_sec) ||
                       !std::isfinite(point.score);
            }),
        points.end());
    std::sort(
        points.begin(), points.end(),
        [](const auto &left, const auto &right) {
            return std::tie(left.time_unix_sec, left.kind, left.score) <
                   std::tie(right.time_unix_sec, right.kind, right.score);
        });

    std::vector<std::vector<CoherentIqCandidatePoint>> clusters;
    for (const auto &point : points) {
        if (clusters.empty() ||
            point.time_unix_sec -
                    clusters.back().back().time_unix_sec >
                tolerance_sec) {
            clusters.push_back({});
        }
        clusters.back().push_back(point);
    }

    std::vector<CoherentIqCandidate> result;
    for (auto &cluster : clusters) {
        std::vector<double> times;
        std::set<std::string> kinds;
        double maximum_score = -std::numeric_limits<double>::infinity();
        for (const auto &point : cluster) {
            times.push_back(point.time_unix_sec);
            kinds.insert(point.kind);
            maximum_score = std::max(maximum_score, point.score);
        }
        std::sort(times.begin(), times.end());
        const double event_time =
            times.size() % 2 == 1
                ? times[times.size() / 2]
                : 0.5 * (times[times.size() / 2 - 1] +
                         times[times.size() / 2]);
        std::string kind_text;
        for (const auto &kind : kinds) {
            if (!kind_text.empty()) {
                kind_text += "+";
            }
            kind_text += kind;
        }
        result.push_back(
            {scan_one_based, network, event_time,
             static_cast<int>(cluster.size()), maximum_score, kind_text,
             1, std::to_string(network)});
    }

    std::sort(
        result.begin(), result.end(),
        [](const auto &left, const auto &right) {
            if (left.supporting_detector_events !=
                right.supporting_detector_events) {
                return left.supporting_detector_events >
                       right.supporting_detector_events;
            }
            if (left.maximum_rtc_score != right.maximum_rtc_score) {
                return left.maximum_rtc_score > right.maximum_rtc_score;
            }
            return left.time_unix_sec < right.time_unix_sec;
        });
    if (static_cast<int>(result.size()) > maximum_candidates) {
        result.resize(static_cast<std::size_t>(maximum_candidates));
    }
    std::sort(
        result.begin(), result.end(),
        [](const auto &left, const auto &right) {
            return left.time_unix_sec < right.time_unix_sec;
        });
    return result;
}

inline std::vector<CoherentIqCandidate>
cluster_coherent_iq_candidates_across_networks(
    std::vector<CoherentIqCandidate> candidates, double tolerance_sec) {
    std::sort(
        candidates.begin(), candidates.end(),
        [](const auto &left, const auto &right) {
            return std::tie(
                       left.scan_one_based, left.time_unix_sec,
                       left.network) <
                   std::tie(
                       right.scan_one_based, right.time_unix_sec,
                       right.network);
        });
    std::vector<std::vector<CoherentIqCandidate>> clusters;
    for (const auto &candidate : candidates) {
        if (clusters.empty() ||
            candidate.scan_one_based !=
                clusters.back().back().scan_one_based ||
            candidate.time_unix_sec -
                    clusters.back().back().time_unix_sec >
                tolerance_sec) {
            clusters.push_back({});
        }
        clusters.back().push_back(candidate);
    }

    std::vector<CoherentIqCandidate> result;
    result.reserve(clusters.size());
    for (const auto &cluster : clusters) {
        std::vector<double> times;
        std::set<int> networks;
        std::set<std::string> kinds;
        int detector_events = 0;
        double maximum_score = -std::numeric_limits<double>::infinity();
        for (const auto &candidate : cluster) {
            times.push_back(candidate.time_unix_sec);
            networks.insert(candidate.network);
            detector_events += candidate.supporting_detector_events;
            maximum_score =
                std::max(maximum_score, candidate.maximum_rtc_score);
            std::size_t begin = 0;
            while (begin < candidate.candidate_kinds.size()) {
                const auto end =
                    candidate.candidate_kinds.find('+', begin);
                kinds.insert(candidate.candidate_kinds.substr(
                    begin, end == std::string::npos
                               ? std::string::npos
                               : end - begin));
                if (end == std::string::npos) {
                    break;
                }
                begin = end + 1;
            }
        }
        std::sort(times.begin(), times.end());
        const double event_time =
            times.size() % 2 == 1
                ? times[times.size() / 2]
                : 0.5 * (times[times.size() / 2 - 1] +
                         times[times.size() / 2]);
        CoherentIqCandidate event;
        event.scan_one_based = cluster.front().scan_one_based;
        event.network = -1;
        event.time_unix_sec = event_time;
        event.supporting_detector_events = detector_events;
        event.maximum_rtc_score = maximum_score;
        event.seed_network_count = static_cast<int>(networks.size());
        for (const auto &kind : kinds) {
            if (!event.candidate_kinds.empty()) {
                event.candidate_kinds += "+";
            }
            event.candidate_kinds += kind;
        }
        for (const auto network : networks) {
            if (!event.seed_networks.empty()) {
                event.seed_networks += " ";
            }
            event.seed_networks += std::to_string(network);
        }
        result.push_back(std::move(event));
    }
    return result;
}

template <class Engine>
std::vector<CoherentIqCandidate> collect_coherent_iq_candidates(
    Engine &engine) {
    const auto &config =
        raw_time_chunk_config(engine).coherent_iq_mode_observer;
    std::vector<CoherentIqCandidate> result;
    const auto tel_time_it = engine.telescope.tel_data.find("TelTime");
    if (tel_time_it == engine.telescope.tel_data.end() ||
        engine.telescope.d_fsmp <= 0.0) {
        throw citlali::error::runtime(
            "coherent-IQ observer requires TelTime and a positive "
            "processed sample rate");
    }
    const auto &tel_time = tel_time_it->second;
    for (Eigen::Index scan = 0;
         scan < engine.telescope.scan_indices.cols(); ++scan) {
        const Eigen::Index outer_start =
            engine.telescope.scan_indices(2, scan);
        if (outer_start < 0 || outer_start >= tel_time.size()) {
            throw citlali::error::runtime(
                "coherent-IQ observer encountered invalid scan indices");
        }
        const double scan_start_time = tel_time(outer_start);
        const auto diagnostics =
            engine.rtcproc.snapshot_coherent_iq_mode_candidates(scan);
        std::map<int, std::vector<CoherentIqCandidatePoint>> by_network;
        for (const auto &row : diagnostics) {
            if (row.det < 0 || row.nw < 0) {
                continue;
            }
            const int network = static_cast<int>(row.nw);
            if (row.step_sample !=
                coherent_iq_unavailable_sample &&
                std::isfinite(row.step_score) &&
                row.step_score >= config.candidate_step_score_min) {
                by_network[network].push_back(
                    {scan_start_time +
                         static_cast<double>(row.step_sample) /
                             engine.telescope.d_fsmp,
                     row.step_score, "step"});
            }
            if (row.impulsive_event_sample !=
                coherent_iq_unavailable_sample &&
                std::isfinite(row.impulsive_event_score) &&
                row.impulsive_event_score >=
                    config.candidate_impulsive_score_min) {
                by_network[network].push_back(
                    {scan_start_time +
                         static_cast<double>(row.impulsive_event_sample) /
                             engine.telescope.d_fsmp,
                     row.impulsive_event_score, "impulsive"});
            }
        }
        for (auto &[network, points] : by_network) {
            auto clustered = cluster_coherent_iq_candidates(
                static_cast<int>(scan + 1), network, std::move(points),
                config.candidate_cluster_tolerance_sec,
                config.max_candidates_per_scan_per_network);
            result.insert(
                result.end(),
                std::make_move_iterator(clustered.begin()),
                std::make_move_iterator(clustered.end()));
        }
    }
    return result;
}

inline std::vector<double> coherent_iq_read_vector(
    netCDF::NcVar variable) {
    const auto dims = variable.getDims();
    if (dims.empty()) {
        return {};
    }
    std::size_t count = 1;
    for (const auto &dim : dims) {
        count *= dim.getSize();
    }
    std::vector<double> result(count);
    variable.getVar(result.data());
    return result;
}

template <class Calib>
CoherentIqModeScore score_coherent_iq_candidate_from_file(
    const CoherentIqCandidate &candidate,
    const CoherentIqModeTemplate &mode_template,
    const std::string &filepath, const Calib &calib,
    const citlali::config::RawTimeChunkCoherentIqModeObserverConfig &config) {
    using namespace netCDF;
    NcFile file(filepath, NcFile::read);
    const auto recv_var = file.getVar("Data.Toltec.RecvTime");
    const auto is_var = file.getVar("Data.Toltec.Is");
    const auto qs_var = file.getVar("Data.Toltec.Qs");
    const auto tone_var = file.getVar("Header.Toltec.ToneFreq");
    if (recv_var.isNull() || is_var.isNull() || qs_var.isNull() ||
        tone_var.isNull()) {
        throw citlali::error::io(
            "coherent-IQ observer input lacks required raw-I/Q variables: " +
            filepath);
    }
    const auto recv_time = coherent_iq_read_vector(recv_var);
    const bool invalid_time = std::any_of(
        recv_time.begin(), recv_time.end(),
        [](double value) { return !std::isfinite(value); });
    const bool nonincreasing_time =
        std::adjacent_find(
            recv_time.begin(), recv_time.end(),
            [](double left, double right) { return left >= right; }) !=
        recv_time.end();
    if (recv_time.size() < 8 || invalid_time || nonincreasing_time) {
        throw citlali::error::io(
            "coherent-IQ observer input has invalid receive time: " +
            filepath);
    }
    const auto is_dims = is_var.getDims();
    const auto qs_dims = qs_var.getDims();
    if (is_dims.size() != 2 || qs_dims.size() != 2) {
        throw citlali::error::io(
            "coherent-IQ observer I/Q variables are not two-dimensional: " +
            filepath);
    }
    const auto n_tones = is_var.getDim(1).getSize();
    if (is_var.getDim(0).getSize() != recv_time.size() ||
        qs_var.getDim(0).getSize() != is_var.getDim(0).getSize() ||
        qs_var.getDim(1).getSize() != n_tones) {
        throw citlali::error::io(
            "coherent-IQ observer I/Q shapes differ: " + filepath);
    }

    const double event = candidate.time_unix_sec;
    const auto pre_begin = std::lower_bound(
        recv_time.begin(), recv_time.end(),
        event - config.guard_window_sec - config.pre_window_sec);
    const auto pre_end = std::lower_bound(
        recv_time.begin(), recv_time.end(),
        event - config.guard_window_sec);
    const auto post_begin = std::upper_bound(
        recv_time.begin(), recv_time.end(),
        event + config.guard_window_sec);
    const auto post_end = std::upper_bound(
        recv_time.begin(), recv_time.end(),
        event + config.guard_window_sec + config.post_window_sec);
    const auto pre_first =
        static_cast<std::size_t>(pre_begin - recv_time.begin());
    const auto pre_last =
        static_cast<std::size_t>(pre_end - recv_time.begin());
    const auto post_first =
        static_cast<std::size_t>(post_begin - recv_time.begin());
    const auto post_last =
        static_cast<std::size_t>(post_end - recv_time.begin());
    if (pre_last - pre_first < 4 || post_last - post_first < 4) {
        CoherentIqModeScore result;
        result.status = "incomplete_raw_window";
        result.template_id = mode_template.template_id;
        result.template_version = mode_template.template_version;
        result.network = candidate.network;
        result.template_tone_count =
            static_cast<int>(mode_template.tones.size());
        result.compatibility_note =
            "candidate lacks four raw samples in both comparison windows";
        return result;
    }

    const std::size_t read_first = pre_first;
    const std::size_t read_last = post_last;
    const std::size_t n_rows = read_last - read_first;
    std::vector<double> is(n_rows * n_tones);
    std::vector<double> qs(n_rows * n_tones);
    is_var.getVar({read_first, 0}, {n_rows, n_tones}, is.data());
    qs_var.getVar({read_first, 0}, {n_rows, n_tones}, qs.data());

    const auto tone_dims = tone_var.getDims();
    std::vector<double> tone_offsets(n_tones);
    if (tone_dims.size() == 2) {
        tone_var.getVar({0, 0}, {1, n_tones}, tone_offsets.data());
    }
    else if (tone_dims.size() == 1) {
        tone_var.getVar(tone_offsets.data());
    }
    else {
        throw citlali::error::io(
            "coherent-IQ observer tone-frequency shape is unsupported: " +
            filepath);
    }

    const auto limits = calib.nw_limits.find(candidate.network);
    if (limits == calib.nw_limits.end()) {
        throw citlali::error::runtime(
            "coherent-IQ observer cannot locate network in APT");
    }
    const auto [apt_start, apt_end] = limits->second;
    if (apt_end - apt_start != static_cast<Eigen::Index>(n_tones)) {
        throw citlali::error::runtime(
            "coherent-IQ observer APT/raw tone counts differ");
    }

    std::vector<int> uids(n_tones, -1);
    std::vector<double> phase_change(n_tones);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t tone = 0; tone < n_tones; ++tone) {
        const Eigen::Index det =
            apt_start + static_cast<Eigen::Index>(tone);
        uids[tone] = static_cast<int>(calib.apt.at("uid")(det));
        const bool apt_usable =
            calib.apt.count("flag") == 0 ||
            calib.apt.at("flag")(det) == 0.0;
        std::complex<double> pre_sum{0.0, 0.0};
        std::complex<double> post_sum{0.0, 0.0};
        int pre_count = 0;
        int post_count = 0;
        for (std::size_t raw_row = pre_first; raw_row < pre_last;
             ++raw_row) {
            const std::size_t row = raw_row - read_first;
            const std::complex<double> value{
                is[row * n_tones + tone],
                qs[row * n_tones + tone]};
            if (std::isfinite(value.real()) &&
                std::isfinite(value.imag())) {
                pre_sum += value;
                ++pre_count;
            }
        }
        for (std::size_t raw_row = post_first; raw_row < post_last;
             ++raw_row) {
            const std::size_t row = raw_row - read_first;
            const std::complex<double> value{
                is[row * n_tones + tone],
                qs[row * n_tones + tone]};
            if (std::isfinite(value.real()) &&
                std::isfinite(value.imag())) {
                post_sum += value;
                ++post_count;
            }
        }
        if (!apt_usable || pre_count == 0 || post_count == 0 ||
            std::abs(pre_sum) == 0.0) {
            phase_change[tone] = nan;
            continue;
        }
        const auto before =
            pre_sum / static_cast<double>(pre_count);
        const auto after =
            post_sum / static_cast<double>(post_count);
        phase_change[tone] = std::arg(after / before) * 1.0e3;
    }
    file.close();
    return score_coherent_iq_mode_event(
        mode_template, candidate.network, uids, tone_offsets,
        phase_change);
}

inline void attach_coherent_iq_cross_network_coincidence(
    std::vector<CoherentIqSidecarRecord> &records,
    double tolerance_sec) {
    for (auto &record : records) {
        std::set<int> networks;
        for (const auto &other : records) {
            if (other.mode_score.status == "scored" &&
                std::abs(other.candidate.time_unix_sec -
                         record.candidate.time_unix_sec) <=
                    tolerance_sec) {
                networks.insert(other.candidate.network);
            }
        }
        record.cross_network_coincident_count =
            static_cast<int>(networks.size());
        for (const auto network : networks) {
            if (!record.cross_network_coincident_networks.empty()) {
                record.cross_network_coincident_networks += " ";
            }
            record.cross_network_coincident_networks +=
                std::to_string(network);
        }
    }
}

inline YAML::Node coherent_iq_score_node(
    const CoherentIqModeScore &score) {
    YAML::Node node;
    node["status"] = score.status;
    node["template_id"] = score.template_id;
    node["template_version"] = score.template_version;
    node["primary_mode_id"] = score.primary_mode_id;
    auto assign_finite = [&node](const char *name, double value) {
        if (std::isfinite(value)) {
            node[name] = value;
        }
        else {
            node[name] = YAML::Node();
        }
    };
    assign_finite(
        "projection_amplitude_mrad",
        score.projection_amplitude_mrad);
    node["sign"] = score.sign;
    assign_finite("cosine_similarity", score.cosine_similarity);
    assign_finite(
        "absolute_cosine_similarity",
        score.absolute_cosine_similarity);
    assign_finite(
        "explained_energy_fraction",
        score.explained_energy_fraction);
    assign_finite(
        "residual_energy_mrad2", score.residual_energy_mrad2);
    assign_finite("total_energy_mrad2", score.total_energy_mrad2);
    assign_finite(
        "multi_mode_explained_energy_fraction",
        score.multi_mode_explained_energy_fraction);
    assign_finite(
        "common_phase_explained_energy_fraction",
        score.common_phase_explained_energy_fraction);
    assign_finite(
        "delay_slope_explained_energy_fraction",
        score.delay_slope_explained_energy_fraction);
    node["compatible_tone_count"] = score.compatible_tone_count;
    node["template_tone_count"] = score.template_tone_count;
    node["compatible_tone_fraction"] =
        score.compatible_tone_fraction;
    node["rejected_tone_count"] = score.rejected_tone_count;
    node["compatibility_note"] = score.compatibility_note;
    return node;
}

template <class Engine, class RawObs, class = void>
struct supports_coherent_iq_mode_sidecar : std::false_type {};

template <class Engine, class RawObs>
struct supports_coherent_iq_mode_sidecar<
    Engine, RawObs,
    std::void_t<
        decltype(std::declval<Engine &>().telescope.scan_indices),
        decltype(std::declval<Engine &>().telescope.tel_data),
        decltype(std::declval<Engine &>().rtcproc
                     .snapshot_coherent_iq_mode_candidates(
                         Eigen::Index{})),
        decltype(std::declval<Engine &>().calib.nw_limits),
        decltype(std::declval<Engine &>().calib.apt),
        decltype(std::declval<Engine &>().observation_identity.obsnum),
        decltype(std::declval<Engine &>().output_paths.obsnum_dir_name),
        decltype(std::declval<const RawObs &>().kidsdata())>>
    : std::true_type {};

template <class Engine, class RawObs, class Logger>
std::filesystem::path write_coherent_iq_mode_sidecar_supported(
    Engine &engine, const RawObs &rawobs, const Logger &logger) {
    const auto &config =
        raw_time_chunk_config(engine).coherent_iq_mode_observer;
    if (!config.enabled) {
        return {};
    }

    std::map<int, CoherentIqModeTemplate> templates;
    for (const auto &path : config.template_paths) {
        auto mode_template = load_coherent_iq_mode_template(path);
        if (!templates.emplace(
                mode_template.network, std::move(mode_template))
                 .second) {
            throw citlali::error::invalid_config(
                "coherent-IQ observer has multiple templates for one "
                "network");
        }
    }

    std::map<int, std::string> raw_files;
    for (const auto &data_item_ref : rawobs.kidsdata()) {
        const auto &data_item =
            detail::unwrap_reference_wrapper(data_item_ref);
        const int network = static_cast<int>(
            rawobs_interface_id(data_item.interface()));
        if (!raw_files.emplace(network, data_item.filepath()).second) {
            throw citlali::error::runtime(
                "coherent-IQ observer received multiple raw files for "
                "one network");
        }
    }
    const auto network_candidates =
        collect_coherent_iq_candidates(engine);
    auto candidates = cluster_coherent_iq_candidates_across_networks(
        network_candidates,
        config.cross_network_tolerance_sec);
    std::vector<CoherentIqSidecarRecord> records;
    records.reserve(candidates.size() * raw_files.size());
    for (const auto &candidate : candidates) {
        for (const auto &[network, filepath] : raw_files) {
            auto network_candidate = candidate;
            network_candidate.network = network;
            CoherentIqModeScore score;
            score.network = network;
            const auto template_it = templates.find(network);
            if (template_it == templates.end()) {
                score.status = "template_unavailable";
                score.compatibility_note =
                    "no configured template for this present network";
            }
            else {
                score = score_coherent_iq_candidate_from_file(
                    network_candidate, template_it->second, filepath,
                    engine.calib, config);
            }
            records.push_back(
                {std::move(network_candidate), std::move(score), 0, ""});
        }
    }
    attach_coherent_iq_cross_network_coincidence(
        records, config.cross_network_tolerance_sec);

    YAML::Node root;
    root["schema_version"] = coherent_iq_sidecar_schema_version;
    root["diagnostic_schema_version"] =
        coherent_iq_diagnostic_schema_version;
    root["lifecycle_state"] = "observe_only";
    root["scientific_effect"] =
        "none; samples, flags, weights, maps, and learning state are "
        "unchanged";
    root["observation"]["obsnum"] =
        engine.observation_identity.obsnum;
    root["requested"]["candidate_step_score_min"] =
        config.candidate_step_score_min;
    root["requested"]["candidate_impulsive_score_min"] =
        config.candidate_impulsive_score_min;
    root["requested"]["candidate_cluster_tolerance_sec"] =
        config.candidate_cluster_tolerance_sec;
    root["requested"]["pre_window_sec"] = config.pre_window_sec;
    root["requested"]["guard_window_sec"] = config.guard_window_sec;
    root["requested"]["post_window_sec"] = config.post_window_sec;
    root["requested"]["cross_network_tolerance_sec"] =
        config.cross_network_tolerance_sec;
    root["requested"]["max_candidates_per_scan_per_network"] =
        config.max_candidates_per_scan_per_network;
    root["events"] = YAML::Node(YAML::NodeType::Sequence);

    for (const auto &[network, filepath] : raw_files) {
        auto network_node = root["network_status"][network];
        network_node["raw_filepath"] = filepath;
        const auto template_it = templates.find(network);
        network_node["template_status"] =
            template_it == templates.end() ? "unavailable" : "loaded";
        if (template_it != templates.end()) {
            network_node["template_id"] =
                template_it->second.template_id;
            network_node["template_version"] =
                template_it->second.template_version;
            network_node["template_path"] =
                template_it->second.source_path;
            network_node["template_sha256"] =
                template_it->second.source_sha256;
            network_node["readout_id"] =
                template_it->second.readout_id;
            network_node["unresolved_metadata"] =
                template_it->second.unresolved_metadata;
        }
    }
    for (const auto &[network, mode_template] : templates) {
        auto template_node = root["template_inventory"][network];
        template_node["template_id"] = mode_template.template_id;
        template_node["template_version"] =
            mode_template.template_version;
        template_node["template_path"] = mode_template.source_path;
        template_node["template_sha256"] =
            mode_template.source_sha256;
        template_node["created_at_utc"] =
            mode_template.created_at_utc;
        template_node["lifecycle_state"] =
            mode_template.lifecycle_state;
        template_node["readout_id"] =
            mode_template.readout_id;
        template_node["readout_coordinate_system"] =
            mode_template.readout_coordinate_system;
        template_node["unresolved_metadata"] =
            mode_template.unresolved_metadata;
        template_node["raw_network_present"] =
            raw_files.count(network) != 0;
    }

    for (const auto &record : records) {
        YAML::Node node;
        node["scan_one_based"] =
            record.candidate.scan_one_based;
        node["network"] = record.candidate.network;
        node["event_time_unix_sec"] =
            record.candidate.time_unix_sec;
        node["candidate_kinds"] =
            record.candidate.candidate_kinds;
        node["seed_network_count"] =
            record.candidate.seed_network_count;
        node["seed_networks"] =
            record.candidate.seed_networks;
        node["supporting_detector_events"] =
            record.candidate.supporting_detector_events;
        node["maximum_rtc_score"] =
            record.candidate.maximum_rtc_score;
        node["mode_score"] =
            coherent_iq_score_node(record.mode_score);
        node["cross_network_coincident_count"] =
            record.cross_network_coincident_count;
        node["cross_network_coincident_networks"] =
            record.cross_network_coincident_networks;
        root["events"].push_back(node);
    }
    root["realized"]["candidate_count"] =
        static_cast<int>(candidates.size());
    root["realized"]["shared_candidate_count"] =
        static_cast<int>(candidates.size());
    root["realized"]["network_seed_candidate_count"] =
        static_cast<int>(network_candidates.size());
    root["realized"]["network_event_score_count"] =
        static_cast<int>(records.size());
    root["realized"]["scored_count"] = static_cast<int>(
        std::count_if(
            records.begin(), records.end(), [](const auto &record) {
                return record.mode_score.status == "scored";
            }));
    root["realized"]["network_count"] =
        static_cast<int>(raw_files.size());
    root["realized"]["template_count"] =
        static_cast<int>(templates.size());

    const std::filesystem::path output_path =
        std::filesystem::path(engine.output_paths.obsnum_dir_name) /
        "raw" / coherent_iq_sidecar_filename;
    write_yaml_file_atomic(output_path, root);
    logger->info(
        "observe-only coherent-IQ mode sidecar: {} ({} event candidates, "
        "{} network scores, {} templates, {} raw networks)",
        output_path.string(), candidates.size(), records.size(),
        templates.size(), raw_files.size());
    engine.rtcproc.reset_coherent_iq_mode_candidates();
    return output_path;
}

template <class Engine, class RawObs, class Logger>
std::filesystem::path write_coherent_iq_mode_sidecar_if_requested(
    Engine &engine, const RawObs &rawobs, const Logger &logger) {
    const auto &config =
        raw_time_chunk_config(engine).coherent_iq_mode_observer;
    if (!config.enabled) {
        return {};
    }
    if constexpr (supports_coherent_iq_mode_sidecar<
                      Engine, RawObs>::value) {
        return write_coherent_iq_mode_sidecar_supported(
            engine, rawobs, logger);
    }
    else {
        throw citlali::error::runtime(
            "coherent-IQ observer requires the production observation "
            "engine and raw-observation interfaces");
    }
}

}  // namespace citlali::pipeline
