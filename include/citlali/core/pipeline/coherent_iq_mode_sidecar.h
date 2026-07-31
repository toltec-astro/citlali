#pragma once

#include <citlali/core/error/error.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/pipeline/atomic_yaml_output.h>
#include <citlali/core/pipeline/coherent_iq_mode_observer.h>
#include <citlali/core/pipeline/coherent_iq_time_refinement.h>
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
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *coherent_iq_sidecar_schema_version =
    "citlali-coherent-iq-mode-sidecar-v2";
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
    CoherentIqTimeRefinement network_time_refinement;
    CoherentIqSharedTimeRefinement shared_time_refinement;
    CoherentIqModeScore refined_mode_score;
    int cross_network_coincident_count = 0;
    std::string cross_network_coincident_networks;
    std::size_t shared_candidate_index = 0;
};

struct CoherentIqSidecarWorkload {
    std::size_t shared_candidate_count = 0;
    std::size_t network_count = 0;
    std::size_t projected_network_event_scores = 0;
    bool budget_exceeded = false;
};

inline CoherentIqSidecarWorkload plan_coherent_iq_sidecar_workload(
    std::size_t shared_candidate_count, std::size_t network_count,
    std::size_t maximum_network_event_scores) {
    CoherentIqSidecarWorkload result;
    result.shared_candidate_count = shared_candidate_count;
    result.network_count = network_count;
    if (network_count != 0 &&
        shared_candidate_count >
            std::numeric_limits<std::size_t>::max() / network_count) {
        result.projected_network_event_scores =
            std::numeric_limits<std::size_t>::max();
        result.budget_exceeded = true;
        return result;
    }
    result.projected_network_event_scores =
        shared_candidate_count * network_count;
    result.budget_exceeded =
        result.projected_network_event_scores >
        maximum_network_event_scores;
    return result;
}

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

class CoherentIqNetworkRawReader {
  public:
    template <class Calib>
    CoherentIqNetworkRawReader(int network, const std::string &filepath,
                               const Calib &calib)
        : network_{network}, filepath_{filepath},
          file_{filepath, netCDF::NcFile::read},
          recv_var_{file_.getVar("Data.Toltec.RecvTime")},
          is_var_{file_.getVar("Data.Toltec.Is")},
          qs_var_{file_.getVar("Data.Toltec.Qs")},
          tone_var_{file_.getVar("Header.Toltec.ToneFreq")} {
        using namespace netCDF;
        if (recv_var_.isNull() || is_var_.isNull() || qs_var_.isNull() ||
            tone_var_.isNull()) {
            throw citlali::error::io("coherent-IQ observer input lacks "
                                     "required raw-I/Q variables: " +
                                     filepath_);
        }
        recv_time_ = coherent_iq_read_vector(recv_var_);
        const bool invalid_time =
            std::any_of(recv_time_.begin(), recv_time_.end(),
                        [](double value) { return !std::isfinite(value); });
        const bool nonincreasing_time =
            std::adjacent_find(recv_time_.begin(), recv_time_.end(),
                               [](double left, double right) {
                                   return left >= right;
                               }) != recv_time_.end();
        if (recv_time_.size() < 8 || invalid_time || nonincreasing_time) {
            throw citlali::error::io(
                "coherent-IQ observer input has invalid receive time: " +
                filepath_);
        }
        const auto is_dims = is_var_.getDims();
        const auto qs_dims = qs_var_.getDims();
        if (is_dims.size() != 2 || qs_dims.size() != 2) {
            throw citlali::error::io(
                "coherent-IQ observer I/Q variables are not two-dimensional: " +
                filepath_);
        }
        n_tones_ = is_var_.getDim(1).getSize();
        if (is_var_.getDim(0).getSize() != recv_time_.size() ||
            qs_var_.getDim(0).getSize() != is_var_.getDim(0).getSize() ||
            qs_var_.getDim(1).getSize() != n_tones_) {
            throw citlali::error::io(
                "coherent-IQ observer I/Q shapes differ: " + filepath_);
        }

        const auto tone_dims = tone_var_.getDims();
        tone_offsets_.resize(n_tones_);
        if (tone_dims.size() == 2) {
            tone_var_.getVar({0, 0}, {1, n_tones_}, tone_offsets_.data());
        } else if (tone_dims.size() == 1) {
            tone_var_.getVar(tone_offsets_.data());
        } else {
            throw citlali::error::io(
                "coherent-IQ observer tone-frequency shape is unsupported: " +
                filepath_);
        }

        const auto limits = calib.nw_limits.find(network_);
        if (limits == calib.nw_limits.end()) {
            throw citlali::error::runtime(
                "coherent-IQ observer cannot locate network in APT");
        }
        const auto [apt_start, apt_end] = limits->second;
        if (apt_end - apt_start != static_cast<Eigen::Index>(n_tones_)) {
            throw citlali::error::runtime(
                "coherent-IQ observer APT/raw tone counts differ");
        }
        uids_.resize(n_tones_, -1);
        apt_usable_.resize(n_tones_, true);
        for (std::size_t tone = 0; tone < n_tones_; ++tone) {
            const Eigen::Index det =
                apt_start + static_cast<Eigen::Index>(tone);
            uids_[tone] = static_cast<int>(calib.apt.at("uid")(det));
            apt_usable_[tone] = calib.apt.count("flag") == 0 ||
                                calib.apt.at("flag")(det) == 0.0;
        }
    }

    CoherentIqModeScore
    score(const CoherentIqCandidate &candidate,
          const CoherentIqModeTemplate &mode_template,
          const citlali::config::RawTimeChunkCoherentIqModeObserverConfig
              &config) {
        if (candidate.network != network_) {
            throw citlali::error::runtime(
                "coherent-IQ observer candidate/raw network mismatch");
        }

        const double event = candidate.time_unix_sec;
        const auto pre_begin = std::lower_bound(
            recv_time_.begin(), recv_time_.end(),
            event - config.guard_window_sec - config.pre_window_sec);
        const auto pre_end =
            std::lower_bound(recv_time_.begin(), recv_time_.end(),
                             event - config.guard_window_sec);
        const auto post_begin =
            std::upper_bound(recv_time_.begin(), recv_time_.end(),
                             event + config.guard_window_sec);
        const auto post_end = std::upper_bound(
            recv_time_.begin(), recv_time_.end(),
            event + config.guard_window_sec + config.post_window_sec);
        const auto pre_first =
            static_cast<std::size_t>(pre_begin - recv_time_.begin());
        const auto pre_last =
            static_cast<std::size_t>(pre_end - recv_time_.begin());
        const auto post_first =
            static_cast<std::size_t>(post_begin - recv_time_.begin());
        const auto post_last =
            static_cast<std::size_t>(post_end - recv_time_.begin());
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
        std::vector<double> is(n_rows * n_tones_);
        std::vector<double> qs(n_rows * n_tones_);
        is_var_.getVar({read_first, 0}, {n_rows, n_tones_}, is.data());
        qs_var_.getVar({read_first, 0}, {n_rows, n_tones_}, qs.data());

        std::vector<double> phase_change(n_tones_);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t tone = 0; tone < n_tones_; ++tone) {
            std::complex<double> pre_sum{0.0, 0.0};
            std::complex<double> post_sum{0.0, 0.0};
            int pre_count = 0;
            int post_count = 0;
            for (std::size_t raw_row = pre_first; raw_row < pre_last;
                 ++raw_row) {
                const std::size_t row = raw_row - read_first;
                const std::complex<double> value{is[row * n_tones_ + tone],
                                                 qs[row * n_tones_ + tone]};
                if (std::isfinite(value.real()) &&
                    std::isfinite(value.imag())) {
                    pre_sum += value;
                    ++pre_count;
                }
            }
            for (std::size_t raw_row = post_first; raw_row < post_last;
                 ++raw_row) {
                const std::size_t row = raw_row - read_first;
                const std::complex<double> value{is[row * n_tones_ + tone],
                                                 qs[row * n_tones_ + tone]};
                if (std::isfinite(value.real()) &&
                    std::isfinite(value.imag())) {
                    post_sum += value;
                    ++post_count;
                }
            }
            if (!apt_usable_[tone] || pre_count == 0 || post_count == 0 ||
                std::abs(pre_sum) == 0.0) {
                phase_change[tone] = nan;
                continue;
            }
            const auto before = pre_sum / static_cast<double>(pre_count);
            const auto after = post_sum / static_cast<double>(post_count);
            phase_change[tone] = std::arg(after / before) * 1.0e3;
        }
        return score_coherent_iq_mode_event(mode_template, candidate.network,
                                            uids_, tone_offsets_, phase_change);
    }

    CoherentIqTimeRefinement refine_time(
        const CoherentIqCandidate &candidate,
        const CoherentIqModeTemplate &mode_template,
        const citlali::config::RawTimeChunkCoherentIqModeObserverConfig
            &config) {
        CoherentIqTimeRefinement result;
        result.seed_time_unix_sec = candidate.time_unix_sec;
        result.template_tone_count =
            static_cast<int>(mode_template.tones.size());
        if (!config.time_refinement.enabled) {
            result.status = "disabled";
            result.note = "time refinement was not requested";
            return result;
        }
        if (candidate.network != network_ ||
            mode_template.network != network_) {
            result.status = "wrong_network";
            result.note = "candidate, template, and raw network differ";
            return result;
        }

        std::map<int, std::size_t> raw_by_uid;
        for (std::size_t tone = 0; tone < n_tones_; ++tone) {
            if (!apt_usable_[tone] ||
                !std::isfinite(tone_offsets_[tone])) {
                continue;
            }
            if (!raw_by_uid.emplace(uids_[tone], tone).second) {
                result.status = "incompatible_tone_map";
                result.note = "usable raw-tone UIDs are not unique";
                return result;
            }
        }
        std::vector<std::pair<std::size_t, const CoherentIqModeTone *>>
            compatible;
        for (const auto &template_tone : mode_template.tones) {
            const auto found = raw_by_uid.find(template_tone.uid);
            if (found == raw_by_uid.end()) {
                continue;
            }
            const auto raw_tone = found->second;
            if (std::abs(tone_offsets_[raw_tone] -
                         template_tone.tone_offset_frequency_hz) <=
                mode_template.tone_offset_tolerance_hz) {
                compatible.emplace_back(raw_tone, &template_tone);
            }
        }
        result.compatible_tone_count = static_cast<int>(compatible.size());
        const auto minimum_count = std::max<std::size_t>(
            3, mode_template.mode_ids.size() + 1);
        const double compatible_fraction =
            mode_template.tones.empty()
                ? 0.0
                : static_cast<double>(compatible.size()) /
                      static_cast<double>(mode_template.tones.size());
        if (compatible.size() < minimum_count ||
            compatible_fraction <
                mode_template.minimum_compatible_tone_fraction) {
            result.status = "insufficient_compatible_tones";
            result.note =
                "compatible tone coverage is below template requirement";
            return result;
        }

        const auto &refinement = config.time_refinement;
        const double read_padding =
            refinement.search_half_width_sec +
            refinement.smoothing_window_sec;
        const auto read_begin = std::lower_bound(
            recv_time_.begin(), recv_time_.end(),
            candidate.time_unix_sec - read_padding);
        const auto read_end = std::upper_bound(
            recv_time_.begin(), recv_time_.end(),
            candidate.time_unix_sec + read_padding);
        const auto read_first = static_cast<std::size_t>(
            read_begin - recv_time_.begin());
        const auto read_last = static_cast<std::size_t>(
            read_end - recv_time_.begin());
        if (read_last - read_first < 7) {
            result.status = "incomplete_projection_window";
            result.note = "raw file does not span the refinement search";
            return result;
        }
        const std::size_t n_rows = read_last - read_first;
        std::vector<double> is(n_rows * n_tones_);
        std::vector<double> qs(n_rows * n_tones_);
        is_var_.getVar({read_first, 0}, {n_rows, n_tones_}, is.data());
        qs_var_.getVar({read_first, 0}, {n_rows, n_tones_}, qs.data());
        std::vector<double> time(
            recv_time_.begin() + static_cast<std::ptrdiff_t>(read_first),
            recv_time_.begin() + static_cast<std::ptrdiff_t>(read_last));
        const std::size_t n_modes = mode_template.mode_ids.size();
        std::vector<std::vector<double>> numerator(
            n_modes, std::vector<double>(n_rows, 0.0));
        std::vector<std::vector<double>> denominator(
            n_modes, std::vector<double>(n_rows, 0.0));
        std::vector<int> valid_tone_count(n_rows, 0);
        constexpr double two_pi = 2.0 * 3.14159265358979323846;
        for (const auto &[raw_tone, template_tone] : compatible) {
            bool have_previous = false;
            double previous_wrapped = 0.0;
            double unwrap_offset = 0.0;
            for (std::size_t row = 0; row < n_rows; ++row) {
                const std::complex<double> value{
                    is[row * n_tones_ + raw_tone],
                    qs[row * n_tones_ + raw_tone]};
                if (!std::isfinite(value.real()) ||
                    !std::isfinite(value.imag()) ||
                    std::abs(value) == 0.0) {
                    continue;
                }
                const double wrapped = std::arg(value);
                if (have_previous) {
                    const double delta = wrapped - previous_wrapped;
                    if (delta > 3.14159265358979323846) {
                        unwrap_offset -= two_pi;
                    } else if (delta < -3.14159265358979323846) {
                        unwrap_offset += two_pi;
                    }
                }
                previous_wrapped = wrapped;
                have_previous = true;
                const double phase_mrad =
                    (wrapped + unwrap_offset) * 1.0e3;
                ++valid_tone_count[row];
                for (std::size_t mode = 0; mode < n_modes; ++mode) {
                    const double loading = template_tone->loadings[mode];
                    numerator[mode][row] += phase_mrad * loading;
                    denominator[mode][row] += loading * loading;
                }
            }
        }
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::vector<std::vector<double>> projected(
            n_modes, std::vector<double>(n_rows, nan));
        for (std::size_t mode = 0; mode < n_modes; ++mode) {
            for (std::size_t row = 0; row < n_rows; ++row) {
                const double row_fraction =
                    static_cast<double>(valid_tone_count[row]) /
                    static_cast<double>(mode_template.tones.size());
                if (valid_tone_count[row] >=
                        static_cast<int>(minimum_count) &&
                    row_fraction >=
                        mode_template.minimum_compatible_tone_fraction &&
                    denominator[mode][row] > 0.0) {
                    projected[mode][row] =
                        numerator[mode][row] / denominator[mode][row];
                }
            }
        }
        auto refined = refine_coherent_iq_projected_event_time(
            time, mode_template.mode_ids, projected,
            candidate.time_unix_sec, refinement.search_half_width_sec,
            refinement.smoothing_window_sec,
            refinement.minimum_derivative_snr,
            refinement.minimum_peak_ratio,
            refinement.peak_exclusion_sec);
        refined.compatible_tone_count = result.compatible_tone_count;
        refined.template_tone_count = result.template_tone_count;
        return refined;
    }

  private:
    int network_ = -1;
    std::string filepath_;
    netCDF::NcFile file_;
    netCDF::NcVar recv_var_;
    netCDF::NcVar is_var_;
    netCDF::NcVar qs_var_;
    netCDF::NcVar tone_var_;
    std::vector<double> recv_time_;
    std::size_t n_tones_ = 0;
    std::vector<double> tone_offsets_;
    std::vector<int> uids_;
    std::vector<bool> apt_usable_;
};

inline void attach_coherent_iq_cross_network_coincidence(
    std::vector<CoherentIqSidecarRecord> &records) {
    std::map<std::size_t, std::set<int>> networks_by_candidate;
    for (const auto &record : records) {
        if (record.mode_score.status == "scored") {
            networks_by_candidate[record.shared_candidate_index].insert(
                record.candidate.network);
        }
    }
    for (auto &record : records) {
        const auto &networks =
            networks_by_candidate[record.shared_candidate_index];
        record.cross_network_coincident_count =
            static_cast<int>(networks.size());
        record.cross_network_coincident_networks.clear();
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

inline YAML::Node coherent_iq_time_refinement_node(
    const CoherentIqTimeRefinement &refinement) {
    YAML::Node node;
    node["status"] = refinement.status;
    node["method"] = refinement.method;
    auto assign_finite = [&node](const char *name, double value) {
        node[name] = std::isfinite(value) ? YAML::Node(value) : YAML::Node();
    };
    assign_finite("seed_time_unix_sec", refinement.seed_time_unix_sec);
    assign_finite(
        "refined_time_unix_sec", refinement.refined_time_unix_sec);
    assign_finite(
        "displacement_from_seed_sec",
        refinement.displacement_from_seed_sec);
    node["primary_mode_id"] = refinement.primary_mode_id;
    assign_finite(
        "peak_absolute_derivative_mrad_per_sec",
        refinement.peak_absolute_derivative_mrad_per_sec);
    assign_finite("derivative_snr", refinement.derivative_snr);
    assign_finite(
        "peak_to_second_ratio", refinement.peak_to_second_ratio);
    node["compatible_tone_count"] = refinement.compatible_tone_count;
    node["template_tone_count"] = refinement.template_tone_count;
    node["note"] = refinement.note;
    return node;
}

inline YAML::Node coherent_iq_shared_time_refinement_node(
    const CoherentIqSharedTimeRefinement &refinement) {
    YAML::Node node;
    node["status"] = refinement.status;
    node["method"] = refinement.method;
    auto assign_finite = [&node](const char *name, double value) {
        node[name] = std::isfinite(value) ? YAML::Node(value) : YAML::Node();
    };
    assign_finite("seed_time_unix_sec", refinement.seed_time_unix_sec);
    assign_finite(
        "refined_time_unix_sec", refinement.refined_time_unix_sec);
    assign_finite(
        "displacement_from_seed_sec",
        refinement.displacement_from_seed_sec);
    node["contributing_network_count"] =
        refinement.contributing_network_count;
    node["contributing_networks"] = refinement.contributing_networks;
    assign_finite(
        "contributing_network_span_sec",
        refinement.contributing_network_span_sec);
    node["note"] = refinement.note;
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
    const auto workload = plan_coherent_iq_sidecar_workload(
        candidates.size(), raw_files.size(),
        static_cast<std::size_t>(config.max_network_event_scores));
    logger->info(
        "observe-only coherent-IQ workload: {} network seeds, {} shared "
        "candidates, {} raw networks, {} projected network-event scores "
        "(budget {})",
        network_candidates.size(), candidates.size(), raw_files.size(),
        workload.projected_network_event_scores,
        config.max_network_event_scores);

    std::vector<CoherentIqSidecarRecord> records;
    std::vector<std::vector<std::pair<int, CoherentIqTimeRefinement>>>
        network_refinements_by_candidate(candidates.size());
    std::vector<CoherentIqSharedTimeRefinement> shared_refinements(
        candidates.size());
    std::size_t raw_network_files_opened = 0;
    std::size_t completed_refinements = 0;
    std::size_t completed_refined_scores = 0;
    if (!workload.budget_exceeded) {
        records.reserve(workload.projected_network_event_scores);
        std::size_t completed_scores = 0;
        for (const auto &[network, filepath] : raw_files) {
            const auto template_it = templates.find(network);
            logger->info(
                "observe-only coherent-IQ scoring network {}: {} shared "
                "candidates, template {}",
                network, candidates.size(),
                template_it == templates.end() ? "unavailable" : "loaded");
            std::unique_ptr<CoherentIqNetworkRawReader> reader;
            if (template_it != templates.end() && !candidates.empty()) {
                reader = std::make_unique<CoherentIqNetworkRawReader>(
                    network, filepath, engine.calib);
                ++raw_network_files_opened;
            }
            for (std::size_t candidate_index = 0;
                 candidate_index < candidates.size(); ++candidate_index) {
                auto network_candidate = candidates[candidate_index];
                network_candidate.network = network;
                CoherentIqModeScore score;
                score.network = network;
                CoherentIqTimeRefinement time_refinement;
                time_refinement.seed_time_unix_sec =
                    network_candidate.time_unix_sec;
                CoherentIqModeScore refined_score;
                refined_score.network = network;
                if (template_it == templates.end()) {
                    score.status = "template_unavailable";
                    score.compatibility_note =
                        "no configured template for this present network";
                    time_refinement.status = "template_unavailable";
                    time_refinement.note =
                        "no configured template for this present network";
                    refined_score.status = "template_unavailable";
                    refined_score.compatibility_note =
                        "no configured template for this present network";
                } else {
                    score = reader->score(network_candidate,
                                          template_it->second, config);
                    time_refinement = reader->refine_time(
                        network_candidate, template_it->second, config);
                    if (config.time_refinement.enabled) {
                        ++completed_refinements;
                    }
                    refined_score.template_id =
                        template_it->second.template_id;
                    refined_score.template_version =
                        template_it->second.template_version;
                    refined_score.template_tone_count = static_cast<int>(
                        template_it->second.tones.size());
                    refined_score.status =
                        config.time_refinement.enabled
                            ? "shared_time_refinement_unavailable"
                            : "time_refinement_disabled";
                    refined_score.compatibility_note =
                        config.time_refinement.enabled
                            ? "shared refined event time is not yet available"
                            : "time refinement was not requested";
                }
                CoherentIqSidecarRecord record;
                record.candidate = std::move(network_candidate);
                record.mode_score = std::move(score);
                record.network_time_refinement =
                    std::move(time_refinement);
                record.refined_mode_score = std::move(refined_score);
                record.shared_candidate_index = candidate_index;
                if (config.time_refinement.enabled) {
                    network_refinements_by_candidate[candidate_index]
                        .emplace_back(
                            network, record.network_time_refinement);
                }
                records.push_back(std::move(record));
                ++completed_scores;
                if (config.progress_interval_scores > 0 &&
                    (completed_scores % static_cast<std::size_t>(
                                            config.progress_interval_scores) ==
                         0 ||
                     completed_scores ==
                         workload.projected_network_event_scores)) {
                    logger->info("observe-only coherent-IQ progress: {}/{} "
                                 "network-event scores",
                                 completed_scores,
                                 workload.projected_network_event_scores);
                }
            }
            logger->info("observe-only coherent-IQ finished network {} ({}/{} "
                         "network-event scores complete)",
                         network, completed_scores,
                         workload.projected_network_event_scores);
        }
        if (config.time_refinement.enabled) {
            for (std::size_t candidate_index = 0;
                 candidate_index < candidates.size(); ++candidate_index) {
                shared_refinements[candidate_index] =
                    consolidate_coherent_iq_time_refinements(
                        candidates[candidate_index].time_unix_sec,
                        network_refinements_by_candidate[candidate_index],
                        config.time_refinement.minimum_networks,
                        config.time_refinement.consensus_tolerance_sec);
            }
            for (auto &record : records) {
                record.shared_time_refinement =
                    shared_refinements[record.shared_candidate_index];
            }

            for (const auto &[network, filepath] : raw_files) {
                const auto template_it = templates.find(network);
                if (template_it == templates.end() || candidates.empty()) {
                    continue;
                }
                const bool has_refined_candidate = std::any_of(
                    shared_refinements.begin(), shared_refinements.end(),
                    [](const auto &refinement) {
                        return refinement.status == "refined";
                    });
                if (!has_refined_candidate) {
                    continue;
                }
                CoherentIqNetworkRawReader reader(
                    network, filepath, engine.calib);
                ++raw_network_files_opened;
                for (auto &record : records) {
                    if (record.candidate.network != network ||
                        record.shared_time_refinement.status != "refined") {
                        if (record.candidate.network == network &&
                            record.shared_time_refinement.status !=
                                "refined") {
                            record.refined_mode_score.status =
                                "shared_time_refinement_" +
                                record.shared_time_refinement.status;
                            record.refined_mode_score.compatibility_note =
                                record.shared_time_refinement.note;
                        }
                        continue;
                    }
                    auto refined_candidate = record.candidate;
                    refined_candidate.time_unix_sec =
                        record.shared_time_refinement
                            .refined_time_unix_sec;
                    record.refined_mode_score = reader.score(
                        refined_candidate, template_it->second, config);
                    ++completed_refined_scores;
                }
            }
        } else {
            for (std::size_t candidate_index = 0;
                 candidate_index < candidates.size(); ++candidate_index) {
                shared_refinements[candidate_index].status = "disabled";
                shared_refinements[candidate_index].seed_time_unix_sec =
                    candidates[candidate_index].time_unix_sec;
                shared_refinements[candidate_index].note =
                    "time refinement was not requested";
            }
            for (auto &record : records) {
                record.shared_time_refinement =
                    shared_refinements[record.shared_candidate_index];
            }
        }
        attach_coherent_iq_cross_network_coincidence(records);
        std::sort(records.begin(), records.end(),
                  [](const auto &left, const auto &right) {
                      return std::tie(left.shared_candidate_index,
                                      left.candidate.network) <
                             std::tie(right.shared_candidate_index,
                                      right.candidate.network);
                  });
    } else {
        logger->warn(
            "observe-only coherent-IQ scoring skipped: {} projected "
            "network-event scores exceed configured budget {}. Required "
            "science products remain unaffected.",
            workload.projected_network_event_scores,
            config.max_network_event_scores);
    }

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
    root["requested"]["max_network_event_scores"] =
        config.max_network_event_scores;
    root["requested"]["progress_interval_scores"] =
        config.progress_interval_scores;
    auto refinement_request =
        root["requested"]["time_refinement"];
    refinement_request["enabled"] =
        config.time_refinement.enabled;
    refinement_request["search_half_width_sec"] =
        config.time_refinement.search_half_width_sec;
    refinement_request["smoothing_window_sec"] =
        config.time_refinement.smoothing_window_sec;
    refinement_request["minimum_derivative_snr"] =
        config.time_refinement.minimum_derivative_snr;
    refinement_request["minimum_peak_ratio"] =
        config.time_refinement.minimum_peak_ratio;
    refinement_request["peak_exclusion_sec"] =
        config.time_refinement.peak_exclusion_sec;
    refinement_request["minimum_networks"] =
        config.time_refinement.minimum_networks;
    refinement_request["consensus_tolerance_sec"] =
        config.time_refinement.consensus_tolerance_sec;
    root["observer_execution"]["status"] =
        workload.budget_exceeded ? "skipped_workload_budget" : "completed";
    root["observer_execution"]["workload_budget_exceeded"] =
        workload.budget_exceeded;
    root["observer_execution"]["projected_network_event_score_count"] =
        workload.projected_network_event_scores;
    root["observer_execution"]["processed_network_event_score_count"] =
        records.size();
    root["observer_execution"]["processed_network_time_refinement_count"] =
        completed_refinements;
    root["observer_execution"]["processed_refined_network_event_score_count"] =
        completed_refined_scores;
    root["observer_execution"]["raw_network_files_opened"] =
        raw_network_files_opened;
    root["observer_execution"]["raw_time_vectors_read"] =
        raw_network_files_opened;
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
        node["network_time_refinement"] =
            coherent_iq_time_refinement_node(
                record.network_time_refinement);
        node["shared_time_refinement"] =
            coherent_iq_shared_time_refinement_node(
                record.shared_time_refinement);
        node["refined_mode_score"] =
            coherent_iq_score_node(record.refined_mode_score);
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
    root["realized"]["projected_network_event_score_count"] =
        workload.projected_network_event_scores;
    root["realized"]["raw_network_files_opened"] =
        raw_network_files_opened;
    root["realized"]["network_time_refinement_count"] =
        completed_refinements;
    root["realized"]["shared_time_refined_candidate_count"] =
        static_cast<int>(std::count_if(
            shared_refinements.begin(), shared_refinements.end(),
            [](const auto &refinement) {
                return refinement.status == "refined";
            }));
    root["realized"]["refined_scored_count"] =
        completed_refined_scores;
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
        "{} network scores, {} templates, {} raw networks, status {})",
        output_path.string(), candidates.size(), records.size(),
        templates.size(), raw_files.size(),
        workload.budget_exceeded ? "skipped_workload_budget" : "completed");
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
