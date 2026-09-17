#pragma once
// Offline serialization only. Runtime diagnostic authorities live in core.
#include <citlali/core/pipeline/timestream_rtc_common_mode.h>
#include <iomanip>
namespace {
void health_fit_csv(const fs::path &path,
                    const std::vector<RtcCommonModeFit> &fits) {
  std::ofstream f(path);
  f << std::setprecision(17);
  f << "detector,interval,first,past_last,cause,gain,offset,correlation,"
       "residual_scatter,reference_scatter,relative_gain,iterations\n";
  for (const auto &a : fits)
    f << a.detector << ',' << a.interval << ',' << a.rows.first << ','
      << a.rows.past_last << ',' << int(a.cause) << ',' << a.gain << ','
      << a.offset << ',' << a.correlation << ',' << a.residual_scatter << ','
      << a.reference_scatter << ',' << a.calibrated_relative_gain << ','
      << a.iterations << '\n';
  f.close();
  require(bool(f), "health fit export failed");
}
void export_health(
    const fs::path &output, std::shared_ptr<const RtcCommonModeEvidence> health,
    std::shared_ptr<const RtcLinePowerEvidence> lines,
    const std::vector<int> &channels, const YAML::Node &cfg,
    const fs::path &config_path, double learning_seconds,
    const std::vector<std::uint32_t> *fixed_inspections = nullptr) {
  const auto start = std::chrono::steady_clock::now();
  auto consideration = RtcCommonModeConsideration::compare(health, lines);
  const auto considered = std::chrono::steady_clock::now();
  const auto checks = health->self_excluded_checks(
      fixed_inspections ? *fixed_inspections
                        : consideration.inspection_targets());
  const auto checked = std::chrono::steady_clock::now();
  fs::create_directories(output);
  health_fit_csv(output / "fits.csv", health->fits());
  std::ofstream summary(output / "detectors.csv");
  summary << std::setprecision(17);
  summary
      << "detector,channel,reference_population,flxscale,fitted_rows,available_"
         "segments,unavailable_segments,gain,gain_mad,relative_gain,negative_"
         "relative_fraction,correlation,residual_scatter,original_candidates,"
         "unavailable_noise_blocks,spectral_peak_hz,spectral_excess_fraction\n";
  for (const auto &s : consideration.detectors())
    summary << s.detector << ',' << channels[s.detector] << ','
            << health->domain().members[s.detector].reference_eligible << ','
            << health->domain().members[s.detector].flxscale << ','
            << s.fitted_rows << ',' << s.available_segments << ','
            << s.unavailable_segments << ',' << s.gain << ',' << s.gain_mad
            << ',' << s.relative_gain << ',' << s.negative_relative_fraction
            << ',' << s.correlation << ',' << s.residual_scatter << ','
            << s.original_candidates << ',' << s.unavailable_noise_blocks << ','
            << s.spectral_peak_hz << ',' << s.spectral_excess_fraction << '\n';
  summary.close();
  require(bool(summary), "health summary export failed");
  const auto &net = health->original_handle()->input_handle()->network(
      health->domain().network);
  const auto &axis = net.occurrence_axis();
  const auto first = axis.first_native_row();
  std::ofstream ref(output / "reference.f64", std::ios::binary);
  // Native time, median candidate reference, speed-eligible bit. Never compact.
  for (auto row = first; row < axis.past_last_native_row(); ++row) {
    const double values[]{
        axis.native_identity(row).reconstructed_time_unix_sec(),
        health->reference()[row - first],
        double(health->speed_admitted()[row - first])};
    ref.write(reinterpret_cast<const char *>(values), sizeof(values));
  }
  ref.close();
  require(bool(ref), "health reference export failed");
  YAML::Node intervals(YAML::NodeType::Sequence);
  for (const auto &s : health->intervals()) {
    YAML::Node a;
    a["scan"] = s.scan;
    a["rows"] = range(s.rows);
    a["contributors"] = s.contributors;
    a["speed_eligible_rows"] = s.speed_eligible_rows;
    a["reference_scatter"] = s.reference_scatter;
    for (std::size_t d = 0; d < channels.size(); ++d) {
      a["reference_reasons"].push_back(int(s.reference_reasons[d]));
      a["baseline"].push_back(s.baseline[d]);
      a["original_paired_rows"].push_back(s.paired_rows[d]);
      a["eligible_target_rows"].push_back(s.target_eligible_rows[d]);
    }
    intervals.push_back(a);
  }
  write_yaml(output / "intervals.yaml", intervals);
  YAML::Node receipt;
  receipt["source_revision"] = std::string(CITLALI_GIT_REVISION);
  receipt["configuration_sha256"] = citlali::utils::sha256_file(config_path);
  receipt["configuration"] = cfg;
  receipt["policy"] = std::string(RtcCommonModePolicy::identity);
  receipt["VAL_generation"] =
      health->original_handle()->val_snapshot_handle()->generation().value;
  receipt["attempt"] = health->attempt();
  receipt["original_stage"] = "original-native-x-paired-valid-initial-VAL";
  receipt["original_parent"] = net.mapping_authority().paired_xr_record_id;
  receipt["processing_generation"] =
      health->domain().scans->processing_generation();
  receipt["membership_authority"] = health->domain().population_authority;
  receipt["AST_source"] = health->domain()
                              .motion->raw_product_handle()
                              ->source_handle()
                              ->metadata()
                              .source_artifact_identity;
  receipt["sampling_speed_limit"] = health->sampling_speed_limit();
  receipt["formal_uncertainty"] =
      "unavailable;segment-MAD-is-empirical-variation-not-standard-error";
  receipt["reference"] = "timewise-median-of-per-interval-median-centered-"
                         "original-x;fixed-contributors-per-interval";
  receipt["fit"] = "free-signed-affine;Huber1.345;MAD1.4826;IRLS<=256;relative-"
                   "convergence1e-8;minimum64";
  receipt["relative_gain"] =
      "flxscale*free-gain / same-interval equal-detector-weight "
      "calibrated-reference-peer median";
  receipt["reference_reason_bits"] =
      "1=population;2=missing-original-pair;4=unresolved-candidate-edge;8="
      "paired-noise-unavailable;16=insufficient-samples";
  receipt["membership_changes"] =
      "allowed-only-at-recorded-interval-boundaries;never-crossed-by-a-fit";
  receipt["residual_product"] =
      "immutable-original-alias+exact-reference+fit-support/gain/"
      "offset;selected-targets-also-materialized";
  receipt["rejection_authorized"] = false;
  receipt["science_path_changed"] = false;
  receipt["observation_role"] =
      (health->domain().network == 0
           ? "SCIENCE-NGC4449-owner-identified-network0-example"
           : "SCIENCE-NGC4449-development-proxy-network12");
  if (cfg["common_mode_census"])
    receipt["observation_role"] =
        "SCIENCE-NGC4449-bounded-repeatability-census";
  receipt["source_protection_authority"] =
      health->original_handle()->protection_handle()->authority_id();
  receipt["learn_seconds"] = learning_seconds;
  receipt["consider_seconds"] =
      std::chrono::duration<double>(considered - start).count();
  receipt["self_check_seconds"] =
      std::chrono::duration<double>(checked - considered).count();
  receipt["logical_evidence_bytes"] = health->logical_owned_bytes();
  receipt["logical_self_check_bytes"] =
      checks.size() * health->reference().size() * sizeof(double);
  for (const auto &check : checks) {
    const auto d = check.detector;
    receipt["self_excluded_channels"].push_back(channels[d]);
    const auto suffix = std::to_string(channels[d]);
    health_fit_csv(output / ("self-excluded-" + suffix + ".csv"), check.fits);
    Eigen::Matrix<double, Eigen::Dynamic, 4> values(axis.occurrence_count(), 4);
    values.setConstant(NAN);
    // original x, self-excluded model, residual, self-excluded reference.
    for (auto row = first; row < axis.past_last_native_row(); ++row) {
      values(row - first, 0) = net.value(NativeReadoutCoordinate::x, row, d);
      values(row - first, 3) = check.reference[row - first];
    }
    for (const auto &f : check.fits)
      if (f.available())
        for (auto row = f.rows.first; row < f.rows.past_last; ++row) {
          values(row - first, 1) =
              f.offset + f.gain * check.reference[row - first];
          values(row - first, 2) =
              values(row - first, 0) - values(row - first, 1);
        }
    write_matrix(output / ("diagnostic-" + suffix + ".f64"), values);
  }
  receipt["export_seconds"] =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - checked)
          .count();
  write_yaml(output / "receipt.yaml", receipt);
}

// Validation-only composition check: the existing RTC owner still constructs
// every reference and fit. No runtime state or estimator variant is introduced.
void export_census_reference_checks(
    const fs::path &output, std::shared_ptr<const RtcCommonModeEvidence> health,
    std::shared_ptr<const RtcLinePowerEvidence> lines,
    const std::vector<int> &channels, const YAML::Node &cfg,
    const fs::path &config_path, const std::vector<std::uint32_t> &targets) {
  std::vector<std::uint32_t> peers;
  for (std::uint32_t d = 0; d < channels.size(); ++d)
    if (health->domain().members[d].reference_eligible &&
        std::find(targets.begin(), targets.end(), d) == targets.end())
      peers.push_back(d);
  const std::vector<std::uint32_t> no_checks;
  for (std::size_t half = 0; half < 2; ++half) {
    auto domain = health->domain();
    for (auto &member : domain.members)
      member.reference_eligible = false;
    YAML::Node partition;
    for (std::size_t i = half; i < peers.size(); i += 2) {
      domain.members[peers[i]].reference_eligible = true;
      partition["eligible_channels"].push_back(channels[peers[i]]);
    }
    domain.population_authority +=
        ":bounded-census-alternating-rank-half=" + std::to_string(half);
    const auto started = std::chrono::steady_clock::now();
    auto evidence = RtcCommonModeEvidence::learn(health->original_handle(),
                                                 std::move(domain), 31 + half);
    const auto path = output / ("partition-" + std::to_string(half));
    export_health(path, evidence, lines, channels, cfg, config_path,
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - started)
                      .count(),
                  &no_checks);
    partition["method"] =
        "alternating-rank-in-initial-eligible-sorted-channel-inventory";
    partition["full_network_leave_one_out"] = false;
    for (const auto d : targets)
      partition["excluded_targets"].push_back(channels[d]);
    write_yaml(path / "partition.yaml", partition);
  }
}
} // namespace
