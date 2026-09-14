// Bounded offline RTC experiment; no production route or output admission.
#define main unused_identity_acceptance_main
#include "identity_route_acceptance.cpp"
#undef main
#include <bit>
#include <citlali/core/pipeline/timestream_rtc_notch_recovery.h>
namespace {
using namespace citlali::pipeline;
void write_pair(const fs::path &p, const auto &v) {
  std::ofstream o(p, std::ios::binary);
  require(bool(o), "cannot write paired output");
  for (Eigen::Index i = 0; i < v.rows(); ++i)
    for (int c = 0; c < 2; ++c) {
      double x = v(i, c);
      o.write(reinterpret_cast<const char *>(&x), 8);
    }
  o.close();
  require(bool(o), "paired output close failed");
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 3,
            "expected exact experiment JSON and new output directory");
    const auto cfg = YAML::LoadFile(argv[1]);
    const fs::path output = argv[2];
    require(!fs::exists(output), "preserve previous experiment output");
    auto checked = [&](const std::string &key) {
      const fs::path p = cfg[key]["path"].as<std::string>();
      require(citlali::utils::sha256_file(p) ==
                  cfg[key]["sha256"].as<std::string>(),
              "changed input: " + key);
      return p;
    };
    const auto raw_path = checked("raw"), tune_path = checked("tune"),
               manifest = checked("manifest");
    const auto samples_path = checked("samples"),
               prior_receipt_path = checked("audit_receipt");
    const auto channel = cfg["channel"].as<int>();
    const auto prior = YAML::LoadFile(prior_receipt_path.string());
    require(prior["raw_sha256"].as<std::string>() ==
                    citlali::utils::sha256_file(raw_path) &&
                prior["tune_sha256"].as<std::string>() ==
                    citlali::utils::sha256_file(tune_path) &&
                prior["manifest_sha256"].as<std::string>() ==
                    citlali::utils::sha256_file(manifest),
            "audit raw/Tune/APT binding mismatch");
    require(samples_path.filename() ==
                "samples-" + std::to_string(channel) + ".f64",
            "audit column filename mismatch");
    auto [logger, logs] = configure_logging();
    const auto verified = apt::verify_bundle_filesystem(manifest, true);
    const auto relation =
        pipeline::admit_canonical_apt_detector_relation_v2(verified);
    netCDF::NcFile raw_file(raw_path.string(), netCDF::NcFile::read);
    const auto nw =
        read_netcdf_scalar<int>(raw_file, "Header.Toltec.RoachIndex");
    const auto obs = read_netcdf_scalar<int>(raw_file, "Header.Toltec.ObsNum");
    const auto sub =
        read_netcdf_scalar<int>(raw_file, "Header.Toltec.SubObsNum");
    const auto scan =
        read_netcdf_scalar<int>(raw_file, "Header.Toltec.ScanNum");
    require(obs == relation.observation().observation &&
                sub == relation.observation().subobservation &&
                scan == relation.observation().scan,
            "raw observation differs from exact APT relation");
    const auto found = std::find_if(
        relation.raw_sources().begin(), relation.raw_sources().end(),
        [&](const auto &s) { return s.network == nw; });
    require(found != relation.raw_sources().end(),
            "APT does not bind input network");
    const auto source = *found;
    verify_raw_file(raw_path, source);
    const auto rows = static_cast<std::int64_t>(
        raw_file.getVar("Data.Toltec.Is").getDim(0).getSize());
    const auto tune_digest = "sha256:" + citlali::utils::sha256_file(tune_path);
    const auto kmp = std::find_if(
        verified.sources.begin(), verified.sources.end(), [&](const auto &s) {
          return s.role == apt::SourceRole::kmp && s.network == nw;
        });
    require(kmp != verified.sources.end() &&
                kmp->content_sha256 == tune_digest &&
                kmp->byte_count == fs::file_size(tune_path),
            "Tune report differs from exact APT KMP source binding");
    const auto tune = read_tune_facts(tune_path, source.channel_count);
    require(tune.observation ==
                    read_netcdf_scalar<int>(raw_file,
                                            "Header.Toltec.TargSweepObsNum") &&
                tune.subobservation ==
                    read_netcdf_scalar<int>(
                        raw_file, "Header.Toltec.TargSweepSubObsNum") &&
                tune.scan == read_netcdf_scalar<int>(
                                 raw_file, "Header.Toltec.TargSweepScanNum") &&
                tune.network == nw,
            "Tune report differs from raw header's calibration relation");
    double fpga = 0, hz = 0;
    std::int64_t accum = 0;
    auto ts = read_timestamp_slice(raw_path, 0, rows, fpga, hz, accum);
    require(fpga > 0 && hz > 0 && accum > 0 && tune.accumulation_length > 0,
            "invalid producer cadence");
    const double duration = static_cast<double>(accum) / fpga;
    require(std::abs(duration - 1 / hz) <=
                8 * std::numeric_limits<double>::epsilon() * duration,
            "producer cadence fields disagree");
    // Native producer clock only; no telescope synchronization or cross-network
    // timing claim.
    auto timing = std::make_shared<const pipeline::NativeNetworkAlignment>(
        pipeline::make_native_network_alignment(nw, 0, ts, fpga, 0.0));
    NetworkInput input{
        source,     raw_path, tune_path, citlali::utils::sha256_file(tune_path),
        fpga,       hz,       accum,     tune.accumulation_length,
        tune.valid, timing};
    require(rows == prior["rows"].as<std::int64_t>() && channel >= 0 &&
                channel < source.channel_count,
            "audit shape mismatch");
    require(fs::file_size(samples_path) ==
                static_cast<std::uintmax_t>(rows) * 32,
            "audit sample byte count mismatch");
    NativePairedReadoutMatrix x(rows, 1), r(rows, 1);
    std::vector<NativeReadoutCoordinateState> xs, rs;
    std::ifstream f(samples_path, std::ios::binary);
    for (std::int64_t i = 0; i < rows; ++i) {
      std::array<double, 4> a;
      f.read(reinterpret_cast<char *>(a.data()), 32);
      require(bool(f), "audit sample read failed");
      x(i, 0) = a[0];
      r(i, 0) = a[1];
      require((a[2] == 0 || a[2] == 1) && (a[3] == 0 || a[3] == 1),
              "audit state malformed");
      xs.push_back(NativeReadoutCoordinateState::measured(
          true, tune.valid[channel] && std::isfinite(a[0]), true,
          std::isfinite(a[0])));
      rs.push_back(NativeReadoutCoordinateState::measured(
          true, tune.valid[channel] && std::isfinite(a[1]), true,
          std::isfinite(a[1])));
      require(xs.back().valid() == (a[2] != 0) &&
                  rs.back().valid() == (a[3] != 0),
              "export differs from exact producer state convention");
    }
    const auto original_x = x, original_r = r;
    auto config = load_runtime_config(checked("effective_config"));
    require(config.interface_offset_present[nw] &&
                config.interface_offsets_sec[nw] == 0,
            "audit and AST timing require the accepted zero network offset");
    const auto aptrow = std::find_if(
        verified.apt.rows.begin(), verified.apt.rows.end(),
        [&](const auto &r) { return r.network == nw && r.channel == channel; });
    require(aptrow != verified.apt.rows.end() && aptrow->array == 2,
            "experiment beam must bind exact a2000 APT row");
    auto mapping = std::make_shared<pipeline::NativeReadoutMappingAuthority>(
        *mapping_identity(input, config));
    mapping->applicability_domain_id =
        "observation=" + std::to_string(obs) + ":network=" + std::to_string(nw);
    mapping->event_time_epoch_meaning_id =
        "producer-native-clock+exact-effective-zero-interface-offset:accepted-"
        "AST-mapping";
    mapping->paired_xr_record_id +=
        ":exact-original-column-projection:sha256:" +
        citlali::utils::sha256_file(samples_path);
    mapping->timing_uncertainty_state_id =
        "unquantified:uniform-average-center-trial:rtc-native-readout-uniform-"
        "average-assumption-v1";
    // The owner-approved uniform-average assumption remains provisional.
    // AST uses the recovered effective zero interface offset; the readout
    // midpoint/boxcar remains the explicit provisional owner assumption.
    auto axis = occurrence_axis(input, 0, rows,
                                NativeEventTimeRole::integration_center);
    const auto runs = axis->contiguous_runs();
    auto detectors = detector_axis(relation, input);
    auto detector_binding = detectors.at(channel);
    detector_binding.storage_column = 0;
    std::vector<NativePairedReadoutNetwork> networks;
    networks.push_back(NativePairedReadoutNetwork::admit(
        axis, {detector_binding}, mapping, std::move(x), std::move(r),
        std::move(xs), std::move(rs)));
    auto parent = std::make_shared<const NativePairedReadoutObservation>(
        NativePairedReadoutObservation::admit(
            NativeObservationScope{obs, sub, scan}, {nw}, std::move(networks)));
    auto val = ValSnapshot::initial(parent);
    auto view = NativePairedReadoutView::full(parent);
    auto protection = RtcSpikeSourceProtection::admit(
        parent, "audit-source-status-unknown-retained",
        RtcSpikeProtection::unavailable);
    auto spikes = learn_rtc_spike_candidates(view, val, protection, 1);
    auto peer = RtcEventPeerPopulation::admit(
        spikes, "single-projected-detector:no-ensemble-inference",
        {{nw, 0, detector_binding.detector_occurrence_id, true}});
    auto events = RtcEventAssessmentDecision::consider(
        learn_rtc_event_assessment(spikes, peer, 2), val, 3);
    auto amplitude = RtcJumpAmplitudeDecision::consider(events, val, 4);
    auto consistency = RtcJumpConsistencyDecision::consider(
        RtcJumpConsistencyEvidence::learn(amplitude, 5), val, 6);
    auto transition = RtcJumpTransitionEvidence::learn(
        RtcJumpTransitionRequest::consider(consistency, val, 7), 8);
    auto support = RtcJumpSupportEvidence::learn(transition, 9);
    auto refit = RtcJumpRefitEvidence::learn(
        RtcJumpRefitRequest::consider(support, val, 10), 11);
    auto remeasurement = RtcJumpReassessmentEvidence::learn(
        RtcJumpRemeasureRequest::consider(refit, val, 12), 13);
    auto admitted = RtcJumpAdmissionDecision::consider(
        RtcJumpReassessmentDecision::consider(remeasurement, val, 14), val, 15);
    // Missing exact existing scan relation remains missing. If any jump
    // were admitted, this throws instead of inventing a scan partition.
    auto jumps = RtcJumpExclusionPlan::consider(admitted, nullptr, val, 16);
    auto transients = RtcTransientExclusionPlan::consider(
        events->original_screening_handle(), jumps, val, 17);
    auto native =
        ValNativeRealization::create(parent, {ValProducer::align, 1}, 1,
                                     ValNativeProductRole::original_input, nw);
    auto identity = RtcSpectralInputIdentity::bind(
        native, val, view->span(nw), RtcSpectralInputStage::original_reference,
        "exact-audit-original-projection", 1);
    auto spectral = RtcNativeSpectralEvidence::learn_initial(
        spikes, {identity},
        {{nw, "audit-four-epoch-ULP-arithmetic-envelope", duration,
          prior["roundoff_bound_fraction"].as<double>()}},
        18);
    auto lines = RtcLinePowerEvidence::learn(
        spectral, val, RtcLinePowerProfile::initial_2_hz, 19);
    auto joint = RtcLinePowerConsideration::rank(
        lines,
        RtcSpectralTransientConsideration::consider(spectral, val, events, val,
                                                    20),
        21);
    const auto telescope =
        load_telescope(checked("telescope"), parent->scope());
    auto ast =
        build_ast_scan_motion_product(telescope.source, ast_identity_binding);
    auto motion = AstScanMotionNetworkView::admit(ast, timing);
    fs::create_directories(output);
    std::ofstream geometry(output / "geometry.f64", std::ios::binary);
    const auto &tel = *telescope.source;
    for (std::int64_t i = 0; i < rows; ++i) {
      auto v = motion->scalar_speed_arcsec_per_sec(i);
      auto support = motion->support(i);
      double ra = NAN, dec = NAN;
      if (support) {
        auto a = tel.local_index(support->lower_source_record.record),
             b = tel.local_index(support->upper_source_record.record);
        ra = support->lower_weight * tel.source_ra_act_rad()[a] +
             support->upper_weight * tel.source_ra_act_rad()[b];
        dec = support->lower_weight * tel.source_dec_act_rad()[a] +
              support->upper_weight * tel.source_dec_act_rad()[b];
      }
      const std::array<double, 4> record{
          axis->native_identity(i).reconstructed_time_unix_sec(),
          v.value_or(NAN), ra, dec};
      geometry.write(reinterpret_cast<const char *>(record.data()), 32);
    }
    geometry.close();
    require(bool(geometry), "geometry output failed");
    Eigen::Matrix<double, Eigen::Dynamic, 2> originals(rows, 2);
    originals.col(0) = original_x;
    originals.col(1) = original_r;
    write_pair(output / "original.f64", originals);
    std::ofstream psd(output / "initial-psd.f64", std::ios::binary);
    for (const auto &s : spectral->spectra())
      psd.write(reinterpret_cast<const char *>(s.psd.data()), s.psd.size() * 8);
    psd.close();
    require(bool(psd), "PSD output failed");
    std::ofstream summary(output / "receipt.json");
    summary << "{\"source_revision\":"
            << std::quoted(std::string(CITLALI_GIT_REVISION))
            << ",\"network\":" << nw << ",\"channel\":" << channel
            << ",\"rows\":" << rows
            << ",\"VAL_generation\":0,\"transient_excluded_cells\":"
            << transients->counts().union_pair_cells
            << ",\"source_status\":\"unknown-retained\",\"trials\":[";
    int trial_index = 0;
    for (const auto &trial : cfg["trials"]) {
      RtcLineTransferSpecification s;
      s.identity = trial["id"].as<std::string>();
      s.lowpass_identity = cfg["lowpass_identity"].as<std::string>();
      s.state_support_identity =
          "mature-odd-reflect-min9;constant-endpoint;forward-reverse;guard-"
          "pole1e-6;FIR-valid;fixed-native-phase0";
      s.input_interval_seconds = spectral->network(nw).interval_seconds;
      s.factor = 2;
      s.centered_lowpass = cfg["fir"].as<std::vector<double>>();
      std::size_t guard = 0;
      if (trial["notch"].as<bool>()) {
        timestream::Filter f;
        f.w0s = {cfg["notch_hz"].as<double>()};
        f.qs = {f.w0s[0] / .5};
        f.make_notch_filter(1 / s.input_interval_seconds);
        RtcNotchResponseSection n;
        n.identity = "explicit-original-line-center-width0.5Hz";
        for (int j = 0; j < 3; ++j) {
          n.a[j] = f.notch_a[0][j];
          n.b[j] = f.notch_b[0][j];
        }
        s.notches.push_back(n);
        guard = f.notch_settle_samples(1 / s.input_interval_seconds, 1e-6);
      }
      auto candidate = RtcLineTransferCandidate::bind(lines, nw, 0, s);
      auto assessment =
          RtcLineTransferAssessment::consider(candidate, joint, val, 22);
      RtcNotchRecoveryDomain domain;
      domain.identity = "152390-a2000-235arcsec-s-experiment-only";
      domain.detector_array_association =
          detector_binding.detector_association_record_id;
      domain.motion = motion;
      domain.array = RtcOpticalArray::a2000;
      domain.speed_ceiling_arcsec_per_sec = 235;
      domain.nominal_interval_seconds = duration;
      domain.notch_guard_samples = guard;
      domain.reject = trial["reject"].as<bool>();
      const auto plan_started = std::chrono::steady_clock::now();
      auto plan = RtcNotchRecoveryPlan::consider(assessment, transients, val,
                                                 domain, 23 + trial_index);
      const auto apply_started = std::chrono::steady_clock::now();
      const std::array partitions{view};
      auto result = RtcNotchRecoveryResult::apply(plan, view, val, partitions);
      const auto apply_finished = std::chrono::steady_clock::now();
      const auto stem = s.identity;
      write_pair(output / (stem + "-native.f64"),
                 result->filtered_native_pair());
      write_pair(output / (stem + "-conditioned.f64"),
                 result->conditioned_native_pair());
      std::ofstream causes(output / (stem + "-causes.u8"), std::ios::binary);
      for (auto c : result->causes()) {
        auto u = static_cast<std::uint8_t>(c);
        causes.write(reinterpret_cast<const char *>(&u), 1);
      }
      causes.close();
      require(bool(causes), "cause output failed");
      std::ofstream selected(output / (stem + "-rows.i64"), std::ios::binary);
      for (auto row : result->output_native_rows())
        selected.write(reinterpret_cast<const char *>(&row), 8);
      selected.close();
      require(bool(selected), "row output failed");
      if (trial_index++)
        summary << ',';
      summary << "{\"id\":" << std::quoted(stem)
              << ",\"guard_samples_each_end\":"
              << guard + s.centered_lowpass.size() / 2
              << ",\"output_rows\":" << result->output_native_rows().size()
              << ",\"runs\":" << plan->runs().size() << "}";
      YAML::Node frozen;
      frozen["identity"] = s.identity;
      frozen["consideration"] = plan->consideration();
      frozen["original_input"] = mapping->paired_xr_record_id;
      frozen["VAL_generation"] = 0;
      frozen["AST_source"] = tel.metadata().source_artifact_identity;
      frozen["detector_occurrence"] = detector_binding.detector_occurrence_id;
      frozen["array_association"] =
          detector_binding.detector_association_record_id;
      frozen["interval_seconds"] = s.input_interval_seconds;
      frozen["factor"] = s.factor;
      frozen["first_native_row"] = plan->first_native_row();
      frozen["phase_native_rows"] = 0;
      frozen["state_support"] = s.state_support_identity;
      frozen["FIR"] = s.centered_lowpass;
      frozen["notch_guard_each_end"] = guard;
      frozen["reject"] = domain.reject;
      frozen["plan_seconds"] =
          std::chrono::duration<double>(apply_started - plan_started).count();
      frozen["Apply_seconds"] =
          std::chrono::duration<double>(apply_finished - apply_started).count();
      for (auto r : plan->runs()) {
        YAML::Node range;
        range.push_back(r.first);
        range.push_back(r.past_last);
        frozen["input_runs"].push_back(range);
      }
      for (const auto &n : s.notches) {
        YAML::Node q;
        q["a"] = std::vector<double>(n.a.begin(), n.a.end());
        q["b"] = std::vector<double>(n.b.begin(), n.b.end());
        q["direction"] = "forward-reverse";
        frozen["notches"].push_back(q);
      }
      YAML::Emitter frozen_out;
      frozen_out.SetDoublePrecision(17);
      frozen_out << frozen;
      std::ofstream fo(output / (stem + "-frozen-plan.yaml"));
      fo << frozen_out.c_str() << '\n';
      fo.close();
      require(bool(fo), "frozen plan export failed");
      // Optional precomputed telescope/beam boxcar injections are exact
      // hashed diagnostic deltas. Every pair reuses this SAME plan.
      for (const auto &inj : cfg["injections"]) {
        const fs::path path = inj["path"].as<std::string>();
        require(
            citlali::utils::sha256_file(output / "geometry.f64") ==
                inj["geometry_binding"]["sha256"].as<std::string>(),
            "injection AST/native geometry is incompatible with this replay");
        require(citlali::utils::sha256_file(path) ==
                    inj["sha256"].as<std::string>(),
                "changed injection");
        require(fs::file_size(path) == static_cast<std::uintmax_t>(rows) * 16,
                "injection shape");
        RtcRecoveryInjection delta;
        delta.plan = plan;
        delta.identity = inj["identity"].as<std::string>();
        delta.delta.resize(rows, 2);
        std::ifstream f(path, std::ios::binary);
        for (std::int64_t i = 0; i < rows; ++i)
          for (int c = 0; c < 2; ++c)
            f.read(reinterpret_cast<char *>(&delta.delta(i, c)), 8);
        require(bool(f), "injection read");
        auto paired =
            RtcNotchRecoveryResult::apply(plan, view, val, partitions, &delta);
        require(paired->causes() == result->causes() &&
                    paired->output_native_rows() ==
                        result->output_native_rows(),
                "injection changed frozen support");
        write_pair(output / (stem + "-" + delta.identity + ".f64"),
                   paired->filtered_native_pair());
      }
    }
    for (std::int64_t i = 0; i < rows; ++i) {
      require(std::bit_cast<std::uint64_t>(parent->network(nw).value(
                  NativeReadoutCoordinate::x, i, 0)) ==
                  std::bit_cast<std::uint64_t>(original_x(i, 0)),
              "original x changed");
      require(std::bit_cast<std::uint64_t>(parent->network(nw).value(
                  NativeReadoutCoordinate::r, i, 0)) ==
                  std::bit_cast<std::uint64_t>(original_r(i, 0)),
              "original r changed");
    }
    summary << "],\"original_pair_unchanged\":true,\"production_filtering_"
               "active\":false}\n";
    summary.close();
    require(bool(summary), "summary close");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "RTC notch recovery FAIL: " << e.what() << '\n';
    return 1;
  }
}
