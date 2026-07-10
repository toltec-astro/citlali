#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

void Beammap::init_beammap_diagnostic_apt_column(
    const std::string &name,
    double fill_value,
    const std::string &unit,
    const std::string &description) {
    calib.apt[name] = Eigen::VectorXd::Constant(calib.n_dets, fill_value);
    calib.apt_header_units[name] = unit;
    calib.apt_header_keys.push_back(name);
    calib.apt_meta[name].push_back("units: " + unit);
    calib.apt_meta[name].push_back(description);
}

void Beammap::init_beammap_diagnostic_apt_columns() {
    init_beammap_diagnostic_apt_column(
        "rfi_masked_samples", 0.0, "samples",
        "number of timestream samples masked by beammap rfi_mask");
    init_beammap_diagnostic_apt_column(
        "rfi_masked_scans", 0.0, "scans",
        "number of scans with at least one sample masked by beammap rfi_mask");
    init_beammap_diagnostic_apt_column(
        "scan_band_masked_samples", 0.0, "samples",
        "number of timestream samples masked by beammap scan_band_mask");
    init_beammap_diagnostic_apt_column(
        "scan_band_masked_rows", 0.0, "rows",
        "number of detector-map edge rows flagged by beammap scan_band_mask");

    init_beammap_diagnostic_apt_column(
        "scan_band_masked_edge", 0.0, "N/A",
        "scan-band edge code (0 none, 1 top, 2 bottom, 3 both)");
    calib.apt_meta["scan_band_masked_edge"].push_back(
        "scan-band edge code (0 none, 1 top, 2 bottom, 3 both)");
    calib.apt_meta["scan_band_masked_edge"].push_back("0=none");
    calib.apt_meta["scan_band_masked_edge"].push_back("1=top");
    calib.apt_meta["scan_band_masked_edge"].push_back("2=bottom");
    calib.apt_meta["scan_band_masked_edge"].push_back("3=both");

    init_beammap_diagnostic_apt_column(
        "scan_band_mask_rejected", 0.0, "N/A",
        "1 if scan_band_mask proposed a mask but rejected it due to max_flagged_fraction");

    init_beammap_diagnostic_apt_column(
        "final_prior_slot_index", -1.0, "N/A",
        "nearest prior slot index for final detector position in prior frame (-1 if unavailable)");
    init_beammap_diagnostic_apt_column(
        "final_prior_d2", std::numeric_limits<double>::quiet_NaN(), "N/A",
        "nearest-slot Mahalanobis d^2 for final detector position in the soft-prior frame");

    init_empirical_template_calibration_columns();
}

void Beammap::init_beammap_flag_metadata() {
    calib.apt_meta["flag2"].push_back("units: N/A");
    calib.apt_meta["flag2"].push_back("bitwise flag");
    calib.apt_meta["flag2"].push_back("Good=0");
    calib.apt_meta["flag2"].push_back("BadFit=1");
    calib.apt_meta["flag2"].push_back("AzFWHM=2");
    calib.apt_meta["flag2"].push_back("ElFWHM=4");
    calib.apt_meta["flag2"].push_back("Sig2Noise=8");
    calib.apt_meta["flag2"].push_back("Sens=16");
    calib.apt_meta["flag2"].push_back("Position=32");
    calib.apt_meta["flag2"].push_back("PriorDist=64");
    calib.apt_meta["flag2"].push_back("NetworkPos=128");

    for (const auto &[arr_index,arr_name]: toltec_io.array_name_map) {
        calib.apt_meta["array_order"].push_back(std::to_string(arr_index) + ": " + arr_name);
    }

    calib.apt_header_units["flag2"] = "N/A";
    calib.apt_header_keys.push_back("flag2");
}
