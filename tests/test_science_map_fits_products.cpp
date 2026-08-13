#include <gtest/gtest.h>

#include <citlali/core/engine/beammap.h>
#include <citlali/core/engine/engine.h>
#include <citlali/core/engine/pointing.h>
#include <citlali/core/engine/telescope.h>
#include <citlali/core/mapmaking/map.h>
#include <citlali/core/mapmaking/naive_mm.h>
#include <citlali/core/pipeline/flxscale_correction.h>
#include <citlali/core/pipeline/map_image_output_helpers.h>
#include <citlali/core/pipeline/calibration_product_admission.h>
#include <citlali/core/pipeline/mapmaking_provenance_lifecycle.h>
#include <citlali/core/pipeline/noise_provenance.h>
#include <citlali/core/pipeline/raw_timestream_provenance_lifecycle.h>
#include <citlali/core/pipeline/reduction_observation_inputs.h>
#include <citlali/core/pipeline/reduction_observation_pipeline.h>
#include <citlali/core/pipeline/science_map_provenance_serialization.h>
#include <citlali/core/utils/fits_io.h>
#include <citlali/core/timestream/ptc/ptcproc.h>
#include <citlali/core/timestream/rtc/calibrate.h>

#include <fitsio.h>
#include <spdlog/sinks/null_sink.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct CapturedHdu {
    std::map<std::string, std::string> keys;

    template <class Value>
    void addKey(const std::string &name, const Value &value,
                const std::string &, bool = false) {
        std::ostringstream stream;
        stream << value;
        keys[name] = stream.str();
    }
};

struct CapturedImage {
    std::string name;
    std::string data_type;
    Eigen::Index rows = 0;
    Eigen::Index cols = 0;
    std::vector<double> values;
};

struct CapturedFitsEntry {
    std::string filepath = "captured-science-map-products";
    std::vector<std::shared_ptr<CapturedHdu>> hdus;
    std::vector<CapturedImage> images;

    template <class Derived>
    void add_hdu(const std::string &name,
                 const Eigen::DenseBase<Derived> &data) {
        using Scalar = std::remove_cv_t<typename Derived::Scalar>;
        std::string data_type;
        if constexpr (std::is_same_v<Scalar, double>) {
            data_type = "float64";
        }
        else if constexpr (std::is_same_v<Scalar, std::int64_t>) {
            data_type = "int64";
        }
        else if constexpr (std::is_same_v<Scalar, std::uint8_t>) {
            data_type = "uint8";
        }
        else {
            data_type = "unexpected";
        }
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(data.size()));
        for (Eigen::Index row = 0; row < data.rows(); ++row) {
            for (Eigen::Index col = 0; col < data.cols(); ++col) {
                values.push_back(
                    static_cast<double>(data.derived()(row, col)));
            }
        }
        images.push_back(
            {name, data_type, data.rows(), data.cols(), std::move(values)});
        hdus.push_back(std::make_shared<CapturedHdu>());
    }

    template <class Hdu, class Wcs>
    void add_wcs(const Hdu &, const Wcs &, double) {}
};

struct DummyWcs {};

using ScienceMapBufferFixture = mapmaking::MapBuffer;

std::shared_ptr<spdlog::logger> science_map_test_logger() {
    static const auto logger = [] {
        auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        return std::make_shared<spdlog::logger>("science-map-fits-test", sink);
    }();
    return logger;
}

std::shared_ptr<ScienceMapBufferFixture> make_science_map_buffer(
    bool coadd = true) {
    auto map = std::make_shared<ScienceMapBufferFixture>(
        coadd ? "cmb" : "omb");
    map->n_rows = 2;
    map->n_cols = 2;
    map->n_noise = 0;
    map->sig_unit = "mJy/beam";
    map->map_grouping = "array";
    map->cov_cut = 1.0;
    map->science_products.allocate(1, 2, 2, coadd, true, true);
    auto &products = map->science_products;
    auto &realized = products.realized[0];
    mapmaking::ScienceMapBundleIdentity identity;
    identity.grouping = map->map_grouping;
    identity.signal_unit = map->sig_unit;
    identity.estimator_identity =
        mapmaking::science_map_coadd_estimator_version;
    identity.response_identity = "identity-response";
    identity.rows = map->n_rows;
    identity.cols = map->n_cols;
    identity.wcs.coordinate_frame = "equatorial-j2000";
    identity.wcs.projection = "TAN";
    identity.wcs.axis_types = {"RA---TAN", "DEC--TAN"};
    identity.wcs.axis_units = {"deg", "deg"};
    identity.wcs.pixel_scale = {-1.0 / 3600.0, 1.0 / 3600.0};
    identity.wcs.reference_world = {123.25, -45.5};
    identity.wcs.reference_pixel = {0.5, 0.5};
    identity.wcs.source_epoch = 2000.0;
    mapmaking::ScienceMapSlotIdentity slot;
    slot.grouping = "array";
    slot.group_identity = "array:0";
    slot.array_identity = 0;
    slot.frequency_hz = 150.0e9;
    identity.ordered_slots.push_back(slot);
    products.bundle_identity = identity;
    products.identity_admitted = true;
    realized.normalization.support_algorithm =
        mapmaking::science_map_normalization_support_version;
    realized.normalization.coefficient_stage =
        products.is_coadd
            ? mapmaking::science_map_coadd_normalization_coefficient_stage
            : mapmaking::science_map_observation_normalization_coefficient_stage;
    realized.science_policy.support_algorithm =
        mapmaking::science_map_policy_support_version;
    realized.science_policy.coefficient_stage = products.coefficient_stage;
    realized.normalization.requested_cut = 0.1;
    realized.normalization.realized_cut = 0.1;
    realized.normalization.realized_threshold = 0.3;
    realized.normalization.selected_positive_value = 3.0;
    realized.normalization.positive_value_count = 3;
    realized.normalization.selected_zero_based_index = 2;
    realized.normalization.selected_index_available = true;
    realized.science_policy.requested_cut = 1.0;
    realized.science_policy.realized_cut = 1.0;
    realized.science_policy.realized_threshold = 3.0;
    realized.science_policy.selected_positive_value = 3.0;
    realized.science_policy.positive_value_count = 3;
    realized.science_policy.selected_zero_based_index = 2;
    realized.science_policy.selected_index_available = true;

    products.geometric_hits[0] << 1, 2, 3, 4;
    products.contributing_hits[0] << 0, 1, 2, 3;
    products.coadd_observation_count[0] << 0, 0, 0, 0;
    products.upstream_eligible_exposure[0] << 0.5, 1.0, 1.5, 2.0;
    products.retained_exposure[0] << 0.0, 1.0, 1.5, 2.0;
    products.normalization_support[0] << 0, 1, 1, 1;
    products.science_policy_support[0] << 0, 0, 1, 1;
    products.science_valid[0] << 0, 0, 1, 1;
    map->signal = {Eigen::MatrixXd::Ones(2, 2)};
    Eigen::MatrixXd weight(2, 2);
    weight << 0.0, 1.0, 3.0, 3.0;
    map->weight = {weight};
    map->coverage = {products.retained_exposure[0]};
    map->median_err = Eigen::VectorXd::Constant(1, 3.0);
    map->median_rms = Eigen::VectorXd::Constant(1, 4.0);
    map->wcs.ctype = identity.wcs.axis_types;
    map->wcs.cunit = identity.wcs.axis_units;
    map->wcs.crval = {123.25F, -45.5F};
    map->wcs.cdelt = {-1.0F / 3600.0F, 1.0F / 3600.0F};
    map->wcs.crpix = {0.5F, 0.5F};
    map->wcs.naxis = {2, 2};
    mapmaking::science_map_finalize_realized_product_facts(*map, 0);
    return map;
}

const CapturedHdu &captured_hdu(const CapturedFitsEntry &entry,
                                const std::string &name) {
    for (std::size_t i = 0; i < entry.images.size(); ++i) {
        if (entry.images[i].name == name) {
            return *entry.hdus[i];
        }
    }
    throw std::runtime_error("missing captured HDU " + name);
}

const CapturedImage &captured_image(const CapturedFitsEntry &entry,
                                    const std::string &name) {
    for (const auto &image : entry.images) {
        if (image.name == name) {
            return image;
        }
    }
    throw std::runtime_error("missing captured image " + name);
}

bool captured_has_image(const CapturedFitsEntry &entry,
                        const std::string &name) {
    for (const auto &image : entry.images) {
        if (image.name == name) {
            return true;
        }
    }
    return false;
}

std::size_t captured_image_count(const CapturedFitsEntry &entry,
                                 const std::string &name) {
    return static_cast<std::size_t>(std::count_if(
        entry.images.begin(), entry.images.end(),
        [&](const auto &image) { return image.name == name; }));
}

void set_noise_stack(
    mapmaking::MapBuffer &map,
    const std::vector<Eigen::MatrixXd> &realizations) {
    ASSERT_FALSE(realizations.empty());
    map.n_noise = static_cast<Eigen::Index>(realizations.size());
    map.noise.clear();
    map.noise.emplace_back(map.n_rows, map.n_cols, map.n_noise);
    for (Eigen::Index realization = 0; realization < map.n_noise;
         ++realization) {
        ASSERT_EQ(realizations[static_cast<std::size_t>(realization)].rows(),
                  map.n_rows);
        ASSERT_EQ(realizations[static_cast<std::size_t>(realization)].cols(),
                  map.n_cols);
        for (Eigen::Index row = 0; row < map.n_rows; ++row) {
            for (Eigen::Index col = 0; col < map.n_cols; ++col) {
                map.noise[0](row, col, realization) =
                    realizations[static_cast<std::size_t>(realization)](
                        row, col);
            }
        }
    }
}

std::shared_ptr<mapmaking::MapBuffer> make_noise_product_fixture(
    const Eigen::MatrixXd &signal, const Eigen::MatrixXd &weight,
    const std::vector<Eigen::MatrixXd> &realizations) {
    auto map = std::make_shared<mapmaking::MapBuffer>("noise-fixture");
    map->n_rows = signal.rows();
    map->n_cols = signal.cols();
    map->cov_cut = 0.0;
    map->signal = {signal};
    map->weight = {weight};
    set_noise_stack(*map, realizations);
    return map;
}

struct F005Correction {
    double factor = 1.0;
    double value() const { return factor; }
};

struct F005RawObservation {
    const F005Correction *correction = nullptr;
    std::string observation_name{"f005-production-observation"};
    const F005Correction *flxscale_correction() const { return correction; }
    const std::string &name() const { return observation_name; }
};

struct F005CorrectionEngine {
    struct Calib {
        std::map<std::string, Eigen::VectorXd> apt;
        Eigen::VectorXd flux_conversion_factor;
        std::map<std::string, double> mean_flux_conversion_factor;
        int setup_calls = 0;

        void calc_flux_calibration(const std::string &, double) {
            ++setup_calls;
            flux_conversion_factor =
                Eigen::VectorXd::Ones(apt.at("flxscale").size());
            mean_flux_conversion_factor.clear();
        }
    } calib;
    struct {
        std::string sig_unit{"mJy/beam"};
        double pixel_size_rad = 1.0e-5;
    } omb;
};

struct F005AppliedCorrectionState {
    double source_flxscale = 1.0;
    double source_sensitivity = 1.0;
    double applied_correction = 1.0;
};

F005AppliedCorrectionState apply_f005_production_correction(double factor) {
    F005CorrectionEngine engine;
    engine.calib.apt["flxscale"] = Eigen::VectorXd::Ones(1);
    engine.calib.apt["sens"] = Eigen::VectorXd::Constant(1, 2.0);
    const auto source_flxscale = engine.calib.apt.at("flxscale");
    const auto source_sensitivity = engine.calib.apt.at("sens");
    const F005Correction correction{factor};
    const F005RawObservation rawobs{&correction,
                                    "f005-production-observation"};
    if (!citlali::pipeline::
             prepare_reduction_observation_flux_calibration_state(
            engine, rawobs, science_map_test_logger())) {
        throw std::runtime_error("valid F005 production correction rejected");
    }
    if (engine.calib.setup_calls != 1) {
        throw std::runtime_error(
            "F005 correction did not follow production carrier setup");
    }
    if (!engine.calib.apt.at("flxscale").isApprox(source_flxscale, 0.0) ||
        !engine.calib.apt.at("sens").isApprox(source_sensitivity, 0.0)) {
        throw std::runtime_error("F005 production correction mutated source APT");
    }
    const auto state = engine.calib.mean_flux_conversion_factor.find(
        std::string{
            citlali::pipeline::observation_flxscale_correction_state_key});
    if (state == engine.calib.mean_flux_conversion_factor.end()) {
        throw std::runtime_error("F005 production correction state missing");
    }
    if (citlali::pipeline::apply_flxscale_correction(
            engine, rawobs, science_map_test_logger())) {
        throw std::runtime_error(
            "duplicate F005 production correction was not rejected");
    }
    if (engine.calib.flux_conversion_factor(0) != factor ||
        !engine.calib.apt.at("flxscale").isApprox(source_flxscale, 0.0) ||
        !engine.calib.apt.at("sens").isApprox(source_sensitivity, 0.0)) {
        throw std::runtime_error(
            "duplicate F005 correction changed applied or source state");
    }
    const double admitted_correction =
        citlali::pipeline::applied_observation_flxscale_correction(
            engine.calib.flux_conversion_factor, 1, state->second);
    if (admitted_correction != factor) {
        throw std::runtime_error(
            "F005 production correction was not admitted exactly");
    }
    return {source_flxscale(0), source_sensitivity(0), admitted_correction};
}

timestream::CalibrationProductAdmissionInputs f005_admission_inputs(
    const F005AppliedCorrectionState &state, bool correction_applied) {
    timestream::CalibrationProductAdmissionInputs inputs;
    inputs.target_unit = "mJy/beam";
    inputs.calibration_requested = true;
    inputs.responsivity_required = true;
    inputs.sensitivity_required = true;
    inputs.beam_template_required = true;
    inputs.acquisition_identity_available = true;
    inputs.acquisition_identity_valid = true;
    inputs.acquisition_identity_detail = "f005-production-binding";
    inputs.apt_lineage_available = true;
    inputs.apt_lineage_valid = true;
    inputs.apt_lineage_detail = "f005-production-lineage";
    inputs.apt_artifact_sha256 = "f005-production-apt";
    inputs.apt_row_association_sha256 = "f005-production-row-association";
    inputs.apt_observation_identity = "42";
    inputs.apt_matched_observation_identity = "42";
    inputs.apt_selected_source = "f005-production-source";
    inputs.tolapt_manifest_association_sha256 =
        "f005-production-manifest-association";
    inputs.acquisition_binding_sha256 = "f005-production-binding-sha";
    inputs.raw_observation_identity = "f005-production-raw";
    inputs.acquisition_binding_mode = "explicit";
    inputs.acquisition_key_schema = "artifact+network+local_tone";
    inputs.response_identity = "f005-production-response";
    inputs.applied_sample_extinction_state_sha256 =
        "f005-production-no-extinction";
    inputs.atmosphere_operator_id = "f005-production-operator";
    inputs.atmosphere_operator_contract_sha256 =
        "f005-production-operator-contract";
    inputs.atmosphere_node_table_sha256 = "f005-production-node-table";
    inputs.passband_set_id = "f005-production-passband";
    inputs.reference_profile_id = "f005-production-reference";
    inputs.tau225 = 0.1;
    auto &lineage = inputs.package_lineage;
    lineage.selected_apt_source_path = "/f005/apt.ecsv";
    lineage.selected_apt_sha256 = inputs.apt_artifact_sha256;
    lineage.apt_row_association_sha256 = inputs.apt_row_association_sha256;
    lineage.modern_tolapt_manifest_available = true;
    lineage.modern_tolapt_manifest_path = "/f005/manifest.yaml";
    lineage.modern_tolapt_manifest_sha256 = "f005-production-manifest";
    lineage.modern_tolapt_contract_version = "tolapt.run.v1";
    lineage.modern_tolapt_run_id = "f005-production-run";
    lineage.modern_tolapt_output_key = "matched_design_apt";
    lineage.modern_tolapt_output_path = "matched.ecsv";
    lineage.tolapt_manifest_association_sha256 =
        inputs.tolapt_manifest_association_sha256;
    lineage.modern_tolapt_design_input =
        {"design.ecsv", "f005-design", 1, "2026-08-11T00:00:00Z"};
    lineage.modern_tolapt_measured_input =
        {"measured.ecsv", "f005-measured", 1, "2026-08-11T00:00:01Z"};
    lineage.raw_artifacts.push_back(
        {"raw.nc", "f005-raw", "toltec0", 0, {1.0e9}});
    timestream::CalibrationLineageRow row;
    row.ordered_detector_index = 0;
    row.selected_source_row_index = 0;
    row.network = 0;
    row.network_local_tone = 0;
    row.absolute_tone_frequency_hz = 1.0e9;
    row.uid = "17";
    row.eligible = true;
    row.validity_basis = "f005-valid";
    row.stable_association = "f005-stable";
    lineage.ordered_rows.push_back(row);
    inputs.target_unit_factor = Eigen::VectorXd::Ones(1);
    inputs.observation_flxscale_correction_applied = correction_applied;
    inputs.applied_observation_flxscale_correction =
        correction_applied ? state.applied_correction : 1.0;
    inputs.observation_flxscale_correction_state =
        correction_applied ? "applied_once" : "not_applied";
    inputs.observation_flxscale_correction_source_identity =
        correction_applied
            ? std::string{timestream::CalibrationProduct::
                              observation_correction_source_identity}
            : "not_applied";
    inputs.observation_flxscale_correction_recipient_identity =
        correction_applied ? inputs.raw_observation_identity : std::string{};
    inputs.detector_flxscale =
        Eigen::VectorXd::Constant(1, state.source_flxscale);
    inputs.detector_responsivity = Eigen::VectorXd::Ones(1);
    inputs.detector_sensitivity =
        Eigen::VectorXd::Constant(1, state.source_sensitivity);
    inputs.detector_beam_major_fwhm_arcsec =
        Eigen::VectorXd::Constant(1, 10.0);
    inputs.detector_beam_minor_fwhm_arcsec =
        Eigen::VectorXd::Constant(1, 9.0);
    inputs.minimum_extinction_correction = Eigen::VectorXd::Ones(1);
    inputs.maximum_extinction_correction = Eigen::VectorXd::Ones(1);
    inputs.applied_sample_extinction_state.available = true;
    inputs.applied_sample_extinction_state_sha256 =
        timestream::applied_sample_extinction_state_identity(
            inputs.applied_sample_extinction_state);
    return inputs;
}

struct F005CalibrationFixture {
    std::map<std::string, Eigen::VectorXd> apt;
};

struct F005ProductionResult {
    double sample = 0.0;
    double compatibility_fcf = 0.0;
    double detector_weight = 0.0;
    double map_signal = 0.0;
    double map_weight = 0.0;
    double realization_minus = 0.0;
    double realization_plus = 0.0;
    double noise_variance_I = 0.0;
};

F005ProductionResult run_f005_production_route(
    const std::string &mode, const F005AppliedCorrectionState &state,
    bool correction_applied) {
    using Data =
        timestream::TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd>;
    timestream::Calibration calibration;
    calibration.logger = science_map_test_logger();
    calibration.admit_product(
        f005_admission_inputs(state, correction_applied));

    Data data;
    data.scans.data.resize(2, 1);
    data.scans.data << 1.0, 3.0;
    data.flags.data =
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
            2, 1, false);
    data.fcf.data = Eigen::VectorXd::Ones(1);
    data.noise.data.resize(2, 1);
    data.noise.data << -1.0, 1.0;
    data.index.data = 0;
    F005CalibrationFixture calibration_fixture;
    calibration_fixture.apt["array"] = Eigen::VectorXd::Zero(1);
    calibration.calibrate_tod(data, calibration_fixture);
    data.status.calibrated = true;

    std::map<std::string, Eigen::VectorXd> apt;
    apt["array"] = Eigen::VectorXd::Zero(1);
    apt["flag"] = Eigen::VectorXd::Zero(1);
    apt["sens"] =
        Eigen::VectorXd::Constant(1, state.source_sensitivity);
    apt["uid"] = Eigen::VectorXd::Constant(1, 17.0);
    apt["x_t"] = Eigen::VectorXd::Zero(1);
    apt["y_t"] = Eigen::VectorXd::Zero(1);
    timestream::PTCProc processor;
    processor.logger = science_map_test_logger();
    processor.weighting_type = mode;
    engine::Telescope telescope;
    telescope.d_fsmp = 1.0;
    processor.calc_weights(data, apt, telescope, false);

    data.kernel.data = Eigen::MatrixXd::Ones(2, 1);
    data.tel_data.data["TelElAct"] = Eigen::VectorXd::Zero(2);
    data.tel_data.data["alt_phys"] = Eigen::VectorXd::Zero(2);
    data.tel_data.data["az_phys"] = Eigen::VectorXd::Zero(2);
    data.pointing_offsets_arcsec.data["az"] = Eigen::VectorXd::Zero(2);
    data.pointing_offsets_arcsec.data["alt"] = Eigen::VectorXd::Zero(2);

    mapmaking::MapBuffer map{"omb"};
    map.n_rows = 1;
    map.n_cols = 1;
    map.pixel_size_rad = 1.0e-5;
    map.map_grouping = "array";
    map.parallel_policy = "seq";
    map.sig_unit = "mJy/beam";
    map.cov_cut = 0.0;
    map.n_noise = 2;
    map.randomize_dets = false;
    map.signal.emplace_back(Eigen::MatrixXd::Zero(1, 1));
    map.weight.emplace_back(Eigen::MatrixXd::Zero(1, 1));
    map.noise.emplace_back(1, 1, 2);
    map.noise.back().setZero();
    mapmaking::MapBuffer unused_coadd{"cmb"};
    Eigen::VectorXi map_indices = Eigen::VectorXi::Zero(1);
    std::string pixel_axes = "altaz";
    mapmaking::NaiveMapmaker mapmaker;
    mapmaker.run_polarization = false;
    mapmaker.populate_maps_naive(
        data, map, unused_coadd, map_indices, pixel_axes, apt, 1.0,
        true, true);
    map.normalize_maps();
    const double realization_minus = map.noise[0](0, 0, 0);
    const double realization_plus = map.noise[0](0, 0, 1);
    map.calc_noise_products(Eigen::Index{0}, false, true);
    map.median_err = Eigen::VectorXd::Ones(1);

    CapturedFitsEntry output;
    DummyWcs wcs;
    auto *map_ptr = &map;
    citlali::pipeline::add_primary_map_image_hdus(
        output, map_ptr, 0, "", "I", wcs, 2000.0, false, true,
        false, false, science_map_test_logger());
    const auto &variance = captured_image(output, "noise_variance_I").values;
    if (variance.size() != 1) {
        throw std::runtime_error("F005 noise_variance_I publication missing");
    }
    return {
        data.scans.data(0, 0), data.fcf.data(0), data.weights.data(0),
        map.signal[0](0, 0), map.weight[0](0, 0), realization_minus,
        realization_plus, variance.front()};
}

TEST(science_map_fits_products,
     conditional_stack_scatter_R1_is_descriptive_but_not_uncertainty) {
    const Eigen::MatrixXd signal = Eigen::MatrixXd::Constant(1, 1, 2.0);
    const Eigen::MatrixXd weight = Eigen::MatrixXd::Ones(1, 1);
    auto map = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, 5.0)});

    map->calc_noise_products(Eigen::Index{0}, false, true);

    EXPECT_DOUBLE_EQ(map->noise_mean[0](0, 0), 5.0);
    EXPECT_DOUBLE_EQ(map->noise_variance[0](0, 0), 0.0);
    EXPECT_EQ(map->noise_stack_scatter_valid(0), 1);
    EXPECT_EQ(map->noise_uncertainty_use_valid(0), 0);
    EXPECT_EQ(map->noise_weight_scale_valid(0), 0);
    EXPECT_TRUE(std::isnan(map->noise_weight_scale(0)));
    EXPECT_TRUE(std::isnan(map->sig2noise_pixel[0](0, 0)));
    EXPECT_TRUE(std::isnan(map->point_source_uncertainty[0](0, 0)));
    EXPECT_TRUE(std::isnan(map->sig2noise_point_source[0](0, 0)));

    auto required_scale = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, 5.0)});
    EXPECT_THROW(required_scale->calc_noise_products(
                     Eigen::Index{0}, true, true),
                 std::runtime_error);

    auto uncentered = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, 5.0)});
    EXPECT_THROW(uncentered->calc_noise_products(
                     Eigen::Index{0}, false, false),
                 std::invalid_argument);
}

TEST(science_map_fits_products,
     conditional_stack_scatter_R2_uses_completed_R_normalization) {
    const Eigen::MatrixXd signal = Eigen::MatrixXd::Constant(1, 1, 2.0);
    const Eigen::MatrixXd weight = Eigen::MatrixXd::Constant(1, 1, 0.25);
    auto map = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, -2.0),
         Eigen::MatrixXd::Constant(1, 1, 2.0)});

    map->calc_noise_products(Eigen::Index{0}, false, true);

    EXPECT_DOUBLE_EQ(map->noise_mean[0](0, 0), 0.0);
    EXPECT_DOUBLE_EQ(map->noise_variance[0](0, 0), 4.0);
    EXPECT_EQ(map->noise_uncertainty_use_valid(0), 1);
    EXPECT_EQ(map->noise_weight_scale_valid(0), 1);
    EXPECT_DOUBLE_EQ(map->noise_weight_median_ratio(0), 1.0);
    EXPECT_DOUBLE_EQ(map->noise_weight_scale(0), 1.0);
    EXPECT_DOUBLE_EQ(map->weight_empirical[0](0, 0), 0.25);
    EXPECT_DOUBLE_EQ(map->sig2noise_pixel[0](0, 0), 1.0);
    EXPECT_DOUBLE_EQ(map->point_source_uncertainty[0](0, 0), 2.0);
    EXPECT_DOUBLE_EQ(map->sig2noise_point_source[0](0, 0), 1.0);

    auto existing_use_only = make_noise_product_fixture(
        signal, Eigen::MatrixXd::Constant(1, 1, 0.5),
        {Eigen::MatrixXd::Constant(1, 1, -2.0),
         Eigen::MatrixXd::Constant(1, 1, 2.0)});
    existing_use_only->calc_noise_products(Eigen::Index{0}, true, true);
    EXPECT_DOUBLE_EQ(existing_use_only->weight_formal[0](0, 0), 0.5);
    EXPECT_DOUBLE_EQ(existing_use_only->noise_weight_scale(0), 0.5);
    EXPECT_DOUBLE_EQ(existing_use_only->weight[0](0, 0), 0.25);
}

TEST(science_map_fits_products,
     duplicate_complementary_and_simple_R2_designs_are_exact) {
    const Eigen::MatrixXd signal = Eigen::MatrixXd::Ones(1, 1);
    const Eigen::MatrixXd weight = Eigen::MatrixXd::Ones(1, 1);

    auto duplicate = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, 3.0),
         Eigen::MatrixXd::Constant(1, 1, 3.0)});
    duplicate->calc_noise_products(Eigen::Index{0}, false, true);
    EXPECT_DOUBLE_EQ(duplicate->noise_variance[0](0, 0), 0.0);
    EXPECT_EQ(duplicate->noise_weight_scale_valid(0), 0);
    EXPECT_TRUE(std::isnan(duplicate->sig2noise_point_source[0](0, 0)));

    auto complementary = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, -2.0),
         Eigen::MatrixXd::Constant(1, 1, 2.0)});
    complementary->calc_noise_products(Eigen::Index{0}, false, true);
    EXPECT_DOUBLE_EQ(complementary->noise_variance[0](0, 0), 4.0);

    auto simple = make_noise_product_fixture(
        signal, weight,
        {Eigen::MatrixXd::Constant(1, 1, 0.0),
         Eigen::MatrixXd::Constant(1, 1, 2.0)});
    simple->calc_noise_products(Eigen::Index{0}, false, true);
    EXPECT_DOUBLE_EQ(simple->noise_mean[0](0, 0), 1.0);
    EXPECT_DOUBLE_EQ(simple->noise_variance[0](0, 0), 1.0);
}

TEST(science_map_fits_products,
     empty_scale_calibration_region_is_unavailable_and_fails_closed) {
    const Eigen::MatrixXd signal = Eigen::MatrixXd::Ones(1, 1);
    const Eigen::MatrixXd zero_weight = Eigen::MatrixXd::Zero(1, 1);
    const std::vector<Eigen::MatrixXd> realizations{
        Eigen::MatrixXd::Constant(1, 1, -1.0),
        Eigen::MatrixXd::Constant(1, 1, 1.0)};
    auto diagnostic = make_noise_product_fixture(
        signal, zero_weight, realizations);

    diagnostic->calc_noise_products(Eigen::Index{0}, false, true);
    EXPECT_EQ(diagnostic->noise_valid_pixels(0), 0.0);
    EXPECT_EQ(diagnostic->noise_weight_scale_valid(0), 0);
    EXPECT_TRUE(std::isnan(diagnostic->noise_weight_median_ratio(0)));
    EXPECT_TRUE(std::isnan(diagnostic->noise_weight_scale(0)));
    EXPECT_TRUE(std::isnan(diagnostic->sig2noise_pixel[0](0, 0)));

    auto required_scale = make_noise_product_fixture(
        signal, zero_weight, realizations);
    EXPECT_THROW(required_scale->calc_noise_products(
                     Eigen::Index{0}, true, true),
                 std::runtime_error);
}

TEST(science_map_fits_products,
     fixed_projection_preserves_two_pixel_covariance_without_dense_matrix) {
    Eigen::MatrixXd signal = Eigen::MatrixXd::Zero(1, 2);
    Eigen::MatrixXd weight = Eigen::MatrixXd::Ones(1, 2);
    Eigen::MatrixXd plus = Eigen::MatrixXd::Ones(1, 2);
    Eigen::MatrixXd minus = -Eigen::MatrixXd::Ones(1, 2);
    auto map = make_noise_product_fixture(signal, weight, {plus, minus});
    map->calc_noise_products(Eigen::Index{0}, false, true);

    Eigen::MatrixXd aperture = Eigen::MatrixXd::Ones(1, 2);
    const double projected_scatter =
        map->calc_fixed_projection_stack_scatter(0, aperture);
    const double diagonal_only = map->noise_variance[0].sum();
    EXPECT_DOUBLE_EQ(diagonal_only, 2.0);
    EXPECT_DOUBLE_EQ(projected_scatter, 4.0);
    EXPECT_DOUBLE_EQ(
        map->calc_fixed_projection_stack_scatter(0, aperture, 2.0),
        1.0);

    Eigen::MatrixXd first(1, 2);
    first << 2.0, 0.0;
    Eigen::MatrixXd second(1, 2);
    second << 0.0, 2.0;
    auto template_map = make_noise_product_fixture(
        signal, weight, {first, second});
    Eigen::MatrixXd fixed_template(1, 2);
    fixed_template << 0.5, -0.5;
    EXPECT_DOUBLE_EQ(
        template_map->calc_fixed_projection_stack_scatter(
            0, fixed_template),
        1.0);
    EXPECT_THROW(
        template_map->calc_fixed_projection_stack_scatter(
            0, fixed_template, 0.0),
        std::invalid_argument);
}

TEST(science_map_fits_products,
     filtered_scatter_validity_distinguishes_exact_failure_reasons) {
    const Eigen::MatrixXd finite_scatter =
        Eigen::MatrixXd::Ones(2, 2);
    Eigen::MatrixXd nonfinite_scatter =
        Eigen::MatrixXd::Constant(
            2, 2, std::numeric_limits<double>::quiet_NaN());
    mapmaking::ScienceMapMaskPlane supported =
        mapmaking::ScienceMapMaskPlane::Ones(2, 2);
    mapmaking::ScienceMapMaskPlane unsupported =
        mapmaking::ScienceMapMaskPlane::Zero(2, 2);
    Eigen::MatrixXd finite_only_off_support(2, 2);
    finite_only_off_support <<
        1.0, 2.0,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN();
    mapmaking::ScienceMapMaskPlane mixed_support(2, 2);
    mixed_support << 0, 0, 1, 1;

    auto validity = citlali::pipeline::filtered_scatter_validity(
        1, finite_scatter, 1.0, &supported);
    EXPECT_FALSE(validity.available);
    EXPECT_STREQ(validity.status, "R_lt_2");

    validity = citlali::pipeline::filtered_scatter_validity(
        2, nonfinite_scatter, 1.0, &supported);
    EXPECT_FALSE(validity.available);
    EXPECT_STREQ(validity.status, "scatter_unavailable_or_nonfinite");

    validity = citlali::pipeline::filtered_scatter_validity(
        2, finite_scatter, 0.0, &supported);
    EXPECT_FALSE(validity.available);
    EXPECT_STREQ(validity.status, "response_invalid");

    validity = citlali::pipeline::filtered_scatter_validity(
        2, finite_scatter, 1.0, &unsupported);
    EXPECT_FALSE(validity.available);
    EXPECT_STREQ(validity.status, "support_invalid");

    validity = citlali::pipeline::filtered_scatter_validity(
        2, finite_only_off_support, 1.0, &mixed_support);
    EXPECT_FALSE(validity.available);
    EXPECT_STREQ(validity.status, "scatter_unavailable_or_nonfinite");

    validity = citlali::pipeline::filtered_scatter_validity(
        2, finite_scatter, 1.0, &supported);
    EXPECT_TRUE(validity.available);
    EXPECT_STREQ(
        validity.status, "available_where_finite_on_valid_support");
}

TEST(science_map_fits_products,
     mixed_filtered_scatter_fails_closed_with_exact_reason) {
    auto map = make_science_map_buffer(false);
    set_noise_stack(
        *map,
        {Eigen::MatrixXd::Constant(2, 2, -2.0),
         Eigen::MatrixXd::Constant(2, 2, 2.0)});
    mapmaking::science_map_finalize_realized_product_facts(*map, 0);
    map->calc_noise_products(Eigen::Index{0}, false, true);
    map->freeze_raw_science_parent();

    const double unavailable =
        std::numeric_limits<double>::quiet_NaN();
    map->point_source_uncertainty[0] <<
        1.0, 2.0, unavailable, unavailable;
    map->sig2noise_point_source[0] << 10.0, 20.0, 30.0, 40.0;

    CapturedFitsEntry output;
    DummyWcs wcs;
    citlali::pipeline::add_coverage_support_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0, true, true, false,
        science_map_test_logger());

    const auto &scatter =
        captured_image(output, "point_source_uncertainty_I");
    const auto &ratio =
        captured_image(output, "sig2noise_point_source_I");
    ASSERT_EQ(scatter.values.size(), 4U);
    ASSERT_EQ(ratio.values.size(), 4U);
    EXPECT_TRUE(std::all_of(
        scatter.values.begin(), scatter.values.end(),
        [](double value) { return std::isnan(value); }));
    EXPECT_TRUE(std::all_of(
        ratio.values.begin(), ratio.values.end(),
        [](double value) { return std::isnan(value); }));
    EXPECT_EQ(
        captured_hdu(output, "point_source_uncertainty_I")
            .keys.at("NOIVALID"),
        "scatter_unavailable_or_nonfinite");
    EXPECT_EQ(
        captured_hdu(output, "sig2noise_point_source_I")
            .keys.at("NOIVALID"),
        "scatter_unavailable_or_nonfinite");
}

TEST(science_map_fits_products,
     retains_one_compatible_noise_plane_with_canonical_identity_metadata) {
    auto map = make_science_map_buffer(false);
    set_noise_stack(
        *map,
        {Eigen::MatrixXd::Constant(2, 2, -2.0),
         Eigen::MatrixXd::Constant(2, 2, 2.0)});
    map->science_products.bundle_identity->required_companions = {
        "noise_realization_0_I", "noise_realization_1_I"};
    mapmaking::science_map_finalize_realized_product_facts(*map, 0);
    map->calc_noise_products(Eigen::Index{0}, false, true);
    ASSERT_EQ(map->noise_weight_scale_valid(0), 1);
    CapturedFitsEntry primary;
    CapturedFitsEntry raw_support;
    DummyWcs wcs;

    citlali::pipeline::add_primary_map_image_hdus(
        primary, map, 0, "", "I", wcs, 2000.0, false, true, false,
        false, science_map_test_logger());
    citlali::pipeline::add_coverage_support_image_hdus(
        raw_support, map, 0, "", "I", wcs, 2000.0, false, true, false,
        science_map_test_logger());

    EXPECT_FALSE(captured_has_image(
        primary, "conditional_stack_scatter_I"));
    ASSERT_TRUE(captured_has_image(primary, "noise_variance_I"));
    const auto &scatter = captured_hdu(primary, "noise_variance_I").keys;
    EXPECT_EQ(scatter.at("ESTTYPE"),
              "conditional_finite_stack_scatter");
    EXPECT_EQ(scatter.at("NOIPKG"), "citlali-noise-products");
    EXPECT_EQ(scatter.at("NOIPROV"),
              "noise_products_provenance.yaml");
    EXPECT_EQ(scatter.at("NOIPRID"),
              "conditional_finite_stack_scatter");
    EXPECT_EQ(scatter.at("NOIPVER"), "SCI-NOI-002-v1");
    EXPECT_EQ(scatter.at("NOIDGST"),
              citlali::pipeline::noise_product_semantic_digest(
                  citlali::pipeline::
                      noise_conditional_stack_scatter_product_id));
    EXPECT_EQ(scatter.find("NOIRCOMP"), scatter.end());
    EXPECT_EQ(scatter.at("DEPRCATD"), "true");
    EXPECT_EQ(scatter.find("ALIASOF"), scatter.end());

    EXPECT_FALSE(captured_has_image(
        raw_support, "coefficient_standardized_signal_I"));
    const auto &standardized = captured_hdu(
        raw_support, "sig2noise_I").keys;
    EXPECT_EQ(standardized.at("ESTTYPE"),
              "coefficient_standardized_signal");
    EXPECT_EQ(standardized.at("SIGSTAT"), "not_significance");
    EXPECT_EQ(standardized.at("NOIPRID"),
              "coefficient_standardized_signal");
    EXPECT_EQ(standardized.find("ALIASOF"), standardized.end());
    EXPECT_EQ(captured_image_count(raw_support, "sig2noise_I"), 1U);
    EXPECT_FALSE(captured_has_image(raw_support, "sig2noise_pixel_I"));

    map->freeze_raw_science_parent();
    CapturedFitsEntry filtered_support;
    citlali::pipeline::add_coverage_support_image_hdus(
        filtered_support, map, 0, "", "I", wcs, 2000.0, true, true,
        false, science_map_test_logger());

    EXPECT_EQ(captured_image_count(filtered_support, "sig2noise_I"), 1U);
    EXPECT_FALSE(captured_has_image(
        filtered_support, "sig2noise_pixel_I"));
    EXPECT_FALSE(captured_has_image(
        filtered_support, "coefficient_standardized_signal_I"));
    EXPECT_FALSE(captured_has_image(
        filtered_support, "filtered_pixel_stack_scatter_I"));
    const auto &filtered_scatter = captured_hdu(
        filtered_support, "point_source_uncertainty_I").keys;
    EXPECT_EQ(filtered_scatter.at("ESTTYPE"),
              "filtered_pixel_stack_scatter");
    EXPECT_EQ(filtered_scatter.at("NOIPRID"),
              "filtered_pixel_stack_scatter");
    EXPECT_NE(filtered_scatter.at("NOIRESTR").find(
                  "strict_parity_pending_FLT"),
              std::string::npos);
    EXPECT_EQ(filtered_scatter.find("ALIASOF"), filtered_scatter.end());
    EXPECT_EQ(filtered_scatter.at("NOIVALID"),
              "available_where_finite_on_valid_support");
    EXPECT_FALSE(captured_has_image(
        filtered_support, "conditional_stack_scatter_ratio_I"));
    const auto &ratio = captured_hdu(
        filtered_support, "sig2noise_point_source_I").keys;
    EXPECT_EQ(ratio.at("ESTTYPE"),
              "conditional_stack_scatter_ratio");
    EXPECT_EQ(ratio.at("SIGSTAT"), "not_significance");
    EXPECT_EQ(ratio.at("NOIPRID"),
              "conditional_stack_scatter_ratio");
    EXPECT_EQ(ratio.find("ALIASOF"), ratio.end());
    EXPECT_EQ(
        ratio.at("NOIVALID"),
        "available_where_finite_positive_denominator_on_valid_support");
}

TEST(science_map_fits_products,
     production_correction_setup_owns_reset_reuse_and_composition_failure) {
    F005CorrectionEngine engine;
    engine.calib.apt["flxscale"] = Eigen::VectorXd::Ones(1);
    engine.calib.apt["sens"] = Eigen::VectorXd::Constant(1, 2.0);
    const auto source_flxscale = engine.calib.apt.at("flxscale");
    const auto source_sensitivity = engine.calib.apt.at("sens");
    const auto logger = science_map_test_logger();
    const std::string state_key{
        citlali::pipeline::observation_flxscale_correction_state_key};

    const F005Correction first_correction{3.0};
    const F005RawObservation first{&first_correction, "observation-a"};
    ASSERT_TRUE(citlali::pipeline::
                    prepare_reduction_observation_flux_calibration_state(
                        engine, first, logger));
    ASSERT_EQ(engine.calib.setup_calls, 1);
    ASSERT_DOUBLE_EQ(engine.calib.flux_conversion_factor(0), 3.0);
    ASSERT_DOUBLE_EQ(engine.calib.mean_flux_conversion_factor.at(state_key),
                     3.0);

    const auto applied_once = engine.calib.flux_conversion_factor;
    EXPECT_FALSE(citlali::pipeline::apply_flxscale_correction(
        engine, first, logger));
    EXPECT_TRUE(engine.calib.flux_conversion_factor.isApprox(
        applied_once, 0.0));

    const F005Correction second_correction{4.0};
    const F005RawObservation second{&second_correction, "observation-b"};
    ASSERT_TRUE(citlali::pipeline::
                    prepare_reduction_observation_flux_calibration_state(
                        engine, second, logger));
    EXPECT_EQ(engine.calib.setup_calls, 2);
    EXPECT_DOUBLE_EQ(engine.calib.flux_conversion_factor(0), 4.0);
    EXPECT_DOUBLE_EQ(engine.calib.mean_flux_conversion_factor.at(state_key),
                     4.0);

    const F005RawObservation uncorrected{nullptr, "observation-c"};
    ASSERT_TRUE(citlali::pipeline::
                    prepare_reduction_observation_flux_calibration_state(
                        engine, uncorrected, logger));
    EXPECT_EQ(engine.calib.setup_calls, 3);
    EXPECT_TRUE(engine.calib.flux_conversion_factor.isOnes());
    EXPECT_EQ(engine.calib.mean_flux_conversion_factor.count(state_key), 0U);

    engine.calib.flux_conversion_factor(0) =
        std::numeric_limits<double>::max();
    const F005Correction overflow_correction{2.0};
    const F005RawObservation overflow{&overflow_correction, "overflow"};
    EXPECT_FALSE(citlali::pipeline::apply_flxscale_correction(
        engine, overflow, logger));
    EXPECT_EQ(engine.calib.flux_conversion_factor(0),
              std::numeric_limits<double>::max());
    EXPECT_EQ(engine.calib.mean_flux_conversion_factor.count(state_key), 0U);

    engine.calib.flux_conversion_factor(0) =
        std::numeric_limits<double>::denorm_min();
    const F005Correction underflow_correction{0.5};
    const F005RawObservation underflow{&underflow_correction, "underflow"};
    EXPECT_FALSE(citlali::pipeline::apply_flxscale_correction(
        engine, underflow, logger));
    EXPECT_EQ(engine.calib.flux_conversion_factor(0),
              std::numeric_limits<double>::denorm_min());
    EXPECT_EQ(engine.calib.mean_flux_conversion_factor.count(state_key), 0U);

    EXPECT_TRUE(engine.calib.apt.at("flxscale").isApprox(
        source_flxscale, 0.0));
    EXPECT_TRUE(engine.calib.apt.at("sens").isApprox(
        source_sensitivity, 0.0));
}

TEST(science_map_fits_products,
     production_observation_correction_reaches_weights_maps_and_noise_variance_I) {
    const F005AppliedCorrectionState uncorrected{1.0, 2.0, 1.0};
    const double correction_factor = 3.0;
    const auto corrected =
        apply_f005_production_correction(correction_factor);
    EXPECT_DOUBLE_EQ(corrected.source_flxscale,
                     uncorrected.source_flxscale);
    EXPECT_DOUBLE_EQ(corrected.source_sensitivity,
                     uncorrected.source_sensitivity);
    EXPECT_DOUBLE_EQ(corrected.applied_correction, correction_factor);

    for (const std::string mode :
         {"approximate", "hybrid", "validated", "full"}) {
        SCOPED_TRACE(mode);
        const auto raw =
            run_f005_production_route(mode, uncorrected, false);
        const auto applied =
            run_f005_production_route(mode, corrected, true);
        EXPECT_DOUBLE_EQ(raw.sample, 1.0);
        EXPECT_DOUBLE_EQ(applied.sample, correction_factor);
        EXPECT_DOUBLE_EQ(raw.compatibility_fcf, 1.0);
        EXPECT_DOUBLE_EQ(applied.compatibility_fcf, correction_factor);
        EXPECT_NEAR(
            applied.detector_weight,
            raw.detector_weight /
                (correction_factor * correction_factor),
            1.0e-14);
        EXPECT_NEAR(applied.map_signal,
                    raw.map_signal * correction_factor, 1.0e-14);
        EXPECT_NEAR(
            applied.map_weight,
            raw.map_weight / (correction_factor * correction_factor),
            1.0e-14);
        EXPECT_NE(raw.realization_minus, 0.0);
        EXPECT_NE(raw.realization_plus, 0.0);
        EXPECT_NEAR(applied.realization_minus,
                    raw.realization_minus * correction_factor, 1.0e-14);
        EXPECT_NEAR(applied.realization_plus,
                    raw.realization_plus * correction_factor, 1.0e-14);
        ASSERT_GT(raw.noise_variance_I, 0.0);
        EXPECT_NEAR(
            applied.noise_variance_I,
            raw.noise_variance_I * correction_factor * correction_factor,
            1.0e-14);
    }
}

TEST(science_map_fits_products, writes_canonical_typed_planes_and_aliases) {
    auto map = make_science_map_buffer();
    CapturedFitsEntry output;
    DummyWcs wcs;

    citlali::pipeline::add_science_map_product_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0,
        science_map_test_logger());

    ASSERT_EQ(output.images.size(), 10U);
    EXPECT_EQ(captured_image(output, "geometric_hits_I").data_type, "int64");
    EXPECT_EQ(captured_image(output, "contributing_hits_I").data_type,
              "int64");
    EXPECT_EQ(captured_image(output, "coadd_observation_count_I").data_type,
              "int64");
    EXPECT_EQ(captured_image(output, "upstream_eligible_exposure_I").data_type,
              "float64");
    EXPECT_EQ(captured_image(output, "retained_exposure_I").data_type,
              "float64");
    EXPECT_EQ(captured_image(output, "normalization_support_I").data_type,
              "uint8");
    EXPECT_EQ(captured_image(output, "science_policy_support_I").data_type,
              "uint8");
    EXPECT_EQ(captured_image(output, "science_valid_I").data_type, "uint8");
    EXPECT_EQ(captured_image(output, "coverage_I").data_type, "float64");
    EXPECT_EQ(captured_image(output, "coverage_bool_I").data_type, "uint8");

    const auto &valid = captured_hdu(output, "science_valid_I").keys;
    EXPECT_EQ(valid.at("DATTYP"), "uint8");
    EXPECT_EQ(valid.at("VALAUTH"), "true");
    EXPECT_EQ(valid.at("ESTTYPE"), valid.at("TYPE"));

    const auto &coverage = captured_hdu(output, "coverage_I").keys;
    EXPECT_EQ(coverage.at("BUNIT"), "detector s");
    EXPECT_EQ(coverage.at("ALIASOF"), "retained_exposure_I");
    EXPECT_EQ(coverage.at("DEPRCATD"), "false");
    EXPECT_EQ(coverage.at("VALAUTH"), "false");

    const auto &coverage_bool =
        captured_hdu(output, "coverage_bool_I").keys;
    EXPECT_EQ(coverage_bool.at("ALIASOF"), "science_policy_support_I");
    EXPECT_EQ(coverage_bool.at("DEPRCATD"), "true");
    EXPECT_EQ(coverage_bool.at("VALAUTH"), "false");
    EXPECT_EQ(coverage_bool.at("WTTHRESH"), "3");
}

TEST(science_map_fits_products, skips_products_declared_unavailable) {
    auto map = make_science_map_buffer(false);
    CapturedFitsEntry output;
    DummyWcs wcs;

    citlali::pipeline::add_science_map_product_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0,
        science_map_test_logger());

    EXPECT_TRUE(captured_has_image(output, "contributing_hits_I"));
    EXPECT_FALSE(captured_has_image(output, "coadd_observation_count_I"));
    EXPECT_TRUE(captured_has_image(output, "science_valid_I"));
}

TEST(science_map_fits_products,
     supported_output_bundle_rejects_all_unavailable_or_missing_inventory) {
    auto map = make_science_map_buffer(false);
    EXPECT_TRUE(citlali::pipeline::science_map_supported_output_bundle_complete(
        map->science_products, map->signal.size(), map->n_rows,
        map->n_cols));

    map->science_products.realized[0].product_available.fill(false);
    EXPECT_FALSE(citlali::pipeline::science_map_supported_output_bundle_complete(
        map->science_products, map->signal.size(), map->n_rows,
        map->n_cols));

    map = make_science_map_buffer(false);
    map->science_products.science_valid.clear();
    EXPECT_FALSE(citlali::pipeline::science_map_supported_output_bundle_complete(
        map->science_products, map->signal.size(), map->n_rows,
        map->n_cols));
}

TEST(science_map_fits_products,
     coadd_writes_f010_hierarchy_without_significance_products) {
    auto map = make_science_map_buffer();
    map->science_products.is_coadd = true;
    map->freeze_raw_science_parent();
    CapturedFitsEntry output;
    DummyWcs wcs;

    EXPECT_NO_THROW(citlali::pipeline::add_coverage_support_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0, true, true, true,
        science_map_test_logger()));

    EXPECT_TRUE(captured_has_image(output, "science_valid_I"));
    EXPECT_FALSE(captured_has_image(output, "formal_standardized_signal_I"));
    EXPECT_FALSE(captured_has_image(
        output, "conditional_stack_scatter_I"));
    EXPECT_FALSE(captured_has_image(
        output, "coefficient_standardized_signal_I"));
    EXPECT_FALSE(captured_has_image(output, "sig2noise_I"));
    EXPECT_FALSE(captured_has_image(output, "sig2noise_pixel_I"));
    EXPECT_FALSE(captured_has_image(output, "point_source_flux_I"));
    EXPECT_FALSE(captured_has_image(output, "point_source_uncertainty_I"));
    EXPECT_FALSE(captured_has_image(
        output, "filtered_pixel_stack_scatter_I"));
    EXPECT_FALSE(captured_has_image(
        output, "conditional_stack_scatter_ratio_I"));
    EXPECT_FALSE(captured_has_image(output, "sig2noise_point_source_I"));
}

TEST(science_map_fits_products,
     selected_frozen_coadd_authority_controls_output_family) {
    auto map = make_science_map_buffer();
    map->freeze_raw_science_parent();
    map->science_products.is_coadd = false;
    CapturedFitsEntry output;
    DummyWcs wcs;

    EXPECT_NO_THROW(citlali::pipeline::add_coverage_support_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0, true, false, true,
        science_map_test_logger()));

    EXPECT_TRUE(captured_has_image(output, "science_valid_I"));
    EXPECT_FALSE(captured_has_image(output, "formal_standardized_signal_I"));
    EXPECT_FALSE(captured_has_image(output, "sig2noise_I"));
}

TEST(science_map_fits_products,
     coadd_primary_weight_omits_uncertainty_metadata) {
    auto map = make_science_map_buffer();
    map->science_products.is_coadd = true;
    CapturedFitsEntry output;
    DummyWcs wcs;

    citlali::pipeline::add_primary_map_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0, false, false, false,
        true, science_map_test_logger());

    const auto &weight = captured_hdu(output, "weight_I").keys;
    EXPECT_EQ(weight.at("PRECSTAT"), "not_established");
    EXPECT_EQ(weight.at("COVSTAT"), "unavailable");
    EXPECT_EQ(weight.find("MEDERR"), weight.end());
}

TEST(science_map_fits_products, rejects_nonidentical_coverage_alias) {
    auto map = make_science_map_buffer();
    map->coverage[0](0, 0) = -0.0;
    map->science_products.retained_exposure[0](0, 0) = 0.0;
    CapturedFitsEntry output;
    DummyWcs wcs;

    EXPECT_THROW(
        citlali::pipeline::add_science_map_product_image_hdus(
            output, map, 0, "", "I", wcs, 2000.0,
            science_map_test_logger()),
        citlali::error::Error);
    EXPECT_TRUE(output.images.empty());
}

TEST(science_map_fits_products,
     filtered_output_carries_immutable_raw_parent_after_live_mutation) {
    auto map = make_science_map_buffer(false);
    map->freeze_raw_science_parent();
    ASSERT_TRUE(map->raw_science_parent);
    const auto raw_digest =
        map->raw_science_parent->realized[0].raw_parent_digest;
    const auto raw_valid = map->raw_science_parent->science_valid[0];

    map->signal[0].setConstant(42.0);
    map->weight[0].setConstant(17.0);
    map->science_products.science_valid[0].setZero();
    map->refresh_science_products_after_coefficient_rescale(0);

    EXPECT_EQ(map->raw_science_parent->realized[0].raw_parent_digest,
              raw_digest);
    EXPECT_TRUE(citlali::pipeline::science_map_planes_bitwise_equal(
        map->raw_science_parent->science_valid[0], raw_valid));

    CapturedFitsEntry output;
    DummyWcs wcs;
    citlali::pipeline::add_science_map_product_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0,
        science_map_test_logger(), true);
    const auto &valid = captured_hdu(output, "science_valid_I").keys;
    EXPECT_EQ(valid.at("RAWSTATE"), "immutable_input");
    EXPECT_EQ(valid.at("RAWPDGST"), raw_digest);
    const auto &coverage = captured_hdu(output, "coverage_I").keys;
    EXPECT_EQ(coverage.at("RAWPDGST"), raw_digest);
}

TEST(science_map_fits_products,
     filtered_output_without_frozen_raw_parent_fails_before_write) {
    auto map = make_science_map_buffer(false);
    CapturedFitsEntry output;
    DummyWcs wcs;
    EXPECT_THROW(
        citlali::pipeline::add_science_map_product_image_hdus(
            output, map, 0, "", "I", wcs, 2000.0,
            science_map_test_logger(), true),
        citlali::error::Error);
    EXPECT_TRUE(output.images.empty());
}

TEST(science_map_fits_products,
     missing_empirical_companions_fail_before_any_primary_or_f010_write) {
    auto map = make_science_map_buffer(false);
    DummyWcs wcs;

    CapturedFitsEntry primary_output;
    EXPECT_THROW(
        citlali::pipeline::add_primary_map_image_hdus(
            primary_output, map, 0, "", "I", wcs, 2000.0, true, true,
            false, false, science_map_test_logger()),
        citlali::error::Error);
    EXPECT_TRUE(primary_output.images.empty());

    CapturedFitsEntry support_output;
    EXPECT_THROW(
        citlali::pipeline::add_coverage_support_image_hdus(
            support_output, map, 0, "", "I", wcs, 2000.0, false, true,
            false, science_map_test_logger()),
        citlali::error::Error);
    EXPECT_TRUE(support_output.images.empty());
}

TEST(science_map_fits_products,
     missing_median_diagnostic_fails_before_primary_write) {
    auto map = make_science_map_buffer(false);
    map->median_err.resize(0);
    CapturedFitsEntry output;
    DummyWcs wcs;
    EXPECT_THROW(
        citlali::pipeline::add_primary_map_image_hdus(
            output, map, 0, "", "I", wcs, 2000.0, false, false,
            false, false, science_map_test_logger()),
        citlali::error::Error);
    EXPECT_TRUE(output.images.empty());
}

TEST(science_map_fits_products,
     products_off_observation_can_prepare_and_publish_median_diagnostic) {
    auto map = make_science_map_buffer(false);
    map->median_err.resize(0);
    map->calc_median_err();
    ASSERT_EQ(map->median_err.size(), 1);
    ASSERT_TRUE(std::isfinite(map->median_err(0)));

    CapturedFitsEntry output;
    DummyWcs wcs;
    EXPECT_NO_THROW(citlali::pipeline::add_primary_map_image_hdus(
        output, map, 0, "", "I", wcs, 2000.0, false, false, false,
        false, science_map_test_logger()));
    ASSERT_EQ(output.images.size(), 2U);
    const auto &weight = captured_hdu(output, "weight_I").keys;
    EXPECT_NE(weight.find("MEDERR"), weight.end());
}

TEST(science_map_fits_products, labels_weights_as_nonprecision_coefficients) {
    CapturedHdu hdu;
    citlali::pipeline::add_weight_map_metadata(
        hdu, "1/(mJy/beam)^2", false);

    EXPECT_EQ(hdu.keys.at("ESTTYPE"),
              "nonprecision_normalization_coefficient");
    EXPECT_EQ(hdu.keys.at("TYPE"), hdu.keys.at("ESTTYPE"));
    EXPECT_EQ(hdu.keys.at("PRECSTAT"), "not_established");
    EXPECT_EQ(hdu.keys.at("COVSTAT"), "unavailable");
    EXPECT_EQ(hdu.keys.at("CALTYPE"), "formal");
    EXPECT_EQ(hdu.keys.at("DESCRIP").find("inverse variance"),
              std::string::npos);
}

struct FitsFileCleanup {
    std::string path;
    ~FitsFileCleanup() { std::remove(path.c_str()); }
};

struct FitsDirectoryCleanup {
    std::filesystem::path path;
    ~FitsDirectoryCleanup() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void configure_production_writer_engine(Engine &engine) {
    engine.logger = science_map_test_logger();
    engine.typed_config.runtime.reduction_type =
        citlali::config::ReductionType::science;
    engine.runtime_config_provenance =
        citlali::config::make_runtime_config_provenance(
            engine.typed_config.runtime, false);
    engine.typed_config.mapmaking.enabled = true;
    engine.typed_config.mapmaking.method = citlali::config::MapMethod::naive;
    engine.typed_config.mapmaking.grouping =
        citlali::config::MapGrouping::array;
    engine.typed_config.coadd.enabled = true;
    engine.typed_config.noise.enabled = true;
    engine.typed_config.noise.n_noise_maps = 2;
    engine.typed_config.noise.write_realizations = true;
    engine.typed_config.noise.products_enabled = true;
    engine.typed_config.post_processing.map_filtering.enabled = false;

    engine.map_indices.n_maps = 1;
    engine.map_indices.maps_to_arrays.resize(1);
    engine.map_indices.maps_to_arrays.setZero();
    engine.map_indices.arrays_to_maps.resize(1);
    engine.map_indices.arrays_to_maps.setZero();
    engine.map_indices.maps_to_stokes.resize(1);
    engine.map_indices.maps_to_stokes.setZero();
    engine.calib.n_arrays = 1;
    engine.calib.arrays.resize(1);
    engine.calib.arrays.setZero();
    engine.omb.map_grouping = "array";
    engine.telescope.pixel_axes = "radec";
    engine.telescope.sim_obs = false;
    engine.telescope.tel_header["Header.Source.Epoch"] =
        Eigen::VectorXd::Constant(1, 2000.0);
    engine.rtcproc.run_polarization = false;
    engine.rtcproc.polarization.stokes_params.clear();
    engine.rtcproc.polarization.stokes_params[0] = "I";
}

TEST(science_map_fits_products,
     calibration_response_basis_separates_requested_effective_and_actual) {
    Engine engine;
    configure_production_writer_engine(engine);
    engine.typed_config.mapmaking.method =
        citlali::config::MapMethod::jinc;
    auto &raw = engine.typed_config.timestream.raw_time_chunk;
    raw.kernel.enabled = true;
    raw.kernel.type = "gaussian";
    raw.filter.enabled = true;
    raw.filter.a_gibbs = 42.0;
    raw.filter.freq_low_Hz = 0.2;
    raw.filter.freq_high_Hz = 16.0;
    raw.filter.n_terms = 32;
    raw.filter.notch.enabled = true;
    raw.iir_filter.enabled = true;
    raw.iir_filter.freq_Hz = 0.1;
    raw.iir_filter.order = 2;
    raw.iir_filter.zero_phase = true;
    raw.downsample.enabled = true;
    raw.downsample.factor = 4;
    engine.rtcproc.run_kernel = true;
    engine.rtcproc.run_tod_filter = true;
    engine.rtcproc.run_tod_notch = true;
    engine.rtcproc.run_tod_iir_highpass = true;
    engine.rtcproc.run_downsample = true;
    engine.rtcproc.filter.a_gibbs = 42.0;
    engine.rtcproc.filter.freq_low_Hz = 0.2;
    engine.rtcproc.filter.freq_high_Hz = 16.0;
    engine.rtcproc.filter.n_terms = 32;
    engine.rtcproc.filter.w0s = {10.0, 20.0};
    engine.rtcproc.filter.qs = {20.0, 40.0};
    engine.rtcproc.downsampler.factor = 4;
    engine.calib.apt["a_fwhm"] = Eigen::VectorXd::Constant(1, 10.0);
    engine.calib.apt["b_fwhm"] = Eigen::VectorXd::Constant(1, 9.0);
    engine.calib.apt["angle"] = Eigen::VectorXd::Constant(1, 0.25);
    engine.raw_timestream_plan.reset_from_request(raw);

    const auto identity =
        citlali::pipeline::calibration_response_identity(engine);
    EXPECT_NE(identity.find("effective_mapmaker_class=jinc"),
              std::string::npos);
    EXPECT_NE(identity.find("effective_map_grouping=array"),
              std::string::npos);
    EXPECT_NE(identity.find("effective_kernel_enabled=true"),
              std::string::npos);
    EXPECT_NE(identity.find("effective_kernel_class=gaussian"),
              std::string::npos);
    EXPECT_NE(identity.find("effective_fir_state=scheduled"),
              std::string::npos);
    EXPECT_NE(identity.find("effective_fir_a_gibbs=0x1.5p+5"),
              std::string::npos);
    EXPECT_NE(identity.find(
                  "effective_fir_application_sample_rate_hz="),
              std::string::npos);
    EXPECT_NE(identity.find("effective_fixed_notch_state=scheduled"),
              std::string::npos);
    EXPECT_NE(identity.find("effective_iir_highpass_enabled=true"),
              std::string::npos);
    EXPECT_NE(identity.find(
                  "effective_iir_highpass_application_sample_rate_hz="),
              std::string::npos);
    EXPECT_NE(identity.find("effective_downsample_enabled=true"),
              std::string::npos);
    EXPECT_NE(identity.find("actual_applied_notch_count=0"),
              std::string::npos);
    EXPECT_EQ(identity.find("actual_applied_notch[0]"), std::string::npos);
    EXPECT_NE(identity.find("no_empirical_response_fidelity"),
              std::string::npos);

    engine.calib.apt["angle"](0) = 0.5;
    EXPECT_NE(citlali::pipeline::calibration_response_identity(engine),
              identity);
}

TEST(science_map_fits_products,
     calibration_response_identity_distinguishes_secondary_detector_notches) {
    Engine engine;
    configure_production_writer_engine(engine);
    engine.rtcproc.begin_observation_applied_response_history();
    for (const auto &[ordinal, center, width] :
         std::vector<std::tuple<Eigen::Index, double, double>>{
             {0, 11.0, 0.2}, {1, 19.0, 0.4}}) {
        timestream::RTCProc::RTCAppliedResponseNotch notch;
        notch.phase = "rtc";
        notch.stage = "post_filter";
        notch.scan = 3;
        notch.scope = "detector";
        notch.detector = 7;
        notch.ordinal = ordinal;
        notch.center_hz = center;
        notch.width_hz = width;
        engine.rtcproc.record_applied_response_notch(std::move(notch));
    }
    const auto snapshot =
        engine.rtcproc.snapshot_applied_response_notches();
    ASSERT_EQ(snapshot.at(3).size(), 2U);
    const auto first =
        citlali::pipeline::calibration_response_identity(engine);
    engine.rtcproc.rtc_applied_response_notches_by_scan[3][1]
        .center_hz = 23.0;
    const auto second =
        citlali::pipeline::calibration_response_identity(engine);
    EXPECT_NE(first, second);
    EXPECT_NE(first.find("ordinal=1"), std::string::npos);
    EXPECT_NE(first.find("center_hz=0x1.3p+4"), std::string::npos);
    const auto consumed =
        engine.rtcproc.consume_applied_response_notches();
    ASSERT_EQ(consumed.at(3).size(), 2U);
    EXPECT_TRUE(engine.rtcproc.snapshot_applied_response_notches().empty());
    const auto repeated = engine.rtcproc.consume_applied_response_notches();
    ASSERT_EQ(repeated.at(3).size(), 2U);
    EXPECT_DOUBLE_EQ(repeated.at(3)[1].center_hz,
                     consumed.at(3)[1].center_hz);
}

TEST(science_map_fits_products,
     calibration_response_identity_keeps_dormant_request_out_of_realized_state) {
    Engine engine;
    configure_production_writer_engine(engine);
    auto &raw = engine.typed_config.timestream.raw_time_chunk;
    raw.filter.enabled = false;
    raw.filter.a_gibbs = 31.0;
    raw.iir_filter.enabled = false;
    raw.iir_filter.freq_Hz = 0.1;
    engine.rtcproc.run_tod_filter = false;
    engine.rtcproc.run_tod_iir_highpass = false;
    engine.raw_timestream_plan.reset_from_request(raw);
    const auto first =
        citlali::pipeline::calibration_response_identity(engine);
    EXPECT_NE(first.find("effective_fir_state=inactive"), std::string::npos);
    EXPECT_EQ(first.find("effective_fir_a_gibbs"), std::string::npos);
    EXPECT_EQ(first.find("effective_fir_application_sample_rate_hz"),
              std::string::npos);
    EXPECT_NE(first.find("effective_iir_highpass_enabled=false"),
              std::string::npos);
    EXPECT_EQ(first.find(
                  "effective_iir_highpass_application_sample_rate_hz"),
              std::string::npos);
    raw.filter.a_gibbs = 49.0;
    const auto second =
        citlali::pipeline::calibration_response_identity(engine);
    EXPECT_EQ(first, second);
    EXPECT_EQ(second.find("effective_fir_a_gibbs"), std::string::npos);
    EXPECT_EQ(second.find("effective_fir_application_sample_rate_hz"),
              std::string::npos);
    EXPECT_EQ(second.find(
                  "effective_iir_highpass_application_sample_rate_hz"),
              std::string::npos);
    engine.raw_timestream_plan.requested.filter.a_gibbs = 51.0;
    EXPECT_NE(citlali::pipeline::calibration_response_identity(engine),
              first);
}

TEST(science_map_fits_products,
     calibration_response_identity_binds_fir_application_sample_rate) {
    const auto identity_at_sample_rate = [](double sample_rate_hz) {
        Engine engine;
        configure_production_writer_engine(engine);
        engine.telescope.fsmp = sample_rate_hz;
        auto &raw = engine.typed_config.timestream.raw_time_chunk;
        raw.filter.enabled = true;
        raw.filter.freq_low_Hz = 0.2;
        raw.filter.freq_high_Hz = 16.0;
        raw.filter.n_terms = 32;
        raw.filter.a_gibbs = 42.0;
        engine.rtcproc.run_tod_filter = true;
        engine.rtcproc.filter.freq_low_Hz = 0.2;
        engine.rtcproc.filter.freq_high_Hz = 16.0;
        engine.rtcproc.filter.n_terms = 32;
        engine.rtcproc.filter.a_gibbs = 42.0;
        engine.raw_timestream_plan.reset_from_request(raw);
        return citlali::pipeline::calibration_response_identity(engine);
    };

    const auto at_100_hz = identity_at_sample_rate(100.0);
    const auto at_200_hz = identity_at_sample_rate(200.0);
    EXPECT_NE(at_100_hz, at_200_hz);
    EXPECT_NE(at_100_hz.find(
                  "effective_fir_application_sample_rate_hz=0x1.9p+6"),
              std::string::npos);
    EXPECT_NE(at_200_hz.find(
                  "effective_fir_application_sample_rate_hz=0x1.9p+7"),
              std::string::npos);
}

TEST(science_map_fits_products,
     calibration_response_identity_binds_iir_highpass_application_sample_rate) {
    const auto identity_at_sample_rate = [](double sample_rate_hz) {
        Engine engine;
        configure_production_writer_engine(engine);
        engine.telescope.fsmp = sample_rate_hz;
        auto &raw = engine.typed_config.timestream.raw_time_chunk;
        raw.iir_filter.enabled = true;
        raw.iir_filter.freq_Hz = 0.1;
        raw.iir_filter.order = 2;
        raw.iir_filter.zero_phase = true;
        engine.rtcproc.run_tod_iir_highpass = true;
        engine.rtcproc.filter.iir_highpass_freq_Hz = 0.1;
        engine.rtcproc.filter.iir_highpass_order = 2;
        engine.rtcproc.filter.iir_highpass_zero_phase = true;
        engine.raw_timestream_plan.reset_from_request(raw);
        return citlali::pipeline::calibration_response_identity(engine);
    };

    const auto at_100_hz = identity_at_sample_rate(100.0);
    const auto at_200_hz = identity_at_sample_rate(200.0);
    EXPECT_NE(at_100_hz, at_200_hz);
    EXPECT_NE(at_100_hz.find(
                  "effective_iir_highpass_application_sample_rate_hz=0x1.9p+6"),
              std::string::npos);
    EXPECT_NE(at_200_hz.find(
                  "effective_iir_highpass_application_sample_rate_hz=0x1.9p+7"),
              std::string::npos);
}

TEST(science_map_fits_products,
     calibration_response_identity_binds_notch_application_sample_rate) {
    const auto identity_at_sample_rate = [](double sample_rate_hz) {
        Engine engine;
        configure_production_writer_engine(engine);
        engine.telescope.fsmp = sample_rate_hz;
        engine.raw_timestream_plan.reset_from_request(
            engine.typed_config.timestream.raw_time_chunk);
        engine.rtcproc.begin_reduced_observation("152390", 0);
        engine.rtcproc.begin_observation_applied_response_history();

        timestream::RTCProc::RTCAppliedResponseNotch notch;
        notch.stage = "configured_filter";
        notch.scan = 0;
        notch.scope = "fixed";
        notch.ordinal = 0;
        notch.center_hz = 10.0;
        notch.width_hz = 2.0;
        notch.sample_rate_hz = sample_rate_hz;
        engine.rtcproc.record_applied_response_notch(std::move(notch));
        return citlali::pipeline::calibration_response_identity(engine);
    };

    const auto at_100_hz = identity_at_sample_rate(100.0);
    const auto at_200_hz = identity_at_sample_rate(200.0);
    EXPECT_NE(at_100_hz, at_200_hz);
    EXPECT_NE(at_100_hz.find("sample_rate_hz=0x1.9p+6"),
              std::string::npos);
    EXPECT_NE(at_200_hz.find("sample_rate_hz=0x1.9p+7"),
              std::string::npos);
}

TEST(science_map_fits_products,
     calibration_response_identity_binds_actual_notch_reduced_observation) {
    const auto identity_for_observation = [](const std::string &observation) {
        Engine engine;
        configure_production_writer_engine(engine);
        engine.telescope.fsmp = 100.0;
        engine.raw_timestream_plan.reset_from_request(
            engine.typed_config.timestream.raw_time_chunk);
        engine.rtcproc.begin_reduced_observation(observation, 0);
        engine.rtcproc.begin_observation_applied_response_history();

        timestream::RTCProc::RTCAppliedResponseNotch notch;
        notch.stage = "configured_filter";
        notch.scan = 0;
        notch.scope = "fixed";
        notch.ordinal = 0;
        notch.center_hz = 10.0;
        notch.width_hz = 2.0;
        notch.sample_rate_hz = 100.0;
        engine.rtcproc.record_applied_response_notch(std::move(notch));
        return citlali::pipeline::calibration_response_identity(engine);
    };

    const auto observation_152390 = identity_for_observation("152390");
    const auto observation_152391 = identity_for_observation("152391");
    EXPECT_NE(observation_152390, observation_152391);
    EXPECT_NE(observation_152390.find(
                  "reduced_observation_identity=152390"),
              std::string::npos);
    EXPECT_NE(observation_152391.find(
                  "reduced_observation_identity=152391"),
              std::string::npos);
}

TEST(science_map_fits_products,
     actual_notch_application_points_record_complete_rtc_and_ptc_state) {
    using Data =
        timestream::TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd>;
    timestream::RTCProc rtcproc;
    rtcproc.logger = science_map_test_logger();
    rtcproc.run_kernel = false;
    rtcproc.begin_observation_applied_response_history();

    Data data;
    data.scans.data = Eigen::MatrixXd::Zero(256, 2);
    data.flags.data =
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
            256, 2, false);
    data.index.data = 4;
    for (Eigen::Index sample = 0; sample < data.scans.data.rows(); ++sample) {
        const double phase =
            2.0 * pi * 8.0 * static_cast<double>(sample) / 64.0;
        data.scans.data(sample, 0) = std::sin(phase);
        data.scans.data(sample, 1) = 0.8 * std::sin(phase);
    }

    auto audit = rtcproc.line_audit;
    audit.enabled = true;
    audit.fixed_notch_enabled = true;
    audit.fixed_notch_freqs_hz = {8.0};
    audit.fixed_notch_widths_hz = {0.5};
    ASSERT_EQ(rtcproc.apply_rtc_line_audit_fixed_notches(
                  data, 64.0, audit),
              1);

    data.index.data = 5;
    audit.fixed_notch_enabled = false;
    audit.apply_shared_notches = true;
    audit.apply_min_support_networks = 1;
    audit.apply_min_detector_frac = 0.0;
    audit.apply_min_common_mode_prominence = 0.0;
    audit.apply_max_notches = 1;
    timestream::RTCProc::RTCNetworkDiagSummary network;
    network.nw = 0;
    timestream::RTCProc::RTCLineAuditSharedCandidate candidate;
    candidate.freq_hz = 8.0;
    candidate.width_hz = 0.5;
    candidate.freq_min_hz = 7.75;
    candidate.freq_max_hz = 8.25;
    candidate.detector_frac = 1.0;
    candidate.common_mode_prominence = 1000.0;
    candidate.notch_score = 1000.0;
    candidate.recommend_notch = true;
    network.line_audit_shared_candidates.push_back(candidate);
    network.post_line_audit.shared_candidates.push_back(candidate);
    rtcproc.rtc_network_summary_by_scan[5] = {network};
    ASSERT_EQ(rtcproc.apply_rtc_line_audit_shared_notches(
                  data, 64.0, audit, false),
              1);

    data.index.data = 6;
    for (Eigen::Index sample = 0; sample < data.scans.data.rows(); ++sample) {
        const double phase =
            2.0 * pi * 8.0 * static_cast<double>(sample) / 64.0;
        data.scans.data(sample, 0) = std::sin(phase) +
            0.01 * std::sin(2.0 * pi * 3.0 *
                            static_cast<double>(sample) / 64.0);
        data.scans.data(sample, 1) = 0.8 * std::sin(phase) +
            0.01 * std::cos(2.0 * pi * 5.0 *
                            static_cast<double>(sample) / 64.0);
    }
    audit.apply_shared_notches = false;
    audit.post_filter_apply_detector_notches = true;
    audit.line_min_hz = 2.0;
    audit.line_max_hz = 20.0;
    audit.segment_sec = 2.0;
    audit.min_segment_sec = 1.0;
    audit.min_windows = 1;
    audit.min_good_frac = 0.9;
    audit.continuum_radius_bins = 2;
    audit.detector_notch_min_prominence = 2.0;
    audit.detector_notch_min_line_power_frac = 0.0;
    audit.detector_notch_max_notches = 1;
    audit.detector_notch_min_width_hz = 0.25;
    audit.detector_notch_max_width_hz = 1.0;
    ASSERT_GT(rtcproc.apply_rtc_line_audit_detector_notches(
                  data, 64.0, audit, 0, data.scans.data.rows()),
              0);

    data.index.data = 7;
    audit.post_filter_apply_detector_notches = false;
    audit.fixed_notch_enabled = true;
    timestream::RTCProc::RTCResponseApplicationContext ptc_context;
    ptc_context.phase = "ptc";
    ptc_context.stage = "model_protected";
    ptc_context.scan = 7;
    ptc_context.ptc_iteration = 2;
    ptc_context.model_subtracted = true;
    ASSERT_EQ(rtcproc.apply_rtc_line_audit_fixed_notches(
                  data, 64.0, audit, ptc_context),
              1);

    data.index.data = 8;
    audit.fixed_notch_enabled = false;
    audit.apply_shared_notches = true;
    rtcproc.rtc_network_summary_by_scan[8] = {network};
    ptc_context.scan = 8;
    ptc_context.ptc_iteration = 3;
    ASSERT_EQ(rtcproc.apply_rtc_line_audit_shared_notches(
                  data, 64.0, audit, true, ptc_context),
              1);

    data.index.data = 9;
    for (Eigen::Index sample = 0; sample < data.scans.data.rows(); ++sample) {
        const double phase =
            2.0 * pi * 8.0 * static_cast<double>(sample) / 64.0;
        data.scans.data(sample, 0) = std::sin(phase) +
            0.01 * std::sin(2.0 * pi * 3.0 *
                            static_cast<double>(sample) / 64.0);
        data.scans.data(sample, 1) = 0.8 * std::sin(phase) +
            0.01 * std::cos(2.0 * pi * 5.0 *
                            static_cast<double>(sample) / 64.0);
    }
    audit.apply_shared_notches = false;
    audit.post_filter_apply_detector_notches = true;
    ptc_context.scan = 9;
    ptc_context.ptc_iteration = 4;
    ASSERT_GT(rtcproc.apply_rtc_line_audit_detector_notches(
                  data, 64.0, audit, 0, data.scans.data.rows(),
                  ptc_context),
              0);

    const auto history = rtcproc.snapshot_applied_response_notches();
    ASSERT_EQ(history.at(4).size(), 1U);
    EXPECT_EQ(history.at(4).front().phase, "rtc");
    EXPECT_EQ(history.at(4).front().scope, "fixed");
    EXPECT_EQ(history.at(4).front().ordinal, 0);
    ASSERT_EQ(history.at(5).size(), 1U);
    EXPECT_EQ(history.at(5).front().scope, "shared");
    ASSERT_FALSE(history.at(6).empty());
    EXPECT_EQ(history.at(6).front().scope, "detector");
    EXPECT_GE(history.at(6).front().detector, 0);
    ASSERT_EQ(history.at(7).size(), 1U);
    EXPECT_EQ(history.at(7).front().phase, "ptc");
    EXPECT_EQ(history.at(7).front().ptc_iteration, 2);
    EXPECT_TRUE(history.at(7).front().model_subtracted);
    ASSERT_EQ(history.at(8).size(), 1U);
    EXPECT_EQ(history.at(8).front().phase, "ptc");
    EXPECT_EQ(history.at(8).front().scope, "shared");
    EXPECT_EQ(history.at(8).front().ptc_iteration, 3);
    ASSERT_FALSE(history.at(9).empty());
    EXPECT_EQ(history.at(9).front().phase, "ptc");
    EXPECT_EQ(history.at(9).front().scope, "detector");
    EXPECT_EQ(history.at(9).front().ptc_iteration, 4);
    for (const auto &[scan, records] : history) {
        for (const auto &record : records) {
            EXPECT_EQ(record.scan, scan);
            EXPECT_EQ(record.geometry, "center_hz_width_hz");
            EXPECT_FALSE(record.phase_convention.empty());
            EXPECT_TRUE(std::isfinite(record.center_hz));
            EXPECT_TRUE(std::isfinite(record.width_hz));
            EXPECT_DOUBLE_EQ(record.sample_rate_hz, 64.0);
        }
    }
}

TEST(science_map_fits_products,
     response_history_lifecycle_resets_and_preserves_finalized_snapshot) {
    timestream::RTCProc rtcproc;
    EXPECT_FALSE(rtcproc.applied_response_history_available());
    EXPECT_TRUE(rtcproc.consume_applied_response_notches().empty());
    EXPECT_FALSE(rtcproc.applied_response_history_available());
    rtcproc.begin_reduced_observation("observation-a", 0);
    rtcproc.begin_observation_applied_response_history();
    timestream::RTCProc::RTCAppliedResponseNotch interrupted;
    interrupted.scan = 0;
    interrupted.stage = "pre_filter";
    interrupted.scope = "fixed";
    interrupted.ordinal = 0;
    interrupted.center_hz = 8.0;
    interrupted.width_hz = 0.5;
    rtcproc.record_applied_response_notch(interrupted);

    rtcproc.begin_observation_applied_response_history();
    EXPECT_TRUE(rtcproc.snapshot_applied_response_notches().empty());
    const auto unavailable = rtcproc.consume_applied_response_notches();
    EXPECT_TRUE(unavailable.empty());
    EXPECT_TRUE(rtcproc.applied_response_history_available());
    EXPECT_TRUE(rtcproc.consume_applied_response_notches().empty());
    EXPECT_THROW(rtcproc.record_applied_response_notch(interrupted),
                 std::logic_error);

    rtcproc.begin_observation_applied_response_history();
    for (const Eigen::Index scan : {0, 1}) {
        auto record = interrupted;
        record.scan = scan;
        record.center_hz += static_cast<double>(scan);
        rtcproc.record_applied_response_notch(std::move(record));
    }
    const auto multiscan = rtcproc.consume_applied_response_notches();
    ASSERT_EQ(multiscan.size(), 2U);
    EXPECT_EQ(multiscan.at(0).front().scan, 0);
    EXPECT_EQ(multiscan.at(1).front().scan, 1);

    rtcproc.begin_observation_applied_response_history();
    auto reused_scan = interrupted;
    reused_scan.center_hz = 12.0;
    rtcproc.record_applied_response_notch(reused_scan);
    const auto reused = rtcproc.consume_applied_response_notches();
    ASSERT_EQ(reused.size(), 1U);
    EXPECT_DOUBLE_EQ(reused.at(0).front().center_hz, 12.0);
    EXPECT_EQ(reused.at(0).front().reduced_observation_identity,
              "observation-a");
    EXPECT_EQ(reused.at(0).front().fruit_iteration, 0);
}

TEST(science_map_fits_products,
     finalized_joins_are_idempotent_homogeneous_and_fail_closed) {
    timestream::RTCProc rtcproc;
    rtcproc.begin_reduced_observation("obs-a", 0);
    rtcproc.record_finalized_calibration_join("obs-a", "cal-a", "pkg-a");
    EXPECT_NO_THROW(rtcproc.record_finalized_calibration_join(
        "obs-a", "cal-a", "pkg-a"));
    EXPECT_THROW(rtcproc.record_finalized_calibration_join(
                     "obs-a", "cal-b", "pkg-b"),
                 std::logic_error);
    rtcproc.begin_reduced_observation("obs-b", 0);
    rtcproc.record_finalized_calibration_join("obs-b", "cal-a", "pkg-a");
    const auto homogeneous =
        rtcproc.homogeneous_calibration_join({"obs-a", "obs-b"});
    EXPECT_EQ(homogeneous.calibration_identity, "cal-a");
    EXPECT_EQ(homogeneous.package_identity, "pkg-a");
    rtcproc.begin_reduced_observation("obs-c", 0);
    rtcproc.record_finalized_calibration_join("obs-c", "cal-c", "pkg-c");
    EXPECT_THROW(rtcproc.homogeneous_calibration_join({"obs-a", "obs-c"}),
                 std::logic_error);
    EXPECT_THROW(rtcproc.homogeneous_calibration_join({"obs-missing"}),
                 std::logic_error);

    rtcproc.begin_reduced_observation("obs-a", 1);
    EXPECT_TRUE(rtcproc.finalized_calibration_joins.empty());
    EXPECT_NO_THROW(rtcproc.record_finalized_calibration_join(
        "obs-a", "cal-fruit-1", "pkg-fruit-1"));
    ASSERT_EQ(rtcproc.finalized_calibration_joins.size(), 1U);
    EXPECT_EQ(rtcproc.finalized_calibration_joins.front().fruit_iteration, 1);
    EXPECT_THROW(rtcproc.homogeneous_calibration_join({"obs-a", "obs-b"}),
                 std::logic_error);

    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(citlali::config::RawTimeChunkConfig{});
    auto &observation = plan.begin_observation();
    observation.calibration_identity = "cal-a";
    observation.calibration_package_identity = "pkg-a";
    observation.calibration_response_identity = "response-a";
    citlali::pipeline::complete_raw_timestream_observation(plan, 2, 3);
    EXPECT_NO_THROW(citlali::pipeline::complete_raw_timestream_observation(
        plan, 2, 3));
    EXPECT_THROW(citlali::pipeline::complete_raw_timestream_observation(
                     plan, 3, 3),
                 std::logic_error);
}

TEST(science_map_fits_products,
     canonical_package_precedes_dependents_and_survives_later_failure) {
    std::vector<std::string> events;
    const auto package =
        citlali::pipeline::publish_canonical_package_before_linked_products(
            [&]() {
                events.push_back("package_published_and_validated");
                return std::string{"canonical-package"};
            },
            [&]() { events.push_back("dependent_published"); });
    EXPECT_EQ(package, "canonical-package");
    EXPECT_EQ(events,
              (std::vector<std::string>{
                  "package_published_and_validated", "dependent_published"}));

    events.clear();
    EXPECT_THROW(
        citlali::pipeline::publish_canonical_package_before_linked_products(
            [&]() {
                events.push_back("package_published_and_validated");
                return std::string{"orphan-package"};
            },
            [&]() {
                events.push_back("dependent_failed_before_publication");
                throw std::runtime_error("dependent output failed");
            }),
        std::runtime_error);
    EXPECT_EQ(events,
              (std::vector<std::string>{
                  "package_published_and_validated",
                  "dependent_failed_before_publication"}));

    const auto package_dir = std::filesystem::path(testing::TempDir()) /
        "citlali-f008-package-first-failure";
    std::filesystem::remove_all(package_dir);
    std::filesystem::create_directories(package_dir);
    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(citlali::config::RawTimeChunkConfig{});
    plan.begin_observation();
    citlali::pipeline::complete_raw_timestream_observation(plan, 2, 1);
    const auto dependent_path = package_dir / "dependent-product.fits";
    EXPECT_THROW(
        citlali::pipeline::publish_canonical_package_before_linked_products(
            [&]() {
                citlali::pipeline::write_raw_timestream_provenance_file(
                    package_dir, plan);
                return citlali::pipeline::raw_timestream_provenance_path(
                    package_dir);
            },
            [&]() {
                throw std::runtime_error(
                    "dependent failed before atomic publication");
            }),
        std::runtime_error);
    EXPECT_TRUE(std::filesystem::is_regular_file(
        citlali::pipeline::raw_timestream_provenance_path(package_dir)));
    EXPECT_FALSE(std::filesystem::exists(dependent_path));
    std::filesystem::remove_all(package_dir);
}

template <class EngineType>
void admit_production_calibration_fixture(EngineType &engine,
                                          bool finalize = true,
                                          bool active_extinction = false) {
    if (engine.observation_identity.obsnum.empty()) {
        engine.observation_identity.obsnum = "152390";
    }
    engine.rtcproc.begin_reduced_observation(
        engine.observation_identity.obsnum, engine.iteration.fruit_iter);
    engine.rtcproc.begin_observation_applied_response_history();
    timestream::CalibrationProductAdmissionInputs inputs;
    inputs.target_unit = "mJy/beam";
    inputs.calibration_requested = true;
    inputs.acquisition_identity_available = true;
    inputs.acquisition_identity_valid = true;
    inputs.acquisition_identity_detail = "production writer fixture";
    inputs.apt_lineage_available = true;
    inputs.apt_lineage_valid = true;
    inputs.apt_lineage_detail = "production writer lineage fixture";
    inputs.apt_artifact_sha256 =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    inputs.apt_row_association_sha256 =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    inputs.apt_observation_identity = "152390";
    inputs.apt_selected_source = "Neptune";
    inputs.acquisition_binding_sha256 =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    inputs.raw_observation_identity = "production-writer-raw-identity";
    inputs.acquisition_binding_mode = "production_writer_fixture_join";
    inputs.acquisition_key_schema = "fixture+network+local_tone";
    inputs.response_identity =
        "calibration-response-basis-provenance-v1;"
        "originating_beam=fixture;realized_mapmaker_class=naive;"
        "realized_map_grouping=array;realized_filtering=disabled;"
        "semantics=provenance_only";
    inputs.atmosphere_operator_id =
        std::string{engine.rtcproc.calibration.operator_id()};
    inputs.atmosphere_operator_contract_sha256 =
        std::string{engine.rtcproc.calibration.operator_contract_sha256()};
    inputs.atmosphere_node_table_sha256 =
        std::string{engine.rtcproc.calibration.operator_nodes_sha256()};
    inputs.passband_set_id =
        std::string{engine.rtcproc.calibration.passband_set_id()};
    inputs.reference_profile_id =
        std::string{engine.rtcproc.calibration.reference_profile_id()};
    inputs.tau225 = 0.0;
    inputs.target_unit_factor = Eigen::VectorXd::Ones(1);
    inputs.observation_flxscale_correction_applied = true;
    inputs.applied_observation_flxscale_correction = 3.0;
    inputs.observation_flxscale_correction_state = "applied_once";
    inputs.observation_flxscale_correction_source_identity =
        std::string{timestream::CalibrationProduct::
                        observation_correction_source_identity};
    inputs.observation_flxscale_correction_recipient_identity =
        inputs.raw_observation_identity;
    inputs.detector_flxscale = Eigen::VectorXd::Constant(1, 2.0);
    inputs.detector_beam_major_fwhm_arcsec =
        Eigen::VectorXd::Constant(1, 10.0);
    inputs.detector_beam_minor_fwhm_arcsec =
        Eigen::VectorXd::Constant(1, 9.0);
    inputs.minimum_extinction_correction = Eigen::VectorXd::Ones(1);
    inputs.maximum_extinction_correction = Eigen::VectorXd::Ones(1);
    inputs.applied_sample_extinction_state.available = true;
    if (active_extinction) {
        inputs.extinction_requested = true;
        inputs.minimum_extinction_correction =
            Eigen::VectorXd::Constant(1, std::exp(0.1));
        inputs.maximum_extinction_correction =
            inputs.minimum_extinction_correction;
        auto &state = inputs.applied_sample_extinction_state;
        state.active = true;
        state.sample_elevation_rad = Eigen::VectorXd::Ones(1);
        state.los_tau_by_array.emplace(
            0, Eigen::VectorXd::Constant(1, 0.1));
        state.los_tau_by_array.emplace(
            1, Eigen::VectorXd::Constant(1, 0.2));
        state.los_tau_by_array.emplace(
            2, Eigen::VectorXd::Constant(1, 0.3));
    }
    inputs.applied_sample_extinction_state_sha256 =
        timestream::applied_sample_extinction_state_identity(
            inputs.applied_sample_extinction_state);
    inputs.package_lineage.selected_apt_source_path = "fixture.ecsv";
    inputs.package_lineage.selected_apt_sha256 =
        inputs.apt_artifact_sha256;
    inputs.package_lineage.apt_row_association_sha256 =
        inputs.apt_row_association_sha256;
    inputs.package_lineage.raw_artifacts.push_back(
        {"fixture-raw.nc", "fixture-raw-digest", "toltec0", 0,
         {1.0e9}});
    timestream::CalibrationLineageRow lineage_row;
    lineage_row.ordered_detector_index = 0;
    lineage_row.selected_source_row_index = 0;
    lineage_row.network = 0;
    lineage_row.network_local_tone = 0;
    lineage_row.absolute_tone_frequency_hz = 1.0e9;
    lineage_row.uid = "0";
    lineage_row.eligible = true;
    lineage_row.validity_basis = "fixture-valid-row";
    lineage_row.stable_association = "fixture-stable-row";
    inputs.package_lineage.ordered_rows.push_back(
        std::move(lineage_row));
    engine.rtcproc.calibration.admit_product(inputs);
    ASSERT_TRUE(engine.rtcproc.calibration.product.valid());
    if (finalize) {
        citlali::pipeline::finalize_complete_calibration_product_identity(
            engine);
    }
}

TEST(science_map_fits_products,
     reused_calibrator_apt_joins_distinct_reduced_observations) {
    Engine engine;
    configure_production_writer_engine(engine);
    engine.observation_identity.obsnum = "science-observation-a";
    admit_production_calibration_fixture(engine);
    const auto calibration_identity =
        engine.rtcproc.calibration.product.calibration_identity;
    const auto package_identity =
        engine.rtcproc.calibration.product.package_identity;

    engine.observation_identity.obsnum = "science-observation-b";
    admit_production_calibration_fixture(engine);
    EXPECT_EQ(engine.rtcproc.calibration.product.calibration_identity,
              calibration_identity);
    EXPECT_EQ(engine.rtcproc.calibration.product.package_identity,
              package_identity);
    const auto joined = engine.rtcproc.homogeneous_calibration_join(
        {"science-observation-a", "science-observation-b"});
    EXPECT_EQ(joined.calibration_identity, calibration_identity);
    EXPECT_EQ(joined.package_identity, package_identity);
}

TEST(science_map_fits_products,
     canonical_yaml_reopens_recomputes_identities_and_preserves_prior_final) {
    const auto package_dir = std::filesystem::path(testing::TempDir()) /
        "citlali-f008-canonical-yaml-readback";
    std::filesystem::remove_all(package_dir);
    std::filesystem::create_directories(package_dir);
    const auto source_apt = package_dir / "source-selected-apt.ecsv";
    {
        std::ofstream output(source_apt, std::ios::out | std::ios::trunc);
        output << "# %ECSV 1.0\n"
                  "# ---\n"
                  "# datatype:\n"
                  "# - {name: uid, datatype: int64}\n"
                  "# schema: astropy-2.0\n"
                  "uid\n42\n";
    }

    Engine engine;
    configure_production_writer_engine(engine);
    admit_production_calibration_fixture(engine, false);
    auto &product = engine.rtcproc.calibration.product;
    product.package_lineage.selected_apt_source_path = source_apt.string();
    product.package_lineage.apt_observation_identity =
        product.apt_observation_identity;
    product.package_lineage.apt_matched_observation_identity =
        product.apt_matched_observation_identity;
    product.package_lineage.apt_selected_source =
        product.apt_selected_source;
    product.apt_artifact_sha256 =
        citlali::utils::sha256_file(source_apt);
    product.package_lineage.selected_apt_sha256 =
        product.apt_artifact_sha256;
    citlali::pipeline::finalize_complete_calibration_product_identity(
        engine);

    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(citlali::config::RawTimeChunkConfig{});
    auto &observation = plan.begin_observation();
    observation.reduced_observation_identity = "152390";
    observation.canonical_calibration_product = product;
    citlali::pipeline::complete_raw_timestream_observation(plan, 2, 1);
    ASSERT_NO_THROW(
        citlali::pipeline::write_raw_timestream_provenance_file(
            package_dir, plan));

    const auto yaml_path =
        citlali::pipeline::raw_timestream_provenance_path(package_dir);
    const auto reopened = YAML::LoadFile(yaml_path.string());
    const auto lineage = reopened["calibration_lineage"]["value"];
    EXPECT_EQ(lineage["calibration_identity"].as<std::string>(),
              product.calibration_identity);
    EXPECT_EQ(lineage["package_identity"].as<std::string>(),
              product.package_identity);
    EXPECT_EQ(
        citlali::pipeline::recomputed_calibration_identity(product),
        product.calibration_identity);
    EXPECT_EQ(
        timestream::calibration_package_identity(product),
        product.package_identity);
    const auto accepted_digest = citlali::utils::sha256_file(yaml_path);

    auto forged_plan = plan;
    auto &forged_product = *forged_plan.observation
        ->canonical_calibration_product;
    forged_product.package_identity =
        "forged-package-identity-that-does-not-recompute";
    EXPECT_THROW(
        citlali::pipeline::write_raw_timestream_provenance_file(
            package_dir, forged_plan),
        std::runtime_error);
    EXPECT_EQ(citlali::utils::sha256_file(yaml_path), accepted_digest);
    EXPECT_FALSE(std::filesystem::exists(yaml_path.string() + ".tmp"));
    EXPECT_FALSE(std::filesystem::exists(
        yaml_path.string() + ".replace-backup"));
    std::filesystem::remove_all(package_dir);
}

TEST(science_map_fits_products,
     tod_only_metadata_reopens_with_finalized_calid_and_pkgid) {
    Engine engine;
    configure_production_writer_engine(engine);
    admit_production_calibration_fixture(engine);
    engine.calib.arrays.resize(1);
    engine.calib.arrays.setZero();
    std::map<int, std::string> array_names{{0, "a1100"}};
    std::map<std::string, Eigen::VectorXd> telescope_data;
    const auto path = std::filesystem::path(testing::TempDir()) /
        "citlali-f007-tod-only-calibration-join.nc";
    std::filesystem::remove(path);
    write_netcdf_atomic(path.string(), [&](netCDF::NcFile &file) {
        citlali::pipeline::add_tod_mean_tau_vars(
            file, false, engine.rtcproc, telescope_data, 0.0,
            engine.calib, array_names);
    });

    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    const auto read_string = [&](const std::string &name) {
        auto variable = file.getVar(name);
        char *raw_value = nullptr;
        const int status =
            nc_get_var_string(file.getId(), variable.getId(), &raw_value);
        if (status != NC_NOERR) {
            throw std::runtime_error(nc_strerror(status));
        }
        const std::string value = raw_value == nullptr
            ? std::string{} : std::string{raw_value};
        if (raw_value != nullptr) {
            nc_free_string(1, &raw_value);
        }
        return value;
    };
    EXPECT_EQ(read_string("CALID"),
              engine.rtcproc.calibration.product.calibration_identity);
    EXPECT_EQ(read_string("CALPKGID"),
              engine.rtcproc.calibration.product.package_identity);
    EXPECT_EQ(read_string("CAL.CALIBRATION_IDENTITY"), read_string("CALID"));
    EXPECT_EQ(read_string("CAL.PACKAGE_IDENTITY"), read_string("CALPKGID"));
    int correction_applied = 0;
    double correction_factor = 0.0;
    file.getVar("CAL.OBSERVATION_FLXSCALE_CORRECTION_APPLIED")
        .getVar(&correction_applied);
    file.getVar("CAL.APPLIED_OBSERVATION_FLXSCALE_CORRECTION")
        .getVar(&correction_factor);
    EXPECT_EQ(correction_applied, 1);
    EXPECT_DOUBLE_EQ(correction_factor, 3.0);
    EXPECT_EQ(read_string("CAL.OBSERVATION_FLXSCALE_CORRECTION_STATE"),
              "applied_once");
    EXPECT_EQ(
        read_string(
            "CAL.OBSERVATION_FLXSCALE_CORRECTION_SOURCE_IDENTITY"),
        timestream::CalibrationProduct::
            observation_correction_source_identity);
    EXPECT_EQ(
        read_string(
            "CAL.OBSERVATION_FLXSCALE_CORRECTION_RECIPIENT_IDENTITY"),
        "production-writer-raw-identity");
    file.close();

    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(citlali::config::RawTimeChunkConfig{});
    auto &observation = plan.begin_observation();
    observation.reduced_observation_identity = "000042";
    observation.canonical_calibration_product =
        engine.rtcproc.calibration.product;
    const auto provenance =
        citlali::pipeline::raw_timestream_provenance_node(plan);
    const auto lineage = provenance["calibration_lineage"]["value"];
    EXPECT_EQ(
        lineage["package_observation_identity"].as<std::string>(),
        "000042");
    const auto requested_preimage =
        lineage["response_basis"]["requested_config_preimage"];
    const auto expected_requested_preimage = YAML::Dump(
        citlali::pipeline::raw_timestream_request_node(plan.requested));
    EXPECT_EQ(
        requested_preimage["serialization"].as<std::string>(),
        "yaml-request-node-v1");
    EXPECT_EQ(
        requested_preimage["value"].as<std::string>(),
        expected_requested_preimage);
    EXPECT_EQ(
        requested_preimage["sha256"].as<std::string>(),
        citlali::utils::sha256(expected_requested_preimage));
    EXPECT_NE(
        lineage["response_basis"]["provenance"].as<std::string>().find(
            "requested_state_sha256=" +
            citlali::utils::sha256(expected_requested_preimage)),
        std::string::npos);
    const auto factors = lineage["factor_operator_state"];
    EXPECT_TRUE(
        factors["observation_flxscale_correction_applied"].as<bool>());
    EXPECT_DOUBLE_EQ(
        factors["applied_observation_flxscale_correction"].as<double>(),
        3.0);
    EXPECT_EQ(
        factors["observation_flxscale_correction_state"].as<std::string>(),
        "applied_once");
    const auto basis = factors["identity_basis"];
    EXPECT_EQ(
        basis["schema_version"].as<std::string>(),
        "sci-cal-001-admitted-factor-identity-basis-v1");
    const auto target = basis["target_unit_factor"];
    EXPECT_EQ(target["count"].as<int>(), 1);
    EXPECT_EQ(target["values"][0].as<std::string>(), "0x1p+0");
    EXPECT_EQ(
        target["sha256"].as<std::string>(),
        timestream::calibration_vector_identity(Eigen::VectorXd::Ones(1)));
    const auto flxscale = basis["detector_flxscale"];
    EXPECT_EQ(flxscale["values"][0].as<std::string>(), "0x1p+1");
    const auto extinction = basis["applied_sample_extinction_state"];
    EXPECT_TRUE(extinction["available"].as<bool>());
    EXPECT_FALSE(extinction["active"].as<bool>());
    EXPECT_EQ(
        extinction["sha256"].as<std::string>(),
        timestream::applied_sample_extinction_state_identity(
            engine.rtcproc.calibration.product
                .applied_sample_extinction_state));
    citlali::pipeline::complete_raw_timestream_observation(plan, 0, 0);
    EXPECT_TRUE(citlali::pipeline::raw_calibration_snapshot_matches(
        *plan.observation, plan.realized));
    std::filesystem::remove(path);
}

TEST(science_map_fits_products,
     active_extinction_identity_basis_roundtrips_exact_hexfloat_yaml) {
    EXPECT_EQ(
        timestream::calibration_hexfloat(
            std::numeric_limits<double>::denorm_min()),
        "0x1p-1074");
    Engine engine;
    configure_production_writer_engine(engine);
    admit_production_calibration_fixture(engine, true, true);
    citlali::pipeline::RawTimestreamExecutionPlan plan;
    plan.reset_from_request(citlali::config::RawTimeChunkConfig{});
    plan.begin_observation().canonical_calibration_product =
        engine.rtcproc.calibration.product;

    const auto serialized = YAML::Dump(
        citlali::pipeline::raw_timestream_provenance_node(plan));
    const auto reopened = YAML::Load(serialized);
    const auto factors = reopened["calibration_lineage"]["value"]
        ["factor_operator_state"];
    const auto basis = factors["identity_basis"];
    EXPECT_EQ(
        basis["target_unit_factor"]["sha256"].as<std::string>(),
        "e651fd05d98b2429fbd1355727fd4e9c7d417582aaf721a67b302ac5c14ab452");
    EXPECT_EQ(
        basis["minimum_extinction_correction"]["values"][0]
            .as<std::string>(),
        "0x1.1aec7b35a00d4p+0");
    EXPECT_EQ(
        basis["minimum_extinction_correction"]["sha256"].as<std::string>(),
        "639d6259d3b18f32f8464b52681e5d54736e81da46a88043ba8a95cd97b7b52d");
    const auto extinction = basis["applied_sample_extinction_state"];
    EXPECT_TRUE(extinction["available"].as<bool>());
    EXPECT_TRUE(extinction["active"].as<bool>());
    EXPECT_EQ(
        extinction["sample_elevation_rad"]["values"][0].as<std::string>(),
        "0x1p+0");
    EXPECT_EQ(
        extinction["los_tau_by_array"][0]["los_tau"]["values"][0]
            .as<std::string>(),
        "0x1.999999999999ap-4");
    EXPECT_EQ(
        extinction["los_tau_by_array"][1]["los_tau"]["values"][0]
            .as<std::string>(),
        "0x1.999999999999ap-3");
    EXPECT_EQ(
        extinction["los_tau_by_array"][2]["los_tau"]["values"][0]
            .as<std::string>(),
        "0x1.3333333333333p-2");
    EXPECT_EQ(
        extinction["sha256"].as<std::string>(),
        "b2043d14a309d6e124e287b0c30b3828d3da340d6cf6bd8d675d519fd2cb7ea4");
}

TEST(science_map_fits_products,
     actual_tod_link_publication_is_atomic_and_excludes_interruption) {
    const auto path = std::filesystem::path(testing::TempDir()) /
        "citlali-f008-actual-tod-link.nc";
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".calibration-link.tmp");
    write_netcdf_atomic(path.string(), [](netCDF::NcFile &file) {
        add_netcdf_var(file, "TOD.SKELETON", 1);
    });

    const auto publish = [&](bool fail_validation) {
        citlali::engine_detail::publish_linked_tod_atomic(
            path,
            [](netCDF::NcFile &staged) {
                add_netcdf_var(staged, "CAL.JOIN_AVAILABLE", true);
                add_netcdf_var(
                    staged, "CALID", std::string{"calibration-id"});
                add_netcdf_var(
                    staged, "CALPKGID", std::string{"package-id"});
            },
            [&](netCDF::NcFile &staged) {
                EXPECT_FALSE(staged.getVar("TOD.SKELETON").isNull());
                EXPECT_FALSE(staged.getVar("CALID").isNull());
                EXPECT_FALSE(staged.getVar("CALPKGID").isNull());
                if (fail_validation) {
                    throw DataIOError("interrupted before atomic replace");
                }
            });
    };

    ASSERT_NO_THROW(publish(false));
    EXPECT_FALSE(std::filesystem::exists(
        path.string() + ".calibration-link.tmp"));
    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::read);
        EXPECT_FALSE(file.getVar("CALID").isNull());
        EXPECT_FALSE(file.getVar("CALPKGID").isNull());
        file.close();
    }

    write_netcdf_atomic(path.string(), [](netCDF::NcFile &file) {
        add_netcdf_var(file, "TOD.SKELETON", 1);
    });
    EXPECT_THROW(publish(true), DataIOError);
    EXPECT_FALSE(std::filesystem::exists(path));
    EXPECT_FALSE(std::filesystem::exists(
        path.string() + ".calibration-link.tmp"));

    write_netcdf_atomic(path.string(), [](netCDF::NcFile &file) {
        add_netcdf_var(file, "TOD.SKELETON", 1);
    });
    EXPECT_THROW(
        citlali::engine_detail::publish_linked_tod_atomic(
            path,
            [](netCDF::NcFile &staged) {
                add_netcdf_var(staged, "CALID", std::string{"partial"});
                throw DataIOError("interrupted during staged write");
            },
            [](netCDF::NcFile &) {}),
        DataIOError);
    EXPECT_FALSE(std::filesystem::exists(path));
    EXPECT_FALSE(std::filesystem::exists(
        path.string() + ".calibration-link.tmp"));
}

TEST(science_map_fits_products,
     inactive_calibration_publishes_true_unavailable_tod_join) {
    timestream::RTCProc rtcproc;
    struct CalibMetadataFixture {
        Eigen::VectorXi arrays = Eigen::VectorXi::Zero(1);
    } calib;
    std::map<int, std::string> array_names{{0, "a1100"}};
    std::map<std::string, Eigen::VectorXd> telescope_data;
    const auto path = std::filesystem::path(testing::TempDir()) /
        "citlali-f008-inactive-calibration.nc";
    std::filesystem::remove(path);
    write_netcdf_atomic(path.string(), [&](netCDF::NcFile &file) {
        citlali::pipeline::add_tod_mean_tau_vars(
            file, false, rtcproc, telescope_data, 0.0,
            calib, array_names);
    });

    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    int join_available = 1;
    file.getVar("CAL.JOIN_AVAILABLE").getVar(&join_available);
    EXPECT_EQ(join_available, 0);
    EXPECT_TRUE(file.getVar("CALID").isNull());
    EXPECT_TRUE(file.getVar("CALPKGID").isNull());
    file.close();
    std::filesystem::remove(path);
}

std::shared_ptr<ScienceMapBufferFixture> make_production_science_map_buffer(
    const Engine &engine, bool coadd, Eigen::Index rows, Eigen::Index cols,
    const std::array<double, 2> &reference_pixel) {
    auto map = std::make_shared<ScienceMapBufferFixture>(
        coadd ? "cmb" : "omb");
    map->n_rows = rows;
    map->n_cols = cols;
    map->n_noise = 2;
    map->pixel_size_rad = 2.0 * ASEC_TO_RAD;
    map->sig_unit = "mJy/beam";
    map->map_grouping = "array";
    map->cov_cut = 1.0;
    map->science_products.allocate(1, rows, cols, coadd, true, true);

    auto &products = map->science_products;
    auto &realized = products.realized[0];
    mapmaking::ScienceMapBundleIdentity identity;
    identity.grouping = map->map_grouping;
    identity.signal_unit = map->sig_unit;
    identity.estimator_identity =
        mapmaking::science_map_coadd_estimator_version;
    identity.response_identity =
        citlali::pipeline::science_map_response_identity(
            engine.rtcproc.kernel, false);
    identity.required_companions = {
        "noise_realization_0_I", "noise_realization_1_I"};
    identity.rows = rows;
    identity.cols = cols;
    identity.wcs.coordinate_frame = "radec";
    identity.wcs.projection = "TAN";
    identity.wcs.axis_types = {"RA---TAN", "DEC--TAN"};
    identity.wcs.axis_units = {"deg", "deg"};
    identity.wcs.pixel_scale = {
        -0.00055555555555555556, 0.00055555555555555556};
    identity.wcs.reference_world = {
        187.046325, 44.093558300000005};
    identity.wcs.reference_pixel = {
        reference_pixel[0], reference_pixel[1]};
    identity.wcs.source_epoch = 2000.0;
    identity.wcs.orientation_rad = 0.0;
    mapmaking::ScienceMapSlotIdentity slot;
    slot.ordered_slot = 0;
    slot.grouping = "array";
    slot.group_identity = "array:0";
    slot.array_identity = 0;
    slot.stokes_identity = 0;
    slot.frequency_hz = engine.toltec_io.array_freq_map.at(0);
    identity.ordered_slots.push_back(slot);
    products.bundle_identity = identity;
    products.identity_admitted = true;

    realized.normalization.support_algorithm =
        mapmaking::science_map_normalization_support_version;
    realized.normalization.coefficient_stage = coadd
        ? mapmaking::science_map_coadd_normalization_coefficient_stage
        : mapmaking::science_map_observation_normalization_coefficient_stage;
    realized.normalization.requested_cut = 0.1;
    realized.normalization.realized_cut = 0.1;
    realized.normalization.realized_threshold = 0.001453413509532904;
    realized.normalization.selected_positive_value = 0.01453413509532904;
    realized.normalization.positive_value_count =
        static_cast<std::size_t>(rows * cols - 1);
    realized.normalization.selected_zero_based_index =
        realized.normalization.positive_value_count / 2;
    realized.normalization.selected_index_available = true;
    realized.science_policy.support_algorithm =
        mapmaking::science_map_policy_support_version;
    realized.science_policy.coefficient_stage =
        realized.normalization.coefficient_stage;
    realized.science_policy.requested_cut = 1.0;
    realized.science_policy.realized_cut = 1.0;
    realized.science_policy.realized_threshold = 0.01453413509532904;
    realized.science_policy.selected_positive_value = 0.01453413509532904;
    realized.science_policy.positive_value_count =
        realized.normalization.positive_value_count;
    realized.science_policy.selected_zero_based_index =
        realized.normalization.selected_zero_based_index;
    realized.science_policy.selected_index_available = true;

    map->signal.assign(1, Eigen::MatrixXd::Zero(rows, cols));
    map->weight.assign(1, Eigen::MatrixXd::Zero(rows, cols));
    map->coverage.assign(1, Eigen::MatrixXd::Zero(rows, cols));
    map->weight_formal.assign(1, Eigen::MatrixXd::Zero(rows, cols));
    map->noise_variance.assign(1, Eigen::MatrixXd::Zero(rows, cols));
    map->sig2noise_pixel.assign(1, Eigen::MatrixXd::Zero(rows, cols));
    map->noise.emplace_back(rows, cols, map->n_noise);
    map->noise[0].setZero();
    for (Eigen::Index row = 0; row < rows; ++row) {
        for (Eigen::Index col = 0; col < cols; ++col) {
            const bool supported = row != 0 || col != 0;
            products.geometric_hits[0](row, col) = 1;
            products.contributing_hits[0](row, col) = supported ? 1 : 0;
            products.coadd_observation_count[0](row, col) =
                coadd && supported ? 1 : 0;
            products.upstream_eligible_exposure[0](row, col) = 1.0;
            products.retained_exposure[0](row, col) = supported ? 1.0 : 0.0;
            products.normalization_support[0](row, col) = supported ? 1 : 0;
            products.science_policy_support[0](row, col) = supported ? 1 : 0;
            products.science_valid[0](row, col) = supported ? 1 : 0;
            map->signal[0](row, col) =
                supported ? 10.0 + row + 0.01 * col : 0.0;
            map->weight[0](row, col) =
                supported ? realized.science_policy.realized_threshold : 0.0;
            map->coverage[0](row, col) =
                products.retained_exposure[0](row, col);
            map->weight_formal[0](row, col) =
                supported ? 2.0 * map->weight[0](row, col) : 0.0;
            map->noise_variance[0](row, col) = supported ? 4.0 : 0.0;
            map->sig2noise_pixel[0](row, col) =
                supported ? map->signal[0](row, col) / 2.0 : 0.0;
            for (Eigen::Index realization = 0;
                 realization < map->n_noise; ++realization) {
                map->noise[0](row, col, realization) = supported
                    ? 100.0 * (realization + 1) + 10.0 * row + col
                    : 0.0;
            }
        }
    }
    map->median_err = Eigen::VectorXd::Constant(1, 1.0);
    map->median_rms = Eigen::VectorXd::Constant(1, 2.0);
    map->wcs.ctype = {"RA---TAN", "DEC--TAN", "FREQ", "STOKES"};
    map->wcs.cunit = {"deg", "deg", "Hz", "1"};
    map->wcs.crval = {0.0F, 0.0F, 0.0F, 0.0F};
    map->wcs.cdelt = {0.0F, 0.0F, 1.0F, 1.0F};
    map->wcs.crpix = {0.0F, 0.0F, 0.0F, 0.0F};
    map->wcs.naxis = {
        static_cast<int>(cols), static_cast<int>(rows), 1, 1};
    mapmaking::science_map_finalize_realized_product_facts(*map, 0);
    return map;
}

std::shared_ptr<ScienceMapBufferFixture> make_production_beammap_noise_buffer(
    Eigen::Index map_count) {
    auto map = std::make_shared<ScienceMapBufferFixture>("omb");
    map->n_rows = 2;
    map->n_cols = 3;
    map->n_noise = 2;
    map->pixel_size_rad = 2.0 * ASEC_TO_RAD;
    map->sig_unit = "mJy/beam";
    map->map_grouping = "detector";
    map->exposure_time = 1.0;
    map->science_products.allocate(
        map_count, map->n_rows, map->n_cols, false, false, false);
    map->signal.assign(
        static_cast<std::size_t>(map_count),
        Eigen::MatrixXd::Ones(map->n_rows, map->n_cols));
    map->weight.assign(
        static_cast<std::size_t>(map_count),
        Eigen::MatrixXd::Constant(map->n_rows, map->n_cols, 2.0));
    map->weight_formal.assign(
        static_cast<std::size_t>(map_count),
        Eigen::MatrixXd::Constant(map->n_rows, map->n_cols, 3.0));
    map->noise_variance.assign(
        static_cast<std::size_t>(map_count),
        Eigen::MatrixXd::Constant(map->n_rows, map->n_cols, 4.0));
    map->sig2noise_pixel.assign(
        static_cast<std::size_t>(map_count),
        Eigen::MatrixXd::Constant(map->n_rows, map->n_cols, 0.5));
    for (Eigen::Index map_index = 0; map_index < map_count; ++map_index) {
        map->noise.emplace_back(map->n_rows, map->n_cols, map->n_noise);
        map->noise.back().setConstant(
            static_cast<double>(map_index + 1));
    }
    map->median_err = Eigen::VectorXd::Ones(map_count);
    map->median_rms = Eigen::VectorXd::Constant(map_count, 2.0);
    map->noise_stack_scatter_valid = Eigen::VectorXi::Ones(map_count);
    map->noise_weight_scale_valid = Eigen::VectorXi::Ones(map_count);
    map->noise_weight_median_ratio = Eigen::VectorXd::Ones(map_count);
    map->noise_weight_scale = Eigen::VectorXd::Ones(map_count);
    map->noise_valid_pixels = Eigen::VectorXd::Constant(
        map_count, static_cast<double>(map->n_rows * map->n_cols));
    map->wcs.ctype = {"AZ---TAN", "EL---TAN", "FREQ", "STOKES"};
    map->wcs.cunit = {"deg", "deg", "Hz", "1"};
    map->wcs.crval = {0.0F, 0.0F, 0.0F, 0.0F};
    map->wcs.cdelt = {-0.001F, 0.001F, 1.0F, 1.0F};
    map->wcs.crpix = {1.0F, 1.0F, 0.0F, 0.0F};
    map->wcs.naxis = {
        static_cast<int>(map->n_cols), static_cast<int>(map->n_rows), 1, 1};
    return map;
}

void configure_production_beammap_writer(
    Beammap &beammap, const std::vector<int> &array_ids,
    const std::vector<int> &flags, Eigen::Index array_count) {
    configure_production_writer_engine(beammap);
    beammap.typed_config.runtime.reduction_type =
        citlali::config::ReductionType::beammap;
    beammap.typed_config.mapmaking.grouping =
        citlali::config::MapGrouping::detector;
    beammap.typed_config.coadd.enabled = false;
    beammap.typed_config.beammap.split_fits_by_flag.enabled = true;
    beammap.typed_config.beammap.split_fits_by_flag.flag_values = {0};
    beammap.noise_plan.reset_from_request(
        beammap.typed_config.noise, true);

    const auto map_count = static_cast<Eigen::Index>(array_ids.size());
    beammap.map_indices.n_maps = map_count;
    beammap.map_indices.maps_to_arrays.resize(map_count);
    beammap.map_indices.arrays_to_maps.resize(map_count);
    beammap.map_indices.maps_to_stokes.resize(map_count);
    beammap.map_indices.maps_to_stokes.setZero();
    for (Eigen::Index map_index = 0; map_index < map_count; ++map_index) {
        beammap.map_indices.maps_to_arrays(map_index) =
            array_ids[static_cast<std::size_t>(map_index)];
        beammap.map_indices.arrays_to_maps(map_index) =
            array_ids[static_cast<std::size_t>(map_index)];
    }

    beammap.calib.n_dets = map_count;
    beammap.calib.n_arrays = array_count;
    beammap.calib.arrays.resize(array_count);
    for (Eigen::Index array_index = 0; array_index < array_count;
         ++array_index) {
        beammap.calib.arrays(array_index) = array_index;
    }
    beammap.calib.run_hwpr = false;
    beammap.calib.apt_header_keys.clear();
    beammap.calib.apt_header_units.clear();
    beammap.calib.apt["flag"].resize(map_count);
    beammap.flag2 = Eigen::Matrix<uint16_t, Eigen::Dynamic, 1>::Zero(
        map_count);
    for (Eigen::Index map_index = 0; map_index < map_count; ++map_index) {
        beammap.calib.apt["flag"](map_index) =
            flags[static_cast<std::size_t>(map_index)];
    }
    for (Eigen::Index array_index = 0; array_index < array_count;
         ++array_index) {
        beammap.calib.array_fwhms[array_index] = {
            6.0 * ASEC_TO_RAD, 5.0 * ASEC_TO_RAD};
        beammap.calib.array_pas[array_index] = 0.0;
        beammap.calib.array_beam_areas[array_index] = 1.0;
    }
    beammap.telescope.fsmp = 1.0;
    beammap.telescope.source_name = "beammap-fixture";
    beammap.telescope.project_id = "SCI-NOI-002";
    beammap.telescope.obs_goal = "beammap";
}

struct FitsSpatialWcs {
    std::array<double, 2> cdelt{};
    std::array<double, 2> crpix{};
    std::array<double, 2> crval{};
    std::array<std::string, 2> ctype{};
    std::array<std::string, 2> cunit{};
    long cols = 0;
    long rows = 0;
};

double read_required_fits_double(fitsfile *file, const char *key) {
    double value = 0.0;
    int status = 0;
    if (fits_read_key(file, TDOUBLE, key, &value, nullptr, &status) != 0) {
        throw std::runtime_error(std::string{"missing FITS double key "} + key);
    }
    return value;
}

std::string read_required_fits_string(fitsfile *file, const char *key) {
    char value[FLEN_VALUE] = {};
    int status = 0;
    if (fits_read_key(file, TSTRING, key, value, nullptr, &status) != 0) {
        throw std::runtime_error(std::string{"missing FITS string key "} + key);
    }
    return value;
}

std::string read_required_fits_long_string(fitsfile *file,
                                           const char *key) {
    char *value = nullptr;
    int status = 0;
    if (fits_read_key_longstr(file, key, &value, nullptr, &status) != 0) {
        throw std::runtime_error(
            std::string{"missing FITS long string key "} + key);
    }
    const std::string result = value == nullptr
        ? std::string{} : std::string{value};
    if (value != nullptr) {
        int free_status = 0;
        fits_free_memory(value, &free_status);
        if (free_status != 0) {
            throw std::runtime_error(
                std::string{"cannot free FITS long string key "} + key);
        }
    }
    return result;
}

void move_to_required_image(fitsfile *file, const std::string &name) {
    int status = 0;
    if (fits_movnam_hdu(file, IMAGE_HDU, const_cast<char *>(name.c_str()), 0,
                       &status) != 0) {
        throw std::runtime_error("missing FITS image " + name);
    }
}

FitsSpatialWcs read_spatial_wcs(fitsfile *file, const std::string &name) {
    move_to_required_image(file, name);
    FitsSpatialWcs wcs;
    for (std::size_t axis = 0; axis < 2; ++axis) {
        const auto suffix = std::to_string(axis + 1);
        wcs.cdelt[axis] =
            read_required_fits_double(file, ("CDELT" + suffix).c_str());
        wcs.crpix[axis] =
            read_required_fits_double(file, ("CRPIX" + suffix).c_str());
        wcs.crval[axis] =
            read_required_fits_double(file, ("CRVAL" + suffix).c_str());
        wcs.ctype[axis] =
            read_required_fits_string(file, ("CTYPE" + suffix).c_str());
        wcs.cunit[axis] =
            read_required_fits_string(file, ("CUNIT" + suffix).c_str());
    }
    long axes[4] = {};
    int status = 0;
    if (fits_get_img_size(file, 4, axes, &status) != 0) {
        throw std::runtime_error("cannot read FITS image shape");
    }
    wcs.cols = axes[0];
    wcs.rows = axes[1];
    return wcs;
}

std::array<double, 2> inverse_tan_world(const FitsSpatialWcs &wcs,
                                        long row, long col) {
    const double deg_to_rad = M_PI / 180.0;
    const double xi =
        ((static_cast<double>(col) + 1.0) - wcs.crpix[0]) *
        wcs.cdelt[0] * deg_to_rad;
    const double eta =
        ((static_cast<double>(row) + 1.0) - wcs.crpix[1]) *
        wcs.cdelt[1] * deg_to_rad;
    const double ra0 = wcs.crval[0] * deg_to_rad;
    const double dec0 = wcs.crval[1] * deg_to_rad;
    const double denominator = std::cos(dec0) - eta * std::sin(dec0);
    const double ra = ra0 + std::atan2(xi, denominator);
    const double dec = std::atan2(
        std::sin(dec0) + eta * std::cos(dec0),
        std::hypot(denominator, xi));
    return {ra, dec};
}

double sky_separation_arcsec(const std::array<double, 2> &lhs,
                             const std::array<double, 2> &rhs) {
    const double half_delta_ra = (lhs[0] - rhs[0]) / 2.0;
    const double half_delta_dec = (lhs[1] - rhs[1]) / 2.0;
    const double haversine =
        std::sin(half_delta_dec) * std::sin(half_delta_dec) +
        std::cos(lhs[1]) * std::cos(rhs[1]) *
            std::sin(half_delta_ra) * std::sin(half_delta_ra);
    return 2.0 * std::asin(std::sqrt(std::clamp(haversine, 0.0, 1.0))) *
        (180.0 / M_PI) * 3600.0;
}

double maximum_wcs_separation_arcsec(
    const mapmaking::ScienceMapWcsIdentity &typed,
    const FitsSpatialWcs &physical) {
    FitsSpatialWcs typed_wcs;
    typed_wcs.cdelt = {typed.pixel_scale[0], typed.pixel_scale[1]};
    typed_wcs.crpix = {
        typed.reference_pixel[0] + 1.0,
        typed.reference_pixel[1] + 1.0};
    typed_wcs.crval = {typed.reference_world[0], typed.reference_world[1]};
    double maximum = 0.0;
    for (long row = 0; row < physical.rows; ++row) {
        for (long col = 0; col < physical.cols; ++col) {
            maximum = std::max(
                maximum,
                sky_separation_arcsec(
                    inverse_tan_world(typed_wcs, row, col),
                    inverse_tan_world(physical, row, col)));
        }
    }
    return maximum;
}

TEST(science_map_fits_products, preserves_native_fits_scalar_types) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    const std::string base =
        "/private/tmp/citlali-science-map-fits-types-" +
        std::to_string(nonce);
    FitsFileCleanup cleanup{base + ".fits"};

    using FitsOutput = fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>;
    FitsOutput output{base};
    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic> counts(1, 3);
    counts << -((std::int64_t{1} << 54) + 7), 17,
        (std::int64_t{1} << 54) + 3;
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic> mask(1, 3);
    mask << 0, 1, 255;
    Eigen::MatrixXd values(1, 3);
    values << -2.5, 0.0, 7.25;
    output.add_hdu("counts", counts);
    output.add_hdu("mask", mask);
    output.add_hdu("values", values);
    output.publish_atomically();

    fitsfile *file = nullptr;
    int status = 0;
    ASSERT_EQ(fits_open_file(&file, cleanup.path.c_str(), READONLY, &status), 0);

    auto move_to_image = [&](const char *name, int expected_bitpix) {
        ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                                  const_cast<char *>(name), 0, &status),
                  0);
        int bitpix = 0;
        ASSERT_EQ(fits_get_img_type(file, &bitpix, &status), 0);
        EXPECT_EQ(bitpix, expected_bitpix);
    };

    move_to_image("counts", LONGLONG_IMG);
    long long count_values[3] = {};
    int any_null = 0;
    ASSERT_EQ(fits_read_img(file, TLONGLONG, 1, 3, nullptr, count_values,
                            &any_null, &status),
              0);
    EXPECT_EQ(count_values[0], static_cast<long long>(counts(0, 2)));
    EXPECT_EQ(count_values[1], static_cast<long long>(counts(0, 1)));
    EXPECT_EQ(count_values[2], static_cast<long long>(counts(0, 0)));

    move_to_image("mask", BYTE_IMG);
    unsigned char mask_values[3] = {};
    ASSERT_EQ(fits_read_img(file, TBYTE, 1, 3, nullptr, mask_values,
                            &any_null, &status),
              0);
    EXPECT_EQ(mask_values[0], 255);
    EXPECT_EQ(mask_values[1], 1);
    EXPECT_EQ(mask_values[2], 0);

    move_to_image("values", DOUBLE_IMG);
    double double_values[3] = {};
    ASSERT_EQ(fits_read_img(file, TDOUBLE, 1, 3, nullptr, double_values,
                            &any_null, &status),
              0);
    EXPECT_DOUBLE_EQ(double_values[0], values(0, 2));
    EXPECT_DOUBLE_EQ(double_values[1], values(0, 1));
    EXPECT_DOUBLE_EQ(double_values[2], values(0, 0));

    EXPECT_EQ(fits_close_file(file, &status), 0);
}

TEST(science_map_fits_products,
     fits_publication_failure_matrix_preserves_existing_complete_final) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    const std::string base =
        "/private/tmp/citlali-f008-fits-publication-" +
        std::to_string(nonce);
    const auto final_path = std::filesystem::path(base + ".fits");
    const auto stage_path = std::filesystem::path(base + ".fits.tmp");
    const auto backup_path =
        std::filesystem::path(base + ".fits.replace-backup");
    FitsFileCleanup cleanup{final_path.string()};
    using FitsOutput =
        fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>;

    const auto configure = [](FitsOutput &output,
                              const std::string &calibration_identity,
                              const std::string &package_identity,
                              double value) {
        output.pfits->pHDU().addKey(
            "CALID", calibration_identity,
            "Canonical complete applied calibration identity");
        output.pfits->pHDU().addKey(
            "CALPKGID", package_identity,
            "Canonical calibration package identity");
        output.require_calibration_join(
            calibration_identity, package_identity);
        Eigen::MatrixXd image(1, 1);
        image(0, 0) = value;
        output.add_hdu("signal_I", image);
    };

    {
        FitsOutput accepted{base};
        configure(accepted, "accepted-calid", "accepted-pkgid", 1.0);
        accepted.publish_atomically();
    }
    const auto accepted_digest =
        citlali::utils::sha256_file(final_path);

    const std::array<FitsOutput::PublicationCheckpoint, 5> failures = {
        FitsOutput::PublicationCheckpoint::after_hdu_write,
        FitsOutput::PublicationCheckpoint::after_library_flush,
        FitsOutput::PublicationCheckpoint::after_close,
        FitsOutput::PublicationCheckpoint::before_reopen,
        FitsOutput::PublicationCheckpoint::after_reopen};
    for (const auto failure : failures) {
        FitsOutput replacement{base};
        configure(replacement, "replacement-calid", "replacement-pkgid",
                  2.0);
        EXPECT_THROW(
            replacement.publish_atomically(
                [&](FitsOutput::PublicationCheckpoint checkpoint) {
                    if (checkpoint == failure) {
                        throw std::runtime_error(
                            "injected late write/close/reopen interruption");
                    }
                }),
            std::exception);
        EXPECT_EQ(citlali::utils::sha256_file(final_path), accepted_digest);
        EXPECT_FALSE(std::filesystem::exists(stage_path));
        EXPECT_FALSE(std::filesystem::exists(backup_path));
    }

    {
        FitsOutput conflicting_join{base};
        conflicting_join.pfits->pHDU().addKey(
            "CALID", std::string{"wrong-calid"}, "wrong join");
        conflicting_join.pfits->pHDU().addKey(
            "CALPKGID", std::string{"wrong-pkgid"}, "wrong join");
        conflicting_join.require_calibration_join(
            "required-calid", "required-pkgid");
        Eigen::MatrixXd image = Eigen::MatrixXd::Ones(1, 1);
        conflicting_join.add_hdu("signal_I", image);
        EXPECT_THROW(conflicting_join.publish_atomically(), std::exception);
        EXPECT_EQ(citlali::utils::sha256_file(final_path), accepted_digest);
        EXPECT_FALSE(std::filesystem::exists(stage_path));
    }

    {
        FitsOutput structurally_incomplete{base};
        configure(structurally_incomplete, "replacement-calid",
                  "replacement-pkgid", 3.0);
        EXPECT_THROW(
            structurally_incomplete.publish_atomically(
                [&](FitsOutput::PublicationCheckpoint checkpoint) {
                    if (checkpoint ==
                        FitsOutput::PublicationCheckpoint::before_reopen) {
                        std::ofstream truncated(
                            structurally_incomplete.staged_path(),
                            std::ios::out | std::ios::trunc);
                        truncated << "not-a-complete-fits-file";
                    }
                }),
            std::exception);
        EXPECT_EQ(citlali::utils::sha256_file(final_path), accepted_digest);
        EXPECT_FALSE(std::filesystem::exists(stage_path));
    }

    {
        FitsOutput replacement{base};
        configure(replacement, "replacement-calid", "replacement-pkgid",
                  4.0);
        ASSERT_NO_THROW(replacement.publish_atomically());
    }
    EXPECT_NE(citlali::utils::sha256_file(final_path), accepted_digest);
    EXPECT_FALSE(std::filesystem::exists(stage_path));
    EXPECT_FALSE(std::filesystem::exists(backup_path));
    fitsfile *file = nullptr;
    int status = 0;
    ASSERT_EQ(fits_open_file(
                  &file, final_path.c_str(), READONLY, &status),
              0);
    EXPECT_EQ(read_required_fits_long_string(file, "CALID"),
              "replacement-calid");
    EXPECT_EQ(read_required_fits_long_string(file, "CALPKGID"),
              "replacement-pkgid");
    EXPECT_EQ(fits_close_file(file, &status), 0);
}

using F008FitsOutput =
    fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>;

enum class F008PublicationFailure {
    write,
    synchronize,
    close,
    reopen,
    validate,
    replace
};

const char *f008_failure_name(F008PublicationFailure failure) {
    switch (failure) {
        case F008PublicationFailure::write:
            return "write";
        case F008PublicationFailure::synchronize:
            return "synchronize";
        case F008PublicationFailure::close:
            return "close";
        case F008PublicationFailure::reopen:
            return "reopen";
        case F008PublicationFailure::validate:
            return "validate";
        case F008PublicationFailure::replace:
            return "replace";
    }
    return "unknown";
}

void configure_f008_output(F008FitsOutput &output,
                           const std::string &calibration_identity,
                           const std::string &package_identity,
                           double value) {
    output.pfits->pHDU().addKey(
        "CALID", calibration_identity,
        "Canonical complete applied calibration identity");
    output.pfits->pHDU().addKey(
        "CALPKGID", package_identity,
        "Canonical calibration package identity");
    output.require_calibration_join(
        calibration_identity, package_identity);
    Eigen::MatrixXd image(1, 1);
    image(0, 0) = value;
    output.add_hdu("signal_I", image);
}

void verify_f008_final(const std::filesystem::path &path,
                       const std::string &calibration_identity,
                       const std::string &package_identity) {
    fitsfile *file = nullptr;
    int status = 0;
    ASSERT_EQ(fits_open_file(&file, path.c_str(), READONLY, &status), 0);
    EXPECT_EQ(read_required_fits_long_string(file, "CALID"),
              calibration_identity);
    EXPECT_EQ(read_required_fits_long_string(file, "CALPKGID"),
              package_identity);
    ASSERT_EQ(fits_movnam_hdu(
                  file, IMAGE_HDU, const_cast<char *>("signal_I"), 0,
                  &status),
              0);
    long dimensions[4] = {};
    int dimension_count = 0;
    ASSERT_EQ(fits_get_img_dim(file, &dimension_count, &status), 0);
    ASSERT_EQ(fits_get_img_size(file, 4, dimensions, &status), 0);
    ASSERT_EQ(dimension_count, 4);
    for (const auto dimension : dimensions) {
        EXPECT_EQ(dimension, 1);
    }
    EXPECT_EQ(fits_close_file(file, &status), 0);
}

void invalidate_f008_calibration_join(
    const std::filesystem::path &staged_path) {
    fitsfile *file = nullptr;
    int status = 0;
    if (fits_open_file(&file, staged_path.c_str(), READWRITE, &status) != 0) {
        throw std::runtime_error("unable to reopen staged FITS for tamper");
    }
    char invalid_calid[] = "invalid-calid";
    if (fits_update_key(file, TSTRING, const_cast<char *>("CALID"),
                        invalid_calid, nullptr, &status) != 0) {
        fits_close_file(file, &status);
        throw std::runtime_error("unable to tamper staged FITS CALID");
    }
    if (fits_close_file(file, &status) != 0) {
        throw std::runtime_error("unable to close tampered staged FITS");
    }
}

void seed_f008_final(const std::string &base, double value) {
    F008FitsOutput accepted{base};
    configure_f008_output(
        accepted, "accepted-calid", "accepted-pkgid", value);
    accepted.publish_atomically();
}

template <mapmaking::MapType MapType>
void exercise_f008_owner_lifecycle_matrix(
    const std::filesystem::path &root, bool science_wiener_owner) {
    const std::string route = [] {
        if constexpr (MapType == mapmaking::RawObs) {
            return "raw_observation";
        }
        else if constexpr (MapType == mapmaking::FilteredObs) {
            return "filtered_observation";
        }
        else if constexpr (MapType == mapmaking::RawCoadd) {
            return "raw_coadd";
        }
        else {
            static_assert(MapType == mapmaking::FilteredCoadd);
            return "filtered_coadd";
        }
    }();
    const std::array<F008PublicationFailure, 6> failures = {
        F008PublicationFailure::write,
        F008PublicationFailure::synchronize,
        F008PublicationFailure::close,
        F008PublicationFailure::reopen,
        F008PublicationFailure::validate,
        F008PublicationFailure::replace};
    const std::array<std::string, 2> artifact_families = {"data", "noise"};

    for (const auto failure : failures) {
        for (const auto &target_family : artifact_families) {
            SCOPED_TRACE(route + "/" + target_family + "/" +
                         f008_failure_name(failure));
            const auto case_root =
                root / (route + "-" + target_family + "-" +
                        f008_failure_name(failure));
            std::filesystem::create_directories(case_root);
            const auto data_base = (case_root / "data").string();
            const auto noise_base = (case_root / "noise").string();
            const auto target_base =
                target_family == "data" ? data_base : noise_base;
            const auto data_final =
                std::filesystem::path(data_base + ".fits");
            const auto noise_final =
                std::filesystem::path(noise_base + ".fits");
            const auto target_final =
                std::filesystem::path(target_base + ".fits");
            const auto target_backup = std::filesystem::path(
                target_base + ".fits.replace-backup");

            seed_f008_final(data_base, 1.0);
            seed_f008_final(noise_base, 2.0);
            const auto accepted_target_digest =
                citlali::utils::sha256_file(target_final);

            std::vector<F008FitsOutput> data_outputs;
            std::vector<F008FitsOutput> noise_outputs;
            data_outputs.emplace_back(data_base);
            noise_outputs.emplace_back(noise_base);
            configure_f008_output(
                data_outputs[0], "replacement-calid", "replacement-pkgid",
                3.0);
            configure_f008_output(
                noise_outputs[0], "replacement-calid", "replacement-pkgid",
                4.0);
            const auto data_stage = data_outputs[0].staged_path();
            const auto noise_stage = noise_outputs[0].staged_path();

            if (failure == F008PublicationFailure::replace) {
                std::filesystem::create_directories(target_backup);
                std::ofstream marker(target_backup / "nonempty");
                marker << "force replacement preservation failure";
            }

            bool publication_recorded = false;
            const auto publish_with_failure = [&](F008FitsOutput &output) {
                if (output.filepath != target_base) {
                    output.publish_atomically();
                    return;
                }
                output.publish_atomically(
                    [&](F008FitsOutput::PublicationCheckpoint checkpoint) {
                        if (failure == F008PublicationFailure::write &&
                            checkpoint == F008FitsOutput::
                                PublicationCheckpoint::after_hdu_write) {
                            throw std::runtime_error(
                                "injected required FITS write failure");
                        }
                        if (failure == F008PublicationFailure::synchronize &&
                            checkpoint == F008FitsOutput::
                                PublicationCheckpoint::after_hdu_write) {
                            std::error_code ignored;
                            std::filesystem::remove(
                                output.staged_path(), ignored);
                        }
                        if (failure == F008PublicationFailure::close &&
                            checkpoint == F008FitsOutput::
                                PublicationCheckpoint::after_close) {
                            throw std::runtime_error(
                                "injected required FITS close failure");
                        }
                        if (failure == F008PublicationFailure::reopen &&
                            checkpoint == F008FitsOutput::
                                PublicationCheckpoint::before_reopen) {
                            std::error_code ignored;
                            std::filesystem::remove(
                                output.staged_path(), ignored);
                        }
                        if (failure == F008PublicationFailure::validate &&
                            checkpoint == F008FitsOutput::
                                PublicationCheckpoint::before_reopen) {
                            invalidate_f008_calibration_join(
                                output.staged_path());
                        }
                    });
            };
            const auto invoke_owner = [&](auto &&publisher) {
                if (science_wiener_owner) {
                    citlali::pipeline::finalize_map_filter_fits_outputs(
                        &data_outputs, &noise_outputs,
                        "filtered science maps", science_map_test_logger(),
                        std::forward<decltype(publisher)>(publisher));
                    publication_recorded = true;
                }
                else {
                    citlali::pipeline::finalize_pointing_map_fits_outputs(
                        &data_outputs, &noise_outputs,
                        std::forward<decltype(publisher)>(publisher),
                        [&] { publication_recorded = true; });
                }
            };

            EXPECT_THROW(invoke_owner(publish_with_failure), std::exception);
            EXPECT_FALSE(publication_recorded);
            EXPECT_FALSE(data_outputs.empty());
            EXPECT_FALSE(noise_outputs.empty());
            EXPECT_EQ(citlali::utils::sha256_file(target_final),
                      accepted_target_digest);
            data_outputs.clear();
            noise_outputs.clear();
            EXPECT_FALSE(std::filesystem::exists(data_stage));
            EXPECT_FALSE(std::filesystem::exists(noise_stage));
            std::error_code ignored;
            std::filesystem::remove_all(target_backup, ignored);

            data_outputs.emplace_back(data_base);
            noise_outputs.emplace_back(noise_base);
            configure_f008_output(
                data_outputs[0], "retry-calid", "retry-pkgid", 5.0);
            configure_f008_output(
                noise_outputs[0], "retry-calid", "retry-pkgid", 6.0);
            const auto retry_data_stage = data_outputs[0].staged_path();
            const auto retry_noise_stage = noise_outputs[0].staged_path();
            publication_recorded = false;
            invoke_owner([](F008FitsOutput &output) {
                output.publish_atomically();
            });

            EXPECT_TRUE(publication_recorded);
            EXPECT_TRUE(data_outputs.empty());
            EXPECT_TRUE(noise_outputs.empty());
            EXPECT_FALSE(std::filesystem::exists(retry_data_stage));
            EXPECT_FALSE(std::filesystem::exists(retry_noise_stage));
            EXPECT_NE(citlali::utils::sha256_file(target_final),
                      accepted_target_digest);
            verify_f008_final(
                data_final, "retry-calid", "retry-pkgid");
            verify_f008_final(
                noise_final, "retry-calid", "retry-pkgid");
        }
    }
}

TEST(science_map_fits_products,
     pointing_owner_lifecycle_covers_all_map_data_noise_failure_routes) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    FitsDirectoryCleanup cleanup{
        std::filesystem::path{"/private/tmp"} /
        ("citlali-f008-pointing-owner-matrix-" +
         std::to_string(nonce))};
    std::filesystem::create_directories(cleanup.path);

    exercise_f008_owner_lifecycle_matrix<mapmaking::RawObs>(
        cleanup.path, false);
    exercise_f008_owner_lifecycle_matrix<mapmaking::FilteredObs>(
        cleanup.path, false);
    exercise_f008_owner_lifecycle_matrix<mapmaking::RawCoadd>(
        cleanup.path, false);
    exercise_f008_owner_lifecycle_matrix<mapmaking::FilteredCoadd>(
        cleanup.path, false);
}

TEST(science_map_fits_products,
     science_wiener_owner_lifecycle_covers_obs_coadd_data_noise_failures) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    FitsDirectoryCleanup cleanup{
        std::filesystem::path{"/private/tmp"} /
        ("citlali-f008-science-wiener-matrix-" +
         std::to_string(nonce))};
    std::filesystem::create_directories(cleanup.path);

    exercise_f008_owner_lifecycle_matrix<mapmaking::FilteredObs>(
        cleanup.path, true);
    exercise_f008_owner_lifecycle_matrix<mapmaking::FilteredCoadd>(
        cleanup.path, true);
}

TEST(science_map_fits_products,
     round_trips_complete_f010_bundle_metadata_aliases_and_wcs) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    const std::string base =
        "/private/tmp/citlali-science-map-f010-bundle-" +
        std::to_string(nonce);
    FitsFileCleanup cleanup{base + ".fits"};
    using FitsOutput = fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>;
    FitsOutput output{base};
    auto map = make_science_map_buffer(true);

    citlali::pipeline::add_science_map_product_image_hdus(
        output, map, 0, "", "I", map->wcs, 2000.0,
        science_map_test_logger());
    output.publish_atomically();

    fitsfile *file = nullptr;
    int status = 0;
    ASSERT_EQ(fits_open_file(&file, cleanup.path.c_str(), READONLY, &status), 0);
    const std::vector<std::pair<std::string, int>> products = {
        {"geometric_hits_I", LONGLONG_IMG},
        {"contributing_hits_I", LONGLONG_IMG},
        {"coadd_observation_count_I", LONGLONG_IMG},
        {"upstream_eligible_exposure_I", DOUBLE_IMG},
        {"retained_exposure_I", DOUBLE_IMG},
        {"normalization_support_I", BYTE_IMG},
        {"science_policy_support_I", BYTE_IMG},
        {"science_valid_I", BYTE_IMG},
        {"coverage_I", DOUBLE_IMG},
        {"coverage_bool_I", BYTE_IMG},
    };
    auto read_string_key = [&](const char *key) {
        char value[FLEN_VALUE] = {};
        char comment[FLEN_COMMENT] = {};
        EXPECT_EQ(fits_read_key(file, TSTRING, key, value, comment, &status),
                  0);
        return std::string(value);
    };
    for (const auto &[name, expected_bitpix] : products) {
        ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                                  const_cast<char *>(name.c_str()), 0,
                                  &status),
                  0)
            << name;
        int bitpix = 0;
        ASSERT_EQ(fits_get_img_type(file, &bitpix, &status), 0) << name;
        EXPECT_EQ(bitpix, expected_bitpix) << name;
        EXPECT_EQ(read_string_key("CTYPE1"), "RA---TAN") << name;
        EXPECT_EQ(read_string_key("CUNIT1"), "deg") << name;
        double value = 0.0;
        ASSERT_EQ(fits_read_key(file, TDOUBLE, "CRVAL1", &value, nullptr,
                                &status),
                  0)
            << name;
        EXPECT_DOUBLE_EQ(value, 123.25) << name;
        ASSERT_EQ(fits_read_key(file, TDOUBLE, "CRPIX1", &value, nullptr,
                                &status),
                  0)
            << name;
        EXPECT_DOUBLE_EQ(value, 1.5) << name;
        ASSERT_EQ(fits_read_key(file, TDOUBLE, "EQUINOX", &value, nullptr,
                                &status),
                  0)
            << name;
        EXPECT_DOUBLE_EQ(value, 2000.0) << name;
    }

    ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                              const_cast<char *>("coverage_I"), 0, &status),
              0);
    EXPECT_EQ(read_string_key("ALIASOF"), "retained_exposure_I");
    EXPECT_EQ(read_string_key("BUNIT"), "detector s");
    double coverage[4] = {};
    int any_null = 0;
    ASSERT_EQ(fits_read_img(file, TDOUBLE, 1, 4, nullptr, coverage, &any_null,
                            &status),
              0);
    ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                              const_cast<char *>("retained_exposure_I"), 0,
                              &status),
              0);
    double retained[4] = {};
    ASSERT_EQ(fits_read_img(file, TDOUBLE, 1, 4, nullptr, retained, &any_null,
                            &status),
              0);
    for (std::size_t index = 0; index < 4; ++index) {
        EXPECT_EQ(std::memcmp(&coverage[index], &retained[index],
                              sizeof(double)),
                  0);
    }

    ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                              const_cast<char *>("coverage_bool_I"), 0,
                              &status),
              0);
    EXPECT_EQ(read_string_key("ALIASOF"), "science_policy_support_I");
    EXPECT_EQ(read_string_key("DEPRCATD"), "true");
    EXPECT_EQ(read_string_key("VALAUTH"), "false");
    unsigned char coverage_mask[4] = {};
    ASSERT_EQ(fits_read_img(file, TBYTE, 1, 4, nullptr, coverage_mask,
                            &any_null, &status),
              0);
    ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                              const_cast<char *>("science_policy_support_I"),
                              0, &status),
              0);
    unsigned char policy_mask[4] = {};
    ASSERT_EQ(fits_read_img(file, TBYTE, 1, 4, nullptr, policy_mask,
                            &any_null, &status),
              0);
    for (std::size_t index = 0; index < 4; ++index) {
        EXPECT_EQ(coverage_mask[index], policy_mask[index]);
    }

    ASSERT_EQ(fits_movnam_hdu(file, IMAGE_HDU,
                              const_cast<char *>("science_valid_I"), 0,
                              &status),
              0);
    EXPECT_EQ(read_string_key("VALAUTH"), "true");
    EXPECT_EQ(read_string_key("DATTYP"), "uint8");
    EXPECT_EQ(fits_close_file(file, &status), 0);
}

TEST(science_map_fits_products,
     filtered_fits_round_trips_complete_raw_parent_digest) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    const std::string base =
        "/private/tmp/citlali-science-map-filtered-parent-" +
        std::to_string(nonce);
    FitsFileCleanup cleanup{base + ".fits"};
    using FitsOutput = fitsIO<file_type_enum::write_fits, CCfits::ExtHDU *>;
    FitsOutput output{base};
    auto map = make_science_map_buffer(false);
    map->freeze_raw_science_parent();
    ASSERT_TRUE(map->raw_science_parent);
    const auto expected =
        map->raw_science_parent->realized[0].raw_parent_digest;
    ASSERT_GT(expected.size(), 68U);
    map->signal[0].setConstant(5.0);

    citlali::pipeline::add_science_map_product_image_hdus(
        output, map, 0, "", "I", map->wcs, 2000.0,
        science_map_test_logger(), true);
    output.publish_atomically();

    fitsfile *file = nullptr;
    int status = 0;
    ASSERT_EQ(fits_open_file(&file, cleanup.path.c_str(), READONLY, &status),
              0);
    ASSERT_EQ(fits_movnam_hdu(
                  file, IMAGE_HDU,
                  const_cast<char *>("science_valid_I"), 0, &status),
              0);
    char *value = nullptr;
    char comment[FLEN_COMMENT] = {};
    ASSERT_EQ(fits_read_key_longstr(file, "RAWPDGST", &value, comment,
                                    &status),
              0);
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(std::string(value), expected);
    int free_status = 0;
    fits_free_memory(value, &free_status);
    EXPECT_EQ(free_status, 0);
    EXPECT_EQ(fits_close_file(file, &status), 0);
}

TEST(science_map_fits_products,
     coadd_enabled_observation_inventory_keeps_required_noise_file) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    FitsDirectoryCleanup cleanup{
        std::filesystem::path{"/private/tmp"} /
        ("citlali-science-map-observation-inventory-" +
         std::to_string(nonce))};
    Engine engine;
    configure_production_writer_engine(engine);
    engine.output_paths.obsnum_dir_name = cleanup.path.string() + "/";
    engine.observation_identity.obsnum = "152390";
    std::filesystem::create_directories(cleanup.path / "raw");

    ASSERT_NO_THROW(engine.create_obs_map_files());
    ASSERT_EQ(engine.map_fits_outputs.obs.size(), 1U);
    ASSERT_EQ(engine.map_fits_outputs.obs_noise.size(), 1U);
    EXPECT_TRUE(engine.map_fits_outputs.filtered_obs.empty());
    EXPECT_TRUE(engine.map_fits_outputs.filtered_obs_noise.empty());
    EXPECT_NE(engine.map_fits_outputs.obs[0].filepath.find("152390"),
              std::string::npos);
    EXPECT_NE(engine.map_fits_outputs.obs_noise[0].filepath.find("152390"),
              std::string::npos);
    EXPECT_NE(engine.map_fits_outputs.obs_noise[0].filepath.find("_noise"),
              std::string::npos);
}

TEST(science_map_fits_products,
     successor_coadd_writer_finalizes_unscaled_and_scaled_only_packages) {
    for (const bool apply_empirical_weights : {false, true}) {
        SCOPED_TRACE(apply_empirical_weights);
        const auto nonce = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        FitsDirectoryCleanup cleanup{
            std::filesystem::path{"/private/tmp"} /
            ("citlali-noise-coadd-package-" +
             std::to_string(nonce) + "-" +
             std::to_string(apply_empirical_weights))};
        std::filesystem::create_directories(cleanup.path);

        Engine engine;
        configure_production_writer_engine(engine);
        engine.typed_config.noise.apply_empirical_weights =
            apply_empirical_weights;
        engine.noise_plan.reset_from_request(
            engine.typed_config.noise, true);
        citlali::pipeline::begin_noise_product_publication(
            cleanup.path, engine.noise_plan);

        decltype(engine.map_fits_outputs.obs) observation_data_files;
        decltype(engine.map_fits_outputs.obs_noise)
            observation_realization_files;
        const auto observation_data_base =
            (cleanup.path / "observation_map").string();
        const auto observation_realization_base =
            (cleanup.path / "observation_noise").string();
        observation_data_files.emplace_back(observation_data_base);
        observation_realization_files.emplace_back(
            observation_realization_base);
        auto observation = make_production_science_map_buffer(
            engine, false, 3, 4, {2.0, 1.5});
        auto *observation_data_ptr = &observation_data_files;
        auto *observation_realization_ptr =
            &observation_realization_files;
        ASSERT_NO_THROW(engine.write_maps(
            observation_data_ptr, observation_realization_ptr,
            observation, 0));
        const std::vector<std::filesystem::path> observation_data_paths{
            observation_data_base + ".fits"};
        const std::vector<std::filesystem::path>
            observation_realization_paths{
                observation_realization_base + ".fits"};
        for (auto &output : observation_data_files) {
            output.publish_atomically();
        }
        for (auto &output : observation_realization_files) {
            output.publish_atomically();
        }
        observation_data_files.clear();
        observation_realization_files.clear();
        ASSERT_NO_THROW(
            citlali::pipeline::record_noise_map_output_publication(
                engine.noise_plan, false, false, *observation,
                observation_data_paths, observation_realization_paths));
        ASSERT_EQ(
            *engine.noise_plan.realized.empirical_product_map_count, 1U);
        ASSERT_EQ(
            *engine.noise_plan.realized.realization_image_write_count, 2U);

        decltype(engine.map_fits_outputs.coadd) data_files;
        decltype(engine.map_fits_outputs.coadd_noise) realization_files;
        const auto data_base =
            (cleanup.path / "coadd_map").string();
        const auto realization_base =
            (cleanup.path / "coadd_noise").string();
        data_files.emplace_back(data_base);
        realization_files.emplace_back(realization_base);
        auto coadd = make_production_science_map_buffer(
            engine, true, 3, 4, {2.0, 1.5});
        auto *data_file_ptr = &data_files;
        auto *realization_file_ptr = &realization_files;
        ASSERT_NO_THROW(engine.write_maps(
            data_file_ptr, realization_file_ptr, coadd, 0));

        const std::vector<std::filesystem::path> data_paths{
            data_base + ".fits"};
        const std::vector<std::filesystem::path> realization_paths{
            realization_base + ".fits"};
        for (auto &output : data_files) {
            output.publish_atomically();
        }
        for (auto &output : realization_files) {
            output.publish_atomically();
        }
        data_files.clear();
        realization_files.clear();
        ASSERT_NO_THROW(
            citlali::pipeline::record_noise_map_output_publication(
                engine.noise_plan, true, false, *coadd,
                data_paths, realization_paths));
        EXPECT_EQ(
            *engine.noise_plan.realized.empirical_product_map_count, 1U);
        EXPECT_EQ(
            *engine.noise_plan.realized.realization_image_write_count, 4U);

        citlali::config::MapmakingConfig mapmaking_request;
        mapmaking_request.method = citlali::config::MapMethod::naive;
        citlali::pipeline::MapmakingExecutionPlan mapmaking;
        mapmaking.reset_from_request(
            mapmaking_request, citlali::config::ReductionType::science);
        mapmaking.begin_iteration();
        mapmaking.begin_observation(
            0, "152390", 1, 4.848136811e-6, 1);
        citlali::pipeline::complete_mapmaking_observation(mapmaking);
        mapmaking.begin_coadd(1, 1);
        citlali::pipeline::complete_mapmaking_coadd(mapmaking);
        citlali::pipeline::record_mapmaking_run_completed(mapmaking);
        ASSERT_NO_THROW(citlali::pipeline::record_noise_run_completed(
            engine.noise_plan, mapmaking, false));
        EXPECT_EQ(
            engine.noise_plan.expected.empirical_product_map_count, 1U);
        EXPECT_EQ(
            *engine.noise_plan.realized.empirical_product_map_count, 1U);
        EXPECT_EQ(
            *engine.noise_plan.realized.realization_image_write_count, 4U);

        ASSERT_NO_THROW(citlali::pipeline::write_noise_provenance_file(
            cleanup.path, engine.noise_plan));
        const auto package = YAML::LoadFile(
            citlali::pipeline::noise_provenance_path(cleanup.path).string());
        const auto members = package["package"]["member_files"];
        ASSERT_TRUE(members.IsSequence());
        EXPECT_EQ(
            members.size(), apply_empirical_weights ? 4U : 3U);

        const auto observation_join =
            citlali::pipeline::validate_noise_fits_joins(
                observation_data_paths.front());
        EXPECT_EQ(observation_join.empirical_map_product_count, 1U);
        const auto observation_realization_join =
            citlali::pipeline::validate_noise_fits_joins(
                observation_realization_paths.front());
        EXPECT_EQ(observation_realization_join.realization_image_count, 2U);

        const auto realization_join =
            citlali::pipeline::validate_noise_fits_joins(
                realization_paths.front());
        EXPECT_EQ(realization_join.realization_image_count, 2U);
        EXPECT_EQ(realization_join.empirical_map_product_count, 0U);
        if (apply_empirical_weights) {
            const auto scaled_join =
                citlali::pipeline::validate_noise_fits_joins(
                    data_paths.front());
            EXPECT_EQ(scaled_join.empirical_map_product_count, 0U);
            EXPECT_EQ(
                scaled_join.product_identities,
                std::vector<std::string>{
                    citlali::pipeline::
                        noise_scaled_coefficient_product_id});
        }
        else {
            EXPECT_THROW(
                citlali::pipeline::validate_noise_fits_joins(
                    data_paths.front()),
                std::runtime_error);
        }
    }
}

TEST(science_map_fits_products,
     split_beammap_writer_finalizes_logical_maps_and_excludes_empty_files) {
    struct SplitShape {
        std::vector<int> arrays;
        std::vector<int> flags;
        Eigen::Index array_count;
        std::size_t selected_array_count;
    };
    const std::array<SplitShape, 2> shapes{{
        {{0, 1}, {0, 0}, 2, 2},
        {{0, 0, 1}, {0, 0, 1}, 2, 1},
    }};

    for (std::size_t shape_index = 0; shape_index < shapes.size();
         ++shape_index) {
        SCOPED_TRACE(shape_index);
        const auto &shape = shapes[shape_index];
        const auto nonce = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        FitsDirectoryCleanup cleanup{
            std::filesystem::path{"/private/tmp"} /
            ("citlali-noise-split-beammap-package-" +
             std::to_string(nonce) + "-" +
             std::to_string(shape_index))};
        std::filesystem::create_directories(cleanup.path);

        Beammap beammap;
        configure_production_beammap_writer(
            beammap, shape.arrays, shape.flags, shape.array_count);
        if (shape_index == 0) {
            beammap.typed_config.timestream.raw_time_chunk
                .extinction_correction_enabled = true;
            beammap.rtcproc.run_extinction = true;
            beammap.rtcproc.calibration.select_reference_spectral_index(2.0);
            beammap.rtcproc.calibration.setup(0.2);
            beammap.telescope.tau_225_GHz = 0.2;
            beammap.telescope.tel_data["TelElAct"] =
                Eigen::VectorXd::Constant(2, 45.0 * pi / 180.0);
        }
        citlali::pipeline::begin_noise_product_publication(
            cleanup.path, beammap.noise_plan);
        auto buffer = make_production_beammap_noise_buffer(
            static_cast<Eigen::Index>(shape.arrays.size()));

        decltype(beammap.map_fits_outputs.obs) data_files;
        decltype(beammap.map_fits_outputs.obs_noise) realization_files;
        std::vector<std::string> data_bases;
        std::vector<std::string> realization_bases;
        for (Eigen::Index array_index = 0;
             array_index < shape.array_count; ++array_index) {
            data_bases.push_back(
                (cleanup.path /
                 ("array" + std::to_string(array_index) + "_map"))
                    .string());
            realization_bases.push_back(
                (cleanup.path /
                 ("array" + std::to_string(array_index) + "_noise"))
                    .string());
            data_files.emplace_back(data_bases.back());
            realization_files.emplace_back(realization_bases.back());
        }
        auto *data_file_ptr = &data_files;
        auto *realization_file_ptr = &realization_files;
        citlali::pipeline::StageProfileCollector stage_profile;
        ASSERT_NO_THROW(
            beammap.write_beammap_map_products<mapmaking::RawObs>(
                buffer.get(), data_file_ptr, realization_file_ptr,
                stage_profile, cleanup.path.string()));
        EXPECT_TRUE(data_files.empty());
        EXPECT_TRUE(realization_files.empty());

        const std::size_t selected_map_count =
            static_cast<std::size_t>(std::count(
                shape.flags.begin(), shape.flags.end(), 0));
        auto make_mapmaking_plan = [&](std::size_t map_count) {
            citlali::config::MapmakingConfig request;
            request.method = citlali::config::MapMethod::naive;
            request.grouping = citlali::config::MapGrouping::detector;
            citlali::pipeline::MapmakingExecutionPlan plan;
            plan.reset_from_request(
                request, citlali::config::ReductionType::beammap);
            plan.begin_iteration();
            plan.begin_observation(
                0, "152390", map_count, 4.848136811e-6, map_count);
            citlali::pipeline::complete_mapmaking_observation(plan);
            citlali::pipeline::record_mapmaking_run_completed(plan);
            return plan;
        };

        auto inconsistent_plan = beammap.noise_plan;
        auto inconsistent_mapmaking =
            make_mapmaking_plan(selected_map_count + 1);
        EXPECT_THROW(
            citlali::pipeline::record_noise_run_completed(
                inconsistent_plan, inconsistent_mapmaking, false),
            std::logic_error);

        auto mapmaking = make_mapmaking_plan(selected_map_count);
        ASSERT_NO_THROW(citlali::pipeline::record_noise_run_completed(
            beammap.noise_plan, mapmaking, false));
        EXPECT_EQ(
            *beammap.noise_plan.realized.empirical_product_map_count,
            selected_map_count);
        EXPECT_EQ(
            *beammap.noise_plan.realized.realization_image_write_count,
            2U * selected_map_count);
        ASSERT_NO_THROW(citlali::pipeline::write_noise_provenance_file(
            cleanup.path, beammap.noise_plan));

        const auto package = YAML::LoadFile(
            citlali::pipeline::noise_provenance_path(cleanup.path).string());
        const auto members = package["package"]["member_files"];
        ASSERT_TRUE(members.IsSequence());
        EXPECT_EQ(members.size(), 2U * shape.selected_array_count);

        for (Eigen::Index array_index = 0;
             array_index < shape.array_count; ++array_index) {
            std::size_t selected_in_array = 0;
            for (std::size_t detector_index = 0;
                 detector_index < shape.arrays.size(); ++detector_index) {
                if (shape.arrays[detector_index] == array_index &&
                    shape.flags[detector_index] == 0) {
                    ++selected_in_array;
                }
            }
            const std::filesystem::path data_path =
                data_bases[static_cast<std::size_t>(array_index)] +
                "_flag0_good.fits";
            const std::filesystem::path realization_path =
                realization_bases[static_cast<std::size_t>(array_index)] +
                "_flag0_good.fits";
            ASSERT_TRUE(std::filesystem::exists(data_path));
            ASSERT_TRUE(std::filesystem::exists(realization_path));
            if (selected_in_array == 0) {
                EXPECT_THROW(
                    citlali::pipeline::validate_noise_fits_joins(data_path),
                    std::runtime_error);
                auto admitted_empty_plan = beammap.noise_plan;
                citlali::pipeline::record_noise_published_member(
                    admitted_empty_plan, data_path,
                    citlali::pipeline::NoisePublishedMemberKind::fits);
                EXPECT_THROW(
                    citlali::pipeline::write_noise_provenance_file(
                        cleanup.path, admitted_empty_plan),
                    std::runtime_error);
                continue;
            }
            const auto data_join =
                citlali::pipeline::validate_noise_fits_joins(data_path);
            const auto realization_join =
                citlali::pipeline::validate_noise_fits_joins(
                    realization_path);
            EXPECT_EQ(
                data_join.empirical_map_product_count, selected_in_array);
            EXPECT_EQ(
                realization_join.realization_image_count,
                2U * selected_in_array);
            if (shape_index == 0 && array_index == 0) {
                fitsfile *file = nullptr;
                int status = 0;
                ASSERT_EQ(
                    fits_open_file(
                        &file, data_path.c_str(), READONLY, &status),
                    0);
                EXPECT_EQ(
                    read_required_fits_string(file, "CAL.OPERATOR_ID"),
                    "am12_fixed_djf25_piecewise_linear_los_tau_v1");
                EXPECT_DOUBLE_EQ(
                    read_required_fits_double(file, "CAL.ALPHA.EFFECTIVE"),
                    2.0);
                EXPECT_DOUBLE_EQ(
                    read_required_fits_double(file, "CAL.ALPHA.REALIZED"),
                    2.0);
                EXPECT_DOUBLE_EQ(
                    read_required_fits_double(file, "CAL.TAU225"), 0.2);
                EXPECT_EQ(
                    read_required_fits_string(file, "CAL.QUALITY_REGIME"),
                    "engineering_availability_regime");
                EXPECT_DOUBLE_EQ(
                    read_required_fits_double(file, "CAL.X_REF"), 0.0);
                int valid = 0;
                ASSERT_EQ(
                    fits_read_key(
                        file, TLOGICAL, "CAL.VALID", &valid, nullptr,
                        &status),
                    0);
                EXPECT_EQ(valid, 0);
                EXPECT_EQ(
                    read_required_fits_string(
                        file, "CAL.VALIDITY_REASON"),
                    "not_evaluated");
                int extrema_available = 1;
                ASSERT_EQ(
                    fits_read_key(
                        file, TLOGICAL,
                        "CAL.TOTAL_MULTIPLIER_EXTREMA_AVAILABLE",
                        &extrema_available, nullptr, &status),
                    0);
                EXPECT_EQ(extrema_available, 0);
                EXPECT_EQ(fits_close_file(file, &status), 0);
            }
        }
    }
}

TEST(science_map_fits_products,
     production_writer_preserves_wcs_threshold_and_realization_contracts) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    FitsDirectoryCleanup cleanup{
        std::filesystem::path{"/private/tmp"} /
        ("citlali-science-map-production-writer-" +
         std::to_string(nonce))};
    Engine engine;
    configure_production_writer_engine(engine);
    engine.output_paths.obsnum_dir_name = cleanup.path.string() + "/obs/";
    engine.observation_identity.obsnum = "152390";
    std::filesystem::create_directories(cleanup.path / "obs" / "raw");
    std::filesystem::create_directories(cleanup.path / "coadd" / "raw");
    ASSERT_NO_THROW(engine.create_obs_map_files());

    const std::string coadd_map_base =
        (cleanup.path / "coadd" / "raw" / "coadd_map").string();
    const std::string coadd_noise_base =
        (cleanup.path / "coadd" / "raw" / "coadd_noise").string();
    engine.map_fits_outputs.coadd.emplace_back(coadd_map_base);
    engine.map_fits_outputs.coadd_noise.emplace_back(coadd_noise_base);

    constexpr Eigen::Index obs_rows = 9;
    constexpr Eigen::Index obs_cols = 11;
    constexpr Eigen::Index coadd_rows = 13;
    constexpr Eigen::Index coadd_cols = 17;
    constexpr long delta_row = 2;
    constexpr long delta_col = 3;
    auto observation = make_production_science_map_buffer(
        engine, false, obs_rows, obs_cols, {5.0, 4.0});
    auto coadd = make_production_science_map_buffer(
        engine, true, coadd_rows, coadd_cols, {8.0, 6.0});
    auto *observation_map_files = &engine.map_fits_outputs.obs;
    auto *observation_noise_files = &engine.map_fits_outputs.obs_noise;
    auto *coadd_map_files = &engine.map_fits_outputs.coadd;
    auto *coadd_noise_files = &engine.map_fits_outputs.coadd_noise;
    ASSERT_NO_THROW(engine.write_maps(
        observation_map_files, observation_noise_files, observation, 0));
    ASSERT_NO_THROW(engine.write_maps(
        coadd_map_files, coadd_noise_files, coadd, 0));

    const auto observation_map_path =
        engine.map_fits_outputs.obs[0].filepath + ".fits";
    const auto observation_noise_path =
        engine.map_fits_outputs.obs_noise[0].filepath + ".fits";
    const auto coadd_map_path = coadd_map_base + ".fits";
    const auto coadd_noise_path = coadd_noise_base + ".fits";

    decltype(engine.map_fits_outputs.obs) failed_map_files;
    decltype(engine.map_fits_outputs.obs_noise) missing_noise_files;
    const std::string failed_map_base =
        (cleanup.path / "required_write_failure").string();
    failed_map_files.emplace_back(failed_map_base);
    auto failed_observation = make_production_science_map_buffer(
        engine, false, obs_rows, obs_cols, {5.0, 4.0});
    const auto failed_wcs = failed_observation->wcs;
    auto *failed_map_file_ptr = &failed_map_files;
    auto *missing_noise_file_ptr = &missing_noise_files;
    EXPECT_THROW(
        engine.write_maps(
            failed_map_file_ptr, missing_noise_file_ptr, failed_observation,
            0),
        std::runtime_error);
    EXPECT_TRUE(failed_map_files[0].hdus.empty());
    EXPECT_EQ(failed_observation->wcs.cdelt, failed_wcs.cdelt);
    EXPECT_EQ(failed_observation->wcs.crpix, failed_wcs.crpix);
    EXPECT_EQ(failed_observation->wcs.crval, failed_wcs.crval);

    engine.map_fits_outputs.obs[0].publish_atomically();
    engine.map_fits_outputs.obs_noise[0].publish_atomically();
    engine.map_fits_outputs.coadd[0].publish_atomically();
    engine.map_fits_outputs.coadd_noise[0].publish_atomically();
    failed_map_files[0].discard_staged_output();

    fitsfile *observation_file = nullptr;
    fitsfile *observation_noise_file = nullptr;
    fitsfile *coadd_file = nullptr;
    fitsfile *coadd_noise_file = nullptr;
    int status = 0;
    ASSERT_EQ(fits_open_file(&observation_file, observation_map_path.c_str(),
                             READONLY, &status),
              0);
    status = 0;
    ASSERT_EQ(fits_open_file(&observation_noise_file,
                             observation_noise_path.c_str(), READONLY,
                             &status),
              0);
    status = 0;
    ASSERT_EQ(fits_open_file(&coadd_file, coadd_map_path.c_str(), READONLY,
                             &status),
              0);
    status = 0;
    ASSERT_EQ(fits_open_file(&coadd_noise_file, coadd_noise_path.c_str(),
                             READONLY, &status),
              0);

    const auto observation_wcs =
        read_spatial_wcs(observation_file, "signal_I");
    const auto coadd_wcs = read_spatial_wcs(coadd_file, "signal_I");
    const auto &observation_identity =
        observation->science_products.bundle_identity->wcs;
    const auto &coadd_identity =
        coadd->science_products.bundle_identity->wcs;
    const double observation_max_separation =
        maximum_wcs_separation_arcsec(
            observation_identity, observation_wcs);
    const double coadd_max_separation =
        maximum_wcs_separation_arcsec(coadd_identity, coadd_wcs);
    EXPECT_GT(observation_max_separation, 0.0);
    EXPECT_GT(coadd_max_separation, 0.0);
    EXPECT_LE(observation_max_separation, 0.1);
    EXPECT_LE(coadd_max_separation, 0.1);

    for (const auto *wcs : {&observation_wcs, &coadd_wcs}) {
        EXPECT_EQ(wcs->ctype[0], "RA---TAN");
        EXPECT_EQ(wcs->ctype[1], "DEC--TAN");
        EXPECT_EQ(wcs->cunit[0], "deg");
        EXPECT_EQ(wcs->cunit[1], "deg");
        EXPECT_TRUE(std::signbit(wcs->cdelt[0]));
        EXPECT_FALSE(std::signbit(wcs->cdelt[1]));
    }
    EXPECT_DOUBLE_EQ(observation_identity.orientation_rad, 0.0);
    EXPECT_DOUBLE_EQ(coadd_identity.orientation_rad, 0.0);
    EXPECT_EQ(observation_wcs.rows, obs_rows);
    EXPECT_EQ(observation_wcs.cols, obs_cols);
    EXPECT_EQ(coadd_wcs.rows, coadd_rows);
    EXPECT_EQ(coadd_wcs.cols, coadd_cols);
    EXPECT_DOUBLE_EQ(
        coadd_identity.reference_pixel[0],
        observation_identity.reference_pixel[0] + delta_col);
    EXPECT_DOUBLE_EQ(
        coadd_identity.reference_pixel[1],
        observation_identity.reference_pixel[1] + delta_row);
    EXPECT_DOUBLE_EQ(
        coadd_wcs.crpix[0], observation_wcs.crpix[0] + delta_col);
    EXPECT_DOUBLE_EQ(
        coadd_wcs.crpix[1], observation_wcs.crpix[1] + delta_row);

    const auto &coadd_realized = coadd->science_products.realized[0];
    const auto normalization_sidecar =
        citlali::pipeline::science_map_threshold_realization_node(
            coadd_realized.normalization);
    const auto policy_sidecar =
        citlali::pipeline::science_map_threshold_realization_node(
            coadd_realized.science_policy);
    const double normalization_authority =
        citlali::pipeline::science_map_exact_double_value(
            normalization_sidecar["realized_threshold"]);
    const double policy_authority =
        citlali::pipeline::science_map_exact_double_value(
            policy_sidecar["realized_threshold"]);
    auto verify_threshold_card = [&](const std::string &hdu_name,
                                     const std::string &estimator,
                                     double authority) {
        move_to_required_image(coadd_file, hdu_name);
        const double card = read_required_fits_double(coadd_file, "WTTHRESH");
        EXPECT_TRUE(std::isfinite(card)) << hdu_name;
        EXPECT_EQ(read_required_fits_string(coadd_file, "BUNIT"), "1")
            << hdu_name;
        EXPECT_EQ(read_required_fits_string(coadd_file, "ESTTYPE"), estimator)
            << hdu_name;
        EXPECT_LE(std::abs(card - authority),
                  1.0e-12 * std::abs(authority))
            << hdu_name;
        return card;
    };
    verify_threshold_card(
        "normalization_support_I", "normalization_support",
        normalization_authority);
    const double policy_card = verify_threshold_card(
        "science_policy_support_I", "science_policy_support",
        policy_authority);
    const double alias_card = verify_threshold_card(
        "coverage_bool_I", "science_policy_support", policy_authority);
    EXPECT_DOUBLE_EQ(policy_card, alias_card);
    move_to_required_image(coadd_file, "coverage_bool_I");
    EXPECT_EQ(read_required_fits_string(coadd_file, "ALIASOF"),
              "science_policy_support_I");

    ASSERT_TRUE(observation->science_products.bundle_identity);
    ASSERT_TRUE(coadd->science_products.bundle_identity);
    EXPECT_EQ(
        observation->science_products.bundle_identity->response_identity,
        coadd->science_products.bundle_identity->response_identity);
    EXPECT_EQ(
        observation->science_products.bundle_identity->required_companions,
        coadd->science_products.bundle_identity->required_companions);
    const std::vector<std::string> realization_names = {
        "signal_0_I", "signal_1_I"};
    auto verify_realization_file = [&](
        fitsfile *file, const ScienceMapBufferFixture &map) {
        int hdu_count = 0;
        int local_status = 0;
        ASSERT_EQ(fits_get_num_hdus(file, &hdu_count, &local_status), 0);
        EXPECT_EQ(hdu_count, 3);
        for (Eigen::Index realization = 0;
             realization < map.n_noise; ++realization) {
            const auto realization_index =
                static_cast<std::size_t>(realization);
            const auto realization_wcs =
                read_spatial_wcs(file, realization_names[realization_index]);
            EXPECT_EQ(realization_wcs.rows, map.n_rows);
            EXPECT_EQ(realization_wcs.cols, map.n_cols);
            EXPECT_EQ(read_required_fits_string(file, "UNIT"), map.sig_unit);
            std::vector<double> values(
                static_cast<std::size_t>(map.n_rows * map.n_cols));
            int any_null = 0;
            local_status = 0;
            ASSERT_EQ(
                fits_read_img(file, TDOUBLE, 1,
                              static_cast<long>(values.size()), nullptr,
                              values.data(), &any_null, &local_status),
                0);
            for (Eigen::Index row = 0; row < map.n_rows; ++row) {
                for (Eigen::Index output_col = 0;
                     output_col < map.n_cols; ++output_col) {
                    const Eigen::Index source_col =
                        map.n_cols - output_col - 1;
                    const auto flat = static_cast<std::size_t>(
                        row * map.n_cols + output_col);
                    EXPECT_DOUBLE_EQ(
                        values[flat],
                        map.noise[0](row, source_col, realization));
                    if (!map.science_products.normalization_support[0](
                            row, source_col)) {
                        EXPECT_DOUBLE_EQ(values[flat], 0.0);
                    }
                }
            }
        }
    };
    verify_realization_file(observation_noise_file, *observation);
    verify_realization_file(coadd_noise_file, *coadd);

    status = 0;
    EXPECT_EQ(fits_close_file(observation_file, &status), 0);
    status = 0;
    EXPECT_EQ(fits_close_file(observation_noise_file, &status), 0);
    status = 0;
    EXPECT_EQ(fits_close_file(coadd_file, &status), 0);
    status = 0;
    EXPECT_EQ(fits_close_file(coadd_noise_file, &status), 0);
}

TEST(science_map_fits_products,
     admitted_calibration_round_trips_through_actual_map_fits_writer) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    FitsDirectoryCleanup cleanup{
        std::filesystem::path{"/private/tmp"} /
        ("citlali-admitted-calibration-map-writer-" +
         std::to_string(nonce))};
    Engine engine;
    configure_production_writer_engine(engine);
    admit_production_calibration_fixture(engine);
    engine.output_paths.obsnum_dir_name = cleanup.path.string() + "/obs/";
    engine.observation_identity.obsnum = "152390";
    engine.calib.array_fwhms[0] = {10.0, 9.0};
    engine.calib.array_pas[0] = 0.0;
    engine.calib.array_beam_areas[0] = 1.0;
    engine.calib.apt_filepath = "fixture.ecsv";
    engine.toltec_io.array_freq_map[0] = 270.0e9;
    engine.telescope.fsmp = 1.0;
    engine.telescope.source_name = "calibration-writer-fixture";
    engine.telescope.project_id = "SCI-CAL-001";
    engine.telescope.obs_goal = "science";
    engine.telescope.tel_header["Header.Source.Ra"] =
        Eigen::VectorXd::Constant(1, 1.0);
    engine.telescope.tel_header["Header.Source.Dec"] =
        Eigen::VectorXd::Constant(1, 0.5);
    engine.telescope.tel_data["TelElAct"] =
        Eigen::VectorXd::Constant(1, 0.8);
    engine.telescope.tel_data["TelAzAct"] =
        Eigen::VectorXd::Constant(1, 1.2);
    engine.telescope.tel_data["ActParAng"] =
        Eigen::VectorXd::Constant(1, 0.1);
    std::filesystem::create_directories(cleanup.path / "obs" / "raw");
    ASSERT_NO_THROW(engine.create_obs_map_files());
    auto observation = make_production_science_map_buffer(
        engine, false, 5, 7, {3.0, 2.0});
    auto *map_files = &engine.map_fits_outputs.obs;
    auto *noise_files = &engine.map_fits_outputs.obs_noise;
    ASSERT_NO_THROW(engine.add_phdu(map_files, observation, 0));
    ASSERT_NO_THROW(engine.add_phdu(noise_files, observation, 0));
    ASSERT_NO_THROW(engine.write_maps(
        map_files, noise_files, observation, 0));
    const auto path = engine.map_fits_outputs.obs[0].filepath + ".fits";
    engine.map_fits_outputs.obs[0].publish_atomically();
    engine.map_fits_outputs.obs_noise[0].publish_atomically();

    fitsfile *file = nullptr;
    int status = 0;
    ASSERT_EQ(fits_open_file(&file, path.c_str(), READONLY, &status), 0);
    int valid = 0;
    ASSERT_EQ(fits_read_key(
                  file, TLOGICAL, "CAL.VALID", &valid, nullptr, &status),
              0);
    EXPECT_EQ(valid, 1);
    EXPECT_EQ(read_required_fits_string(file, "CAL.TARGET_UNIT"),
              "mJy/beam");
    const auto apt_link = read_required_fits_long_string(
        file, "CAL.APT_ARTIFACT_SHA256");
    const auto acquisition_link = read_required_fits_long_string(
        file, "CAL.ACQUISITION_BINDING_SHA256");
    EXPECT_GE(apt_link.size(), 32U);
    EXPECT_GE(acquisition_link.size(), 32U);
    EXPECT_EQ(engine.rtcproc.calibration.product.apt_artifact_sha256
                  .substr(0, apt_link.size()),
              apt_link);
    EXPECT_EQ(engine.rtcproc.calibration.product.acquisition_binding_sha256
                  .substr(0, acquisition_link.size()),
              acquisition_link);
    EXPECT_EQ(read_required_fits_long_string(file, "CAL.RESPONSE_IDENTITY")
                  .find("calibration-response-basis-provenance-v3"),
              0U);
    EXPECT_EQ(read_required_fits_long_string(
                  file, "CALID"),
              engine.rtcproc.calibration.product.calibration_identity);
    EXPECT_EQ(read_required_fits_long_string(file, "CALPKGID"),
              engine.rtcproc.calibration.product.package_identity);
    int correction_applied = 0;
    double correction_factor = 0.0;
    ASSERT_EQ(fits_read_key(
                  file, TLOGICAL, "CAL.OBS_FLXSCALE_APPLIED",
                  &correction_applied, nullptr, &status),
              0);
    ASSERT_EQ(fits_read_key(
                  file, TDOUBLE, "CAL.OBS_FLXSCALE_FACTOR",
                  &correction_factor, nullptr, &status),
              0);
    EXPECT_EQ(correction_applied, 1);
    EXPECT_DOUBLE_EQ(correction_factor, 3.0);
    EXPECT_EQ(read_required_fits_long_string(
                  file, "CAL.OBS_FLXSCALE_STATE"),
              "applied_once");
    EXPECT_EQ(fits_close_file(file, &status), 0);

    engine.rtcproc.begin_reduced_observation("152391", 0);
    engine.rtcproc.record_finalized_calibration_join(
        "152391", engine.rtcproc.calibration.product.calibration_identity,
        engine.rtcproc.calibration.product.package_identity);
    engine.observation_dates.date_obs = {
        "2026-08-11T00:00:00", "2026-08-11T00:01:00"};
    auto homogeneous_coadd = make_production_science_map_buffer(
        engine, true, 5, 7, {3.0, 2.0});
    homogeneous_coadd->obsnums = {"152390", "152391"};
    decltype(engine.map_fits_outputs.coadd) homogeneous_files;
    const auto homogeneous_base =
        (cleanup.path / "homogeneous_coadd").string();
    homogeneous_files.emplace_back(homogeneous_base);
    auto *homogeneous_file_ptr = &homogeneous_files;
    ASSERT_NO_THROW(engine.add_phdu(
        homogeneous_file_ptr, homogeneous_coadd, 0));
    homogeneous_files[0].publish_atomically();
    fitsfile *coadd_file = nullptr;
    status = 0;
    ASSERT_EQ(fits_open_file(
                  &coadd_file, (homogeneous_base + ".fits").c_str(),
              READONLY, &status),
              0);
    EXPECT_EQ(read_required_fits_long_string(coadd_file, "CALID"),
              engine.rtcproc.calibration.product.calibration_identity);
    EXPECT_EQ(read_required_fits_long_string(coadd_file, "CALPKGID"),
              engine.rtcproc.calibration.product.package_identity);
    EXPECT_EQ(fits_close_file(coadd_file, &status), 0);

    engine.rtcproc.begin_reduced_observation("152392", 0);
    engine.rtcproc.record_finalized_calibration_join(
        "152392", "heterogeneous-calid", "heterogeneous-pkgid");
    auto heterogeneous_coadd = make_production_science_map_buffer(
        engine, true, 5, 7, {3.0, 2.0});
    heterogeneous_coadd->obsnums = {"152390", "152392"};
    decltype(engine.map_fits_outputs.coadd) heterogeneous_files;
    heterogeneous_files.emplace_back(
        (cleanup.path / "heterogeneous_coadd").string());
    auto *heterogeneous_file_ptr = &heterogeneous_files;
    EXPECT_THROW(engine.add_phdu(
                     heterogeneous_file_ptr, heterogeneous_coadd, 0),
                 std::runtime_error);
    EXPECT_TRUE(heterogeneous_files[0].hdus.empty());
}

TEST(science_map_fits_products,
     actual_beammap_apt_writer_reopens_published_ecsv) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    FitsDirectoryCleanup cleanup{
        std::filesystem::path{"/private/tmp"} /
        ("citlali-beammap-apt-writer-" + std::to_string(nonce))};
    Beammap beammap;
    configure_production_beammap_writer(beammap, {0}, {0}, 1);
    beammap.output_paths.obsnum_dir_name = cleanup.path.string() + "/obs/";
    beammap.observation_identity.obsnum = "152390";
    std::filesystem::create_directories(cleanup.path / "obs" / "raw");
    beammap.calib.apt_header_keys = {"uid", "flag", "flag2"};
    beammap.calib.apt["uid"] = Eigen::VectorXd::Constant(1, 42.0);
    beammap.calib.apt["flag"] = Eigen::VectorXd::Zero(1);
    admit_production_calibration_fixture(beammap, false);
    ASSERT_NO_THROW(beammap.populate_beammap_tau_metadata());
    EXPECT_FALSE(beammap.calib.apt_meta["calibration_join_available"]
                     .as<bool>());
    EXPECT_EQ(beammap.calib.apt_meta["calibration_join_state"]
                  .as<std::string>(),
              "pending_finalization");
    EXPECT_FALSE(
        beammap.rtcproc.calibration.product.applied_identity_finalized);
    citlali::pipeline::finalize_complete_calibration_product_identity(
        beammap);
    ASSERT_NO_THROW(beammap.populate_beammap_tau_metadata());
    EXPECT_TRUE(beammap.calib.apt_meta["calibration_join_available"]
                    .as<bool>());
    EXPECT_EQ(beammap.calib.apt_meta["calibration_join_state"]
                  .as<std::string>(),
              "finalized");

    const auto base = beammap.write_beammap_apt_table();
    const auto path = std::filesystem::path(base + ".ecsv");
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    ASSERT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
    const auto [table, header, meta] = to_matrix_from_ecsv(path.string());
    ASSERT_EQ(table.rows(), 1);
    ASSERT_EQ(table.cols(), 3);
    EXPECT_DOUBLE_EQ(table(0, 0), 42.0);
    EXPECT_DOUBLE_EQ(table(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(table(0, 2), 0.0);
    EXPECT_EQ(header,
              (std::vector<std::string>{"uid", "flag", "flag2"}));
    EXPECT_EQ(meta["calibration_identity"].as<std::string>(),
              beammap.rtcproc.calibration.product.calibration_identity);
    EXPECT_EQ(meta["package_identity"].as<std::string>(),
              beammap.rtcproc.calibration.product.package_identity);
    EXPECT_TRUE(
        meta["observation_flxscale_correction_applied"].as<bool>());
    EXPECT_DOUBLE_EQ(
        meta["applied_observation_flxscale_correction"].as<double>(),
        3.0);
    EXPECT_EQ(meta["observation_flxscale_correction_state"]
                  .as<std::string>(),
              "applied_once");
}

TEST(science_map_fits_products,
     beammap_ecsv_atomic_publication_round_trips_nonfinite_diagnostics) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    FitsDirectoryCleanup cleanup{
        std::filesystem::path{"/private/tmp"} /
        ("citlali-beammap-apt-nonfinite-writer-" +
         std::to_string(nonce))};
    Beammap beammap;
    configure_production_beammap_writer(
        beammap, {0, 0, 0}, {0, 0, 0}, 1);
    beammap.output_paths.obsnum_dir_name =
        cleanup.path.string() + "/obs/";
    beammap.observation_identity.obsnum = "152390";
    std::filesystem::create_directories(cleanup.path / "obs" / "raw");
    beammap.calib.apt_header_keys = {
        "uid", "final_prior_d2", "flag", "flag2"};
    beammap.calib.apt["uid"] =
        Eigen::VectorXd::LinSpaced(3, 42.0, 44.0);
    beammap.calib.apt["final_prior_d2"].resize(3);
    beammap.calib.apt["final_prior_d2"] <<
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity();
    admit_production_calibration_fixture(beammap, false);
    ASSERT_NO_THROW(beammap.populate_beammap_tau_metadata());
    citlali::pipeline::finalize_complete_calibration_product_identity(
        beammap);
    ASSERT_NO_THROW(beammap.populate_beammap_tau_metadata());

    const auto base = beammap.write_beammap_apt_table();
    const auto path = std::filesystem::path(base + ".ecsv");
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    ASSERT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
    ASSERT_FALSE(std::filesystem::exists(
        path.string() + ".replace-backup"));

    const auto [table, header, meta] =
        read_uniform_float64_ecsv(path.string());
    ASSERT_EQ(table.rows(), 3);
    ASSERT_EQ(table.cols(), 4);
    EXPECT_EQ(header, (std::vector<std::string>{
                          "uid", "final_prior_d2", "flag", "flag2"}));
    EXPECT_TRUE(std::isnan(table(0, 1)));
    EXPECT_EQ(table(1, 1), std::numeric_limits<double>::infinity());
    EXPECT_EQ(table(2, 1), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(meta["calibration_identity"].as<std::string>(),
              beammap.rtcproc.calibration.product.calibration_identity);
    EXPECT_EQ(meta["package_identity"].as<std::string>(),
              beammap.rtcproc.calibration.product.package_identity);
}

TEST(science_map_fits_products,
     beammap_ecsv_interruption_preserves_existing_valid_final) {
    const auto nonce = std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    const auto base = std::filesystem::path{"/private/tmp"} /
        ("citlali-f008-beammap-ecsv-" + std::to_string(nonce));
    const auto final_path = std::filesystem::path(base.string() + ".ecsv");
    FitsFileCleanup cleanup{final_path.string()};
    std::vector<std::string> header{"uid", "flag"};
    YAML::Node accepted_meta;
    accepted_meta["calibration_join_available"] = true;
    accepted_meta["calibration_identity"] = "accepted-calid";
    accepted_meta["package_identity"] = "accepted-pkgid";
    Eigen::MatrixXd accepted(1, 2);
    accepted << 42.0, 0.0;
    ASSERT_NO_THROW(to_ecsv_from_matrix(
        base.string(), accepted, header, accepted_meta));
    const auto accepted_digest =
        citlali::utils::sha256_file(final_path);

    YAML::Node replacement_meta;
    replacement_meta["calibration_join_available"] = true;
    replacement_meta["calibration_identity"] = "replacement-calid";
    replacement_meta["package_identity"] = "replacement-pkgid";
    Eigen::MatrixXd replacement(1, 2);
    replacement << 84.0, 1.0;
    EXPECT_THROW(
        to_ecsv_from_matrix_validated(
            base.string(), replacement, header, replacement_meta,
            [](const Eigen::MatrixXd &,
               const std::vector<std::string> &, const YAML::Node &) {
                throw std::runtime_error(
                    "interrupted after ECSV reopen validation");
            }),
        citlali::error::Error);
    EXPECT_EQ(citlali::utils::sha256_file(final_path), accepted_digest);
    EXPECT_FALSE(std::filesystem::exists(final_path.string() + ".tmp"));
    EXPECT_FALSE(std::filesystem::exists(
        final_path.string() + ".replace-backup"));

    ASSERT_NO_THROW(to_ecsv_from_matrix(
        base.string(), replacement, header, replacement_meta));
    const auto [reopened_table, reopened_header, reopened_meta] =
        to_matrix_from_ecsv(final_path.string());
    ASSERT_EQ(reopened_table.rows(), 1);
    ASSERT_EQ(reopened_table.cols(), 2);
    EXPECT_DOUBLE_EQ(reopened_table(0, 0), 84.0);
    EXPECT_EQ(reopened_header, header);
    EXPECT_EQ(reopened_meta["calibration_identity"].as<std::string>(),
              "replacement-calid");
    EXPECT_EQ(reopened_meta["package_identity"].as<std::string>(),
              "replacement-pkgid");
    EXPECT_FALSE(std::filesystem::exists(final_path.string() + ".tmp"));
}

TEST(science_map_fits_products,
     unavailable_profile_representation_has_explicit_absence) {
    mapmaking::ScienceMapProducts detector;
    detector.allocate(2, 3, 4, false, false, false);
    EXPECT_TRUE(detector.initialized);
    EXPECT_FALSE(detector.ordinary_contribution_predicate_available);
    EXPECT_TRUE(detector.geometric_hits.empty());
    ASSERT_EQ(detector.realized.size(), 2U);
    for (const auto &record : detector.realized) {
        for (const auto &reason : record.product_absence_reason) {
            EXPECT_EQ(reason,
                      "method-specific contribution predicate unavailable");
        }
    }
    EXPECT_TRUE(citlali::pipeline::science_map_unavailable_output_bundle_complete(
        detector, 2));
}

TEST(science_map_fits_products,
     detector_profile_preserves_empty_coverage_output_guard) {
    auto detector = std::make_shared<mapmaking::MapBuffer>("omb");
    detector->n_rows = 3;
    detector->n_cols = 4;
    detector->signal = {Eigen::MatrixXd::Ones(3, 4)};
    detector->weight = {Eigen::MatrixXd::Ones(3, 4)};
    detector->science_products.allocate(
        1, 3, 4, false, false, false,
        "detector-grouping science-map product profile is unavailable");
    ASSERT_TRUE(detector->coverage.empty());
    ASSERT_FALSE(citlali::pipeline::science_map_successor_coadd_product(
        detector->science_products));
    CapturedFitsEntry output;
    DummyWcs wcs;

    EXPECT_NO_THROW(citlali::pipeline::add_coverage_support_image_hdus(
        output, detector, 0, "detector_", "I", wcs, 2000.0, false,
        true, false, science_map_test_logger()));
    EXPECT_TRUE(output.images.empty());
}

TEST(science_map_fits_products,
     unavailable_legacy_coadd_is_not_promoted_to_successor_output_policy) {
    mapmaking::ScienceMapProducts legacy_coadd;
    legacy_coadd.allocate(
        1, 3, 4, true, false, false,
        "method-specific contribution predicate unavailable");

    EXPECT_FALSE(citlali::pipeline::science_map_successor_coadd_product(
        legacy_coadd));
}

TEST(science_map_fits_products,
     rejects_tampered_unavailable_inventory_before_first_hdu) {
    mapmaking::ScienceMapProducts unavailable;
    unavailable.allocate(
        1, 3, 4, false, false, false,
        "non-array map-grouping science-map product profile is unavailable");
    unavailable.realized[0].product_absence_reason[0].clear();
    CapturedFitsEntry output;

    EXPECT_THROW(
        citlali::pipeline::require_science_map_output_profile_authority(
            unavailable, 1, 3, 4, science_map_test_logger()),
        std::runtime_error);
    EXPECT_TRUE(output.images.empty());

    unavailable.realized[0].product_absence_reason[0] = "restored";
    unavailable.geometric_hits.emplace_back(
        mapmaking::ScienceMapCountPlane::Zero(3, 4));
    EXPECT_THROW(
        citlali::pipeline::require_science_map_output_profile_authority(
            unavailable, 1, 3, 4, science_map_test_logger()),
        std::runtime_error);
    EXPECT_TRUE(output.images.empty());
}

}  // namespace
