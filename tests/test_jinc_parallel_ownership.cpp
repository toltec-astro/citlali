#include <citlali/core/mapmaking/jinc_mm.h>
#include <citlali/core/pipeline/map_buffer_allocation.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using PtcData =
    timestream::TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd>;
using Apt = std::map<std::string, Eigen::VectorXd>;

constexpr Eigen::Index kRows = 7;
constexpr Eigen::Index kCols = 7;
constexpr Eigen::Index kSamples = 8;
constexpr Eigen::Index kDetectors = 4;
constexpr Eigen::Index kMaps = 5;
constexpr Eigen::Index kNoiseRealizations = 2;
constexpr double kPixelSizeRad = 1.0e-5;
constexpr double kSampleRateHz = 8.0;

struct Fixture {
    PtcData data;
    Apt apt;
    Eigen::VectorXi map_indices;
};

struct RunOptions {
    std::string policy = "seq";
    int subpixel_n = 1;
    bool noise_in_cmb = false;
    bool run_noise = true;
    bool run_kernel = true;
    bool run_coverage = true;
    bool contribution_diag = true;
    int repetitions = 1;
    const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps = nullptr;
};

struct RunResult {
    mapmaking::MapBuffer omb;
    mapmaking::MapBuffer cmb;
};

Fixture make_fixture(bool reverse_detectors = false,
                     bool invalid_weights = false) {
    Fixture fixture;
    auto &data = fixture.data;
    data.scans.data.resize(kSamples, kDetectors);
    data.kernel.data.resize(kSamples, kDetectors);
    for (Eigen::Index sample = 0; sample < kSamples; ++sample) {
        for (Eigen::Index det = 0; det < kDetectors; ++det) {
            data.scans.data(sample, det) =
                static_cast<double>((sample + 1) * (det + 2)) - 3.25;
            data.kernel.data(sample, det) =
                0.125 * static_cast<double>((sample + 2) * (det + 1));
        }
    }
    data.scans.data(3, 2) = std::numeric_limits<double>::quiet_NaN();
    data.flags.data.resize(kSamples, kDetectors);
    data.flags.data.setConstant(false);
    data.flags.data(1, 0) = true;
    data.flags.data(5, 3) = true;
    data.weights.data.resize(kDetectors);
    data.weights.data << 1.25, 0.75, 2.0, 1.5;
    if (invalid_weights) {
        data.weights.data(1) = std::numeric_limits<double>::quiet_NaN();
        data.weights.data(3) = -1.0;
    }
    data.noise.data.resize(kNoiseRealizations, kDetectors);
    data.noise.data << 1, -1, 1, -1,
                      -1, 1, 1, -1;
    data.index.data = 23;

    Eigen::VectorXd lat(kSamples);
    Eigen::VectorXd lon(kSamples);
    lat << -3.4, -3.0, -1.65, -0.2, 0.45, 2.35, 3.0, 4.2;
    lon << 3.0, -2.4, -0.55, 0.3, 1.75, -3.0, 3.4, -4.1;
    lat *= kPixelSizeRad;
    lon *= kPixelSizeRad;
    data.tel_data.data["TelElAct"] = Eigen::VectorXd::Zero(kSamples);
    data.tel_data.data["alt_phys"] = lat;
    data.tel_data.data["az_phys"] = lon;
    data.pointing_offsets_arcsec.data["az"] =
        Eigen::VectorXd::Zero(kSamples);
    data.pointing_offsets_arcsec.data["alt"] =
        Eigen::VectorXd::Zero(kSamples);

    fixture.apt["array"] = Eigen::VectorXd::Zero(kDetectors);
    fixture.apt["flag"] = Eigen::VectorXd::Zero(kDetectors);
    fixture.apt["x_t"] = Eigen::VectorXd::Zero(kDetectors);
    fixture.apt["y_t"] = Eigen::VectorXd::Zero(kDetectors);
    fixture.apt["uid"].resize(kDetectors);
    fixture.apt["uid"] << 4101.0, 4102.0, 4103.0, 4104.0;
    fixture.map_indices.resize(kDetectors);
    fixture.map_indices << 3, 1, 4, 0;

    if (reverse_detectors) {
        const std::array<Eigen::Index, kDetectors> order{3, 2, 1, 0};
        auto scans = data.scans.data;
        auto kernel = data.kernel.data;
        auto flags = data.flags.data;
        auto weights = data.weights.data;
        auto noise = data.noise.data;
        auto map_indices = fixture.map_indices;
        auto apt = fixture.apt;
        for (Eigen::Index new_col = 0; new_col < kDetectors; ++new_col) {
            const Eigen::Index old_col = order[static_cast<std::size_t>(new_col)];
            data.scans.data.col(new_col) = scans.col(old_col);
            data.kernel.data.col(new_col) = kernel.col(old_col);
            data.flags.data.col(new_col) = flags.col(old_col);
            data.weights.data(new_col) = weights(old_col);
            data.noise.data.col(new_col) = noise.col(old_col);
            fixture.map_indices(new_col) = map_indices(old_col);
            for (auto &[name, values] : fixture.apt) {
                values(new_col) = apt.at(name)(old_col);
            }
        }
    }
    return fixture;
}

mapmaking::MapBuffer make_map(const std::string &name, bool with_signal,
                              bool with_noise, bool with_kernel,
                              bool with_coverage, bool contribution_diag,
                              const std::string &policy) {
    mapmaking::MapBuffer map{name};
    map.n_rows = kRows;
    map.n_cols = kCols;
    map.pixel_size_rad = kPixelSizeRad;
    map.map_grouping = "detector";
    map.parallel_policy = policy;
    map.n_noise = with_noise ? kNoiseRealizations : 0;
    map.randomize_dets = true;
    if (with_signal) {
        citlali::pipeline::allocate_map_matrices(
            map, kMaps, true, with_kernel, with_coverage, false);
    }
    if (with_noise) {
        for (Eigen::Index map_index = 0; map_index < kMaps; ++map_index) {
            map.noise.emplace_back(kRows, kCols, kNoiseRealizations);
            map.noise.back().setZero();
        }
    }
    map.contribution_diag_enabled = contribution_diag;
    return map;
}

mapmaking::JincMapmaker make_mapmaker(int subpixel_n) {
    mapmaking::JincMapmaker mapmaker;
    mapmaker.run_polarization = false;
    mapmaker.subpixel_n = subpixel_n;
    Eigen::MatrixXd kernel(3, 3);
    kernel << 0.125, 0.25, 0.125,
              0.25, 1.0, 0.25,
              0.125, 0.25, 0.125;
    mapmaker.jinc_weights_mat[0] = kernel;
    mapmaker.jinc_weights_sq_mat[0] = kernel.array().square().matrix();
    if (subpixel_n > 1) {
        const std::array<double, 4> scales{0.875, 0.9375, 1.0625, 1.125};
        for (double scale : scales) {
            Eigen::MatrixXd shifted = kernel * scale;
            mapmaker.jinc_weights_mat_subpix[0].push_back(shifted);
            mapmaker.jinc_weights_sq_mat_subpix[0].push_back(
                shifted.array().square().matrix());
        }
    }
    return mapmaker;
}

RunResult run_fixture(Fixture fixture, const RunOptions &options,
                      bool parallel) {
    RunResult result{
        make_map("omb", true, options.run_noise && !options.noise_in_cmb,
                 options.run_kernel, options.run_coverage,
                 options.contribution_diag, options.policy),
        make_map("cmb", false, options.run_noise && options.noise_in_cmb,
                 false, false, false, options.policy)};
    auto mapmaker = make_mapmaker(options.subpixel_n);
    std::string pixel_axes = "altaz";
    for (int repetition = 0; repetition < options.repetitions; ++repetition) {
        fixture.data.index.data = 23 + repetition;
        if (parallel) {
            mapmaker.populate_maps_jinc_parallel(
                fixture.data, result.omb, result.cmb, fixture.map_indices,
                pixel_axes, fixture.apt, kSampleRateHz, true,
                options.run_noise, options.active_maps);
        }
        else {
            mapmaker.populate_maps_jinc(
                fixture.data, result.omb, result.cmb, fixture.map_indices,
                pixel_axes, fixture.apt, kSampleRateHz, true,
                options.run_noise, options.active_maps);
        }
    }
    return result;
}

template <class Lhs, class Rhs>
void expect_matrix_exact(const Lhs &lhs, const Rhs &rhs) {
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    for (Eigen::Index col = 0; col < lhs.cols(); ++col) {
        for (Eigen::Index row = 0; row < lhs.rows(); ++row) {
            if constexpr (std::is_floating_point_v<typename Lhs::Scalar>) {
                EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
                    static_cast<double>(lhs(row, col)),
                    static_cast<double>(rhs(row, col))))
                    << "row=" << row << " col=" << col;
            }
            else {
                EXPECT_EQ(lhs(row, col), rhs(row, col))
                    << "row=" << row << " col=" << col;
            }
        }
    }
}

void expect_matrix_vector_exact(const std::vector<Eigen::MatrixXd> &lhs,
                                const std::vector<Eigen::MatrixXd> &rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        SCOPED_TRACE("map=" + std::to_string(i));
        expect_matrix_exact(lhs[i], rhs[i]);
    }
}

void expect_int_matrix_vector_exact(const std::vector<Eigen::MatrixXi> &lhs,
                                    const std::vector<Eigen::MatrixXi> &rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        SCOPED_TRACE("map=" + std::to_string(i));
        expect_matrix_exact(lhs[i], rhs[i]);
    }
}

void expect_noise_exact(const std::vector<Eigen::Tensor<double, 3>> &lhs,
                        const std::vector<Eigen::Tensor<double, 3>> &rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());
    for (std::size_t map_index = 0; map_index < lhs.size(); ++map_index) {
        ASSERT_EQ(lhs[map_index].dimensions(), rhs[map_index].dimensions());
        for (Eigen::Index realization = 0;
             realization < lhs[map_index].dimension(2); ++realization) {
            for (Eigen::Index col = 0; col < lhs[map_index].dimension(1);
                 ++col) {
                for (Eigen::Index row = 0; row < lhs[map_index].dimension(0);
                     ++row) {
                    EXPECT_TRUE(mapmaking::science_map_exact_double_equal(
                        lhs[map_index](row, col, realization),
                        rhs[map_index](row, col, realization)))
                        << "map=" << map_index << " realization=" << realization
                        << " row=" << row << " col=" << col;
                }
            }
        }
    }
}

void expect_map_exact(const mapmaking::MapBuffer &lhs,
                      const mapmaking::MapBuffer &rhs) {
    expect_matrix_vector_exact(lhs.signal, rhs.signal);
    expect_matrix_vector_exact(lhs.grid_weight, rhs.grid_weight);
    expect_matrix_vector_exact(lhs.weight, rhs.weight);
    expect_matrix_vector_exact(lhs.coverage, rhs.coverage);
    expect_matrix_vector_exact(lhs.kernel, rhs.kernel);
    expect_noise_exact(lhs.noise, rhs.noise);
    expect_matrix_vector_exact(lhs.contribution_max_abs,
                               rhs.contribution_max_abs);
    expect_matrix_vector_exact(lhs.contribution_signal,
                               rhs.contribution_signal);
    expect_matrix_vector_exact(lhs.contribution_weight,
                               rhs.contribution_weight);
    expect_matrix_vector_exact(lhs.contribution_variance_weight,
                               rhs.contribution_variance_weight);
    expect_matrix_vector_exact(lhs.contribution_total_signal,
                               rhs.contribution_total_signal);
    expect_matrix_vector_exact(lhs.contribution_total_weight,
                               rhs.contribution_total_weight);
    expect_matrix_vector_exact(lhs.contribution_total_variance_weight,
                               rhs.contribution_total_variance_weight);
    expect_int_matrix_vector_exact(lhs.contribution_uid, rhs.contribution_uid);
    expect_int_matrix_vector_exact(lhs.contribution_scan, rhs.contribution_scan);
    expect_int_matrix_vector_exact(lhs.contribution_sample,
                                   rhs.contribution_sample);
}

void seed_destination_state(mapmaking::MapBuffer &map) {
    double value = 10.0;
    auto seed_matrices = [&](auto &planes) {
        for (auto &plane : planes) {
            plane.setConstant(value);
            value += 1.0;
        }
    };
    seed_matrices(map.signal);
    seed_matrices(map.grid_weight);
    seed_matrices(map.weight);
    seed_matrices(map.coverage);
    seed_matrices(map.kernel);
    for (auto &noise : map.noise) {
        noise.setConstant(value);
        value += 1.0;
    }
}

std::string invoke_invalid(
    const Eigen::VectorXi &indices,
    const std::function<void(mapmaking::MapBuffer &, mapmaking::MapBuffer &)> &mutate = {}) {
    auto fixture = make_fixture();
    fixture.map_indices = indices;
    auto omb = make_map("omb", true, true, true, true, true, "seq");
    auto cmb = make_map("cmb", false, false, false, false, false, "seq");
    seed_destination_state(omb);
    if (mutate) {
        mutate(omb, cmb);
    }
    const auto omb_before = omb;
    const auto cmb_before = cmb;
    auto mapmaker = make_mapmaker(2);
    std::string pixel_axes = "altaz";
    try {
        mapmaker.populate_maps_jinc_parallel(
            fixture.data, omb, cmb, fixture.map_indices, pixel_axes,
            fixture.apt, kSampleRateHz, true, true);
    }
    catch (const std::runtime_error &error) {
        expect_map_exact(omb, omb_before);
        expect_map_exact(cmb, cmb_before);
        return error.what();
    }
    ADD_FAILURE() << "invalid ownership input did not fail closed";
    expect_map_exact(omb, omb_before);
    expect_map_exact(cmb, cmb_before);
    return {};
}

TEST(JincParallelOwnership, DuplicateMapIsRejectedBeforeMutation) {
    Eigen::VectorXi indices(kDetectors);
    indices << 0, 0, 2, 3;
    EXPECT_EQ(
        invoke_invalid(indices),
        "populate_maps_jinc_parallel ownership-preflight: duplicate "
        "map_index=0 for det_col=1; first owned by det_col=0");
}

TEST(JincParallelOwnership, InvalidSizesAndIndicesFailBeforeMutation) {
    Eigen::VectorXi undersized(kDetectors - 1);
    undersized << 0, 1, 2;
    EXPECT_EQ(
        invoke_invalid(undersized),
        "populate_maps_jinc_parallel ownership-preflight: "
        "map_indices.size()=3 does not match n_dets=4");

    Eigen::VectorXi oversized(kDetectors + 1);
    oversized << 0, 1, 2, 3, 4;
    EXPECT_EQ(
        invoke_invalid(oversized),
        "populate_maps_jinc_parallel ownership-preflight: "
        "map_indices.size()=5 does not match n_dets=4");

    Eigen::VectorXi negative(kDetectors);
    negative << 0, -1, 2, 3;
    EXPECT_EQ(
        invoke_invalid(negative),
        "populate_maps_jinc_parallel ownership-preflight: det_col=1 "
        "map_index=-1 is outside omb.signal [0, 4]");

    Eigen::VectorXi out_of_range(kDetectors);
    out_of_range << 0, 5, 2, 3;
    EXPECT_EQ(
        invoke_invalid(out_of_range),
        "populate_maps_jinc_parallel ownership-preflight: det_col=1 "
        "map_index=5 is outside omb.signal [0, 4]");
}

TEST(JincParallelOwnership, InconsistentDestinationsFailBeforeMutation) {
    Eigen::VectorXi indices(kDetectors);
    indices << 0, 1, 2, 4;

    EXPECT_EQ(
        invoke_invalid(indices, [](auto &omb, auto &) {
            omb.weight.pop_back();
        }),
        "populate_maps_jinc_parallel ownership-preflight: "
        "omb.weight.size()=4 does not match omb.signal.size()=5");
    EXPECT_EQ(
        invoke_invalid(indices, [](auto &omb, auto &) {
            omb.grid_weight.pop_back();
        }),
        "populate_maps_jinc_parallel ownership-preflight: "
        "omb.grid_weight.size()=4 does not match omb.signal.size()=5");
    EXPECT_EQ(
        invoke_invalid(indices, [](auto &omb, auto &) {
            omb.coverage.pop_back();
        }),
        "populate_maps_jinc_parallel ownership-preflight: "
        "omb.coverage.size()=4 does not match omb.signal.size()=5");
    EXPECT_EQ(
        invoke_invalid(indices, [](auto &omb, auto &) {
            omb.kernel.pop_back();
        }),
        "populate_maps_jinc_parallel ownership-preflight: "
        "omb.kernel.size()=4 does not match omb.signal.size()=5");
    EXPECT_EQ(
        invoke_invalid(indices, [](auto &omb, auto &) {
            omb.noise.pop_back();
        }),
        "populate_maps_jinc_parallel ownership-preflight: det_col=3 "
        "map_index=4 is outside omb.noise [0, 3]");
    EXPECT_EQ(
        invoke_invalid(indices, [](auto &omb, auto &) {
            omb.signal[4].conservativeResize(kRows - 1, kCols);
        }),
        "populate_maps_jinc_parallel ownership-preflight: omb.signal[4] "
        "has dims 6x7; expected 7x7");
}

TEST(JincParallelOwnership, ValidUniqueMappingsMatchSerialExactly) {
    Eigen::Matrix<bool, Eigen::Dynamic, 1> active_maps(kMaps);
    active_maps << true, false, true, true, true;
    for (int subpixel_n : {1, 2}) {
        for (bool noise_in_cmb : {false, true}) {
            for (bool reverse_detectors : {false, true}) {
                for (bool invalid_weights : {false, true}) {
                    for (bool identity_mapping : {false, true}) {
                        SCOPED_TRACE("subpixel_n=" +
                                     std::to_string(subpixel_n));
                        SCOPED_TRACE("noise_in_cmb=" +
                                     std::to_string(noise_in_cmb));
                        SCOPED_TRACE("reverse_detectors=" +
                                     std::to_string(reverse_detectors));
                        SCOPED_TRACE("invalid_weights=" +
                                     std::to_string(invalid_weights));
                        SCOPED_TRACE("identity_mapping=" +
                                     std::to_string(identity_mapping));
                        RunOptions options;
                        options.policy = "seq";
                        options.subpixel_n = subpixel_n;
                        options.noise_in_cmb = noise_in_cmb;
                        options.active_maps = &active_maps;
                        auto fixture =
                            make_fixture(reverse_detectors, invalid_weights);
                        if (identity_mapping) {
                            fixture.map_indices << 0, 1, 2, 3;
                        }
                        const auto sequential =
                            run_fixture(fixture, options, false);
                        const auto parallel_entry =
                            run_fixture(fixture, options, true);
                        expect_map_exact(parallel_entry.omb, sequential.omb);
                        expect_map_exact(parallel_entry.cmb, sequential.cmb);
                    }
                }
            }
        }
    }
}

TEST(JincParallelOwnership,
     ValidUniqueParallelPoliciesPreserveExactRepeatedResults) {
    Eigen::Matrix<bool, Eigen::Dynamic, 1> active_maps(kMaps);
    active_maps << true, true, false, true, true;
    for (bool reverse_detectors : {false, true}) {
        for (bool noise_in_cmb : {false, true}) {
            for (bool optional_destinations : {false, true}) {
                SCOPED_TRACE("reverse_detectors=" +
                             std::to_string(reverse_detectors));
                SCOPED_TRACE("noise_in_cmb=" +
                             std::to_string(noise_in_cmb));
                SCOPED_TRACE("optional_destinations=" +
                             std::to_string(optional_destinations));
                RunOptions reference_options;
                reference_options.policy = "seq";
                reference_options.subpixel_n = 2;
                reference_options.noise_in_cmb = noise_in_cmb;
                reference_options.run_kernel = optional_destinations;
                reference_options.run_coverage = optional_destinations;
                reference_options.contribution_diag = optional_destinations;
                reference_options.repetitions = 3;
                reference_options.active_maps = &active_maps;
                auto fixture = make_fixture(reverse_detectors, false);
                const auto reference =
                    run_fixture(fixture, reference_options, true);

                auto omp_options = reference_options;
                omp_options.policy = "omp";
                const auto omp_result = run_fixture(fixture, omp_options, true);
                expect_map_exact(omp_result.omb, reference.omb);
                expect_map_exact(omp_result.cmb, reference.cmb);
            }
        }
    }
}

TEST(JincParallelOwnership, ValidUniqueMappingAllowsExtraMapSlots) {
    auto fixture = make_fixture();
    ASSERT_EQ(fixture.map_indices.size(), kDetectors);
    ASSERT_EQ(kMaps, kDetectors + 1);
    RunOptions options;
    options.policy = "omp";
    const auto result = run_fixture(fixture, options, true);
    EXPECT_TRUE((result.omb.signal[2].array() == 0.0).all());
    EXPECT_TRUE((result.omb.grid_weight[2].array() == 0.0).all());
    EXPECT_TRUE((result.omb.weight[2].array() == 0.0).all());
    EXPECT_TRUE((result.omb.coverage[2].array() == 0.0).all());
    EXPECT_TRUE((result.omb.kernel[2].array() == 0.0).all());
}

}  // namespace
