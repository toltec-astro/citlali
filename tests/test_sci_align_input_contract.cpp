#include <citlali/core/engine/detail/sci_align_netcdf_input_contract.h>
#include <citlali/core/engine/detail/sci_align_packet_slot_contract.h>
#include <citlali/core/pipeline/output_netcdf_metadata.h>

#include <gtest/gtest.h>

#include <netcdf>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

namespace nc_contract =
    citlali::engine_detail::sci_align_netcdf;

TEST(sci_align_input_contract,
     rejects_dimension_narrowing_and_matrix_cardinality_before_allocation) {
    EXPECT_EQ(nc_contract::require_eigen_index_size(17, "vector"), 17);
    EXPECT_EQ(nc_contract::require_eigen_matrix_rows(17, 6, "matrix"),
              17);

    if constexpr (sizeof(std::size_t) >= sizeof(Eigen::Index)) {
        const auto eigen_max = static_cast<std::size_t>(
            std::numeric_limits<Eigen::Index>::max());
        EXPECT_THROW(nc_contract::require_eigen_index_size(
                         eigen_max + 1, "vector"),
                     DataIOError);
        EXPECT_THROW(nc_contract::require_eigen_matrix_rows(
                         eigen_max / 6 + 1, 6, "matrix"),
                     DataIOError);
    }
    EXPECT_THROW(
        nc_contract::require_eigen_matrix_rows(1, 0, "matrix"),
        DataIOError);
}

class TemporaryInputContractFile {
public:
    TemporaryInputContractFile() {
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("citlali-sci-align-input-contract-" +
                std::to_string(nonce) + ".nc");
    }

    ~TemporaryInputContractFile() {
        std::error_code error;
        std::filesystem::remove(path, error);
        std::filesystem::remove(path.string() + ".tmp", error);
    }

    std::filesystem::path path;
};

TEST(sci_align_input_contract,
     rejects_missing_wrong_rank_shape_type_dimension_and_units_before_read) {
    TemporaryInputContractFile temporary;
    {
        netCDF::NcFile file(temporary.path.string(),
                            netCDF::NcFile::replace);
        const auto time = file.addDim("time", 3);
        const auto same_size_wrong_identity =
            file.addDim("other_time", 3);
        const auto six = file.addDim("tlen", 6);
        const auto five = file.addDim("wrong_tlen", 5);
        const auto text_width = file.addDim("text_width", 128);
        const auto wrong_text_width =
            file.addDim("wrong_text_width", 129);

        auto scalar = file.addVar("scalar", netCDF::ncInt);
        const int scalar_value = 7;
        scalar.putVar(&scalar_value);

        auto vector = file.addVar("vector", netCDF::ncDouble, time);
        vector.putAtt("units", "sec");
        const double vector_values[] = {1.0, 2.0, 3.0};
        vector.putVar(vector_values);

        auto wrong_dimension = file.addVar(
            "wrong_dimension", netCDF::ncDouble,
            same_size_wrong_identity);
        wrong_dimension.putAtt("units", "sec");
        wrong_dimension.putVar(vector_values);

        auto wrong_vector_type =
            file.addVar("wrong_vector_type", netCDF::ncInt, time);
        wrong_vector_type.putAtt("units", "sec");
        const int integer_values[] = {1, 2, 3};
        wrong_vector_type.putVar(integer_values);

        auto wrong_units =
            file.addVar("wrong_units", netCDF::ncDouble, time);
        wrong_units.putAtt("units", "rad");
        wrong_units.putVar(vector_values);

        auto text = file.addVar("text", netCDF::ncChar, text_width);
        std::vector<char> text_values(128, ' ');
        const std::string label = "Map";
        std::copy(label.begin(), label.end(), text_values.begin());
        text.putVar(text_values.data());
        auto nul_padded_text = file.addVar(
            "nul_padded_text", netCDF::ncChar, text_width);
        std::vector<char> nul_padded_text_values(128, '\0');
        std::copy(label.begin(), label.end(),
                  nul_padded_text_values.begin());
        nul_padded_text.putVar(nul_padded_text_values.data());
        auto malformed_space_padding = file.addVar(
            "malformed_space_padding", netCDF::ncChar, text_width);
        auto malformed_space_padding_values = text_values;
        malformed_space_padding_values[4] = 'X';
        malformed_space_padding.putVar(
            malformed_space_padding_values.data());
        auto malformed_nul_padding = file.addVar(
            "malformed_nul_padding", netCDF::ncChar, text_width);
        auto malformed_nul_padding_values = nul_padded_text_values;
        malformed_nul_padding_values[4] = 'X';
        malformed_nul_padding.putVar(
            malformed_nul_padding_values.data());
        auto wrong_text = file.addVar(
            "wrong_text", netCDF::ncChar, wrong_text_width);
        std::vector<char> wrong_text_values(129, ' ');
        wrong_text.putVar(wrong_text_values.data());

        const std::vector<netCDF::NcDim> exact_matrix_dims{time, six};
        auto exact_matrix = file.addVar(
            "exact_matrix", netCDF::ncInt, exact_matrix_dims);
        const int matrix_values[18] = {};
        exact_matrix.putVar(matrix_values);

        const std::vector<netCDF::NcDim> wrong_matrix_dims{time, five};
        auto wrong_matrix = file.addVar(
            "wrong_matrix", netCDF::ncInt, wrong_matrix_dims);
        const int wrong_matrix_values[15] = {};
        wrong_matrix.putVar(wrong_matrix_values);
        file.close();
    }

    netCDF::NcFile file(temporary.path.string(), netCDF::NcFile::read);
    EXPECT_THROW(nc_contract::require_variable(file, "absent"),
                 DataIOError);

    const auto scalar = nc_contract::require_variable(file, "scalar");
    EXPECT_NO_THROW(nc_contract::require_scalar(scalar, "scalar"));
    EXPECT_THROW(nc_contract::require_units(scalar, "scalar", {"count"}),
                 DataIOError);
    EXPECT_THROW(nc_contract::require_nonempty_vector(scalar, "scalar"),
                 DataIOError);

    const auto vector = nc_contract::require_variable(file, "vector");
    const auto time =
        nc_contract::require_nonempty_vector(vector, "vector");
    EXPECT_THROW(nc_contract::require_scalar(vector, "vector"),
                 DataIOError);
    EXPECT_NO_THROW(nc_contract::require_type(
        vector, "vector", {netCDF::NcType::nc_DOUBLE}));
    EXPECT_NO_THROW(nc_contract::require_units(vector, "vector", {"sec"}));

    const auto wrong_dimension =
        nc_contract::require_variable(file, "wrong_dimension");
    EXPECT_THROW(nc_contract::require_vector_on_dimension(
                     wrong_dimension, "wrong_dimension", time),
                 DataIOError);
    const auto wrong_vector_type =
        nc_contract::require_variable(file, "wrong_vector_type");
    EXPECT_THROW(nc_contract::require_type(
                     wrong_vector_type, "wrong_vector_type",
                     {netCDF::NcType::nc_DOUBLE}),
                 DataIOError);
    const auto wrong_units =
        nc_contract::require_variable(file, "wrong_units");
    EXPECT_THROW(
        nc_contract::require_units(wrong_units, "wrong_units", {"sec"}),
        DataIOError);
    const auto text = nc_contract::require_variable(file, "text");
    EXPECT_EQ(nc_contract::read_fixed_width_text(text, "text", 128),
              "Map");
    const auto nul_padded_text =
        nc_contract::require_variable(file, "nul_padded_text");
    EXPECT_EQ(nc_contract::read_fixed_width_text(
                  nul_padded_text, "nul_padded_text", 128),
              "Map");
    const auto malformed_space_padding =
        nc_contract::require_variable(file, "malformed_space_padding");
    EXPECT_THROW(nc_contract::read_fixed_width_text(
                     malformed_space_padding, "malformed_space_padding",
                     128),
                 DataIOError);
    const auto malformed_nul_padding =
        nc_contract::require_variable(file, "malformed_nul_padding");
    EXPECT_THROW(nc_contract::read_fixed_width_text(
                     malformed_nul_padding, "malformed_nul_padding", 128),
                 DataIOError);
    const auto wrong_text =
        nc_contract::require_variable(file, "wrong_text");
    EXPECT_THROW(nc_contract::read_fixed_width_text(
                     wrong_text, "wrong_text", 128),
                 DataIOError);

    const auto exact_matrix =
        nc_contract::require_variable(file, "exact_matrix");
    EXPECT_NO_THROW(nc_contract::require_matrix_shape(
        exact_matrix, "exact_matrix", 6));
    const auto wrong_matrix =
        nc_contract::require_variable(file, "wrong_matrix");
    EXPECT_THROW(nc_contract::require_matrix_shape(
                     wrong_matrix, "wrong_matrix", 6),
                 DataIOError);
}

TEST(sci_align_input_contract,
     accepts_only_the_selected_legacy_toltec_timing_schema_fingerprint) {
    TemporaryInputContractFile temporary;
    {
        netCDF::NcFile file(temporary.path.string(),
                            netCDF::NcFile::replace);
        const auto time = file.addDim("time", 3);
        const auto other_time = file.addDim("other_time", 3);
        const auto detector = file.addDim("iqlen", 2);
        const auto other_detector = file.addDim("other_iqlen", 2);
        const auto timestamp_component = file.addDim("tlen", 6);

        const std::vector<netCDF::NcDim> detector_dimensions{
            time, detector};
        auto in_phase = file.addVar("Data.Toltec.Is", netCDF::ncInt,
                                    detector_dimensions);
        auto quadrature = file.addVar("Data.Toltec.Qs", netCDF::ncInt,
                                      detector_dimensions);
        const int detector_values[6] = {};
        in_phase.putVar(detector_values);
        quadrature.putVar(detector_values);

        const std::vector<netCDF::NcDim> timestamp_dimensions{
            time, timestamp_component};
        auto timestamps = file.addVar("Data.Toltec.Ts", netCDF::ncInt,
                                      timestamp_dimensions);
        timestamps.putAtt(
            "long_name", std::string(nc_contract::legacy_toltec_ts_long_name));
        const int timestamp_values[18] = {};
        timestamps.putVar(timestamp_values);

        const std::vector<netCDF::NcDim> wrong_axis_dimensions{
            other_time, timestamp_component};
        auto wrong_axis = file.addVar("wrong_axis", netCDF::ncInt,
                                      wrong_axis_dimensions);
        wrong_axis.putAtt(
            "long_name", std::string(nc_contract::legacy_toltec_ts_long_name));
        wrong_axis.putVar(timestamp_values);

        auto wrong_fingerprint = file.addVar(
            "wrong_fingerprint", netCDF::ncInt, timestamp_dimensions);
        wrong_fingerprint.putAtt("long_name", "alternative timing schema");
        wrong_fingerprint.putVar(timestamp_values);

        auto wrong_q_type = file.addVar(
            "wrong_q_type", netCDF::ncDouble, detector_dimensions);
        const double wrong_detector_values[6] = {};
        wrong_q_type.putVar(wrong_detector_values);
        const std::vector<netCDF::NcDim> wrong_detector_dimensions{
            time, other_detector};
        auto wrong_q_detector = file.addVar(
            "wrong_q_detector", netCDF::ncInt,
            wrong_detector_dimensions);
        wrong_q_detector.putVar(detector_values);
        file.close();
    }

    netCDF::NcFile file(temporary.path.string(), netCDF::NcFile::read);
    const auto timestamps = file.getVar("Data.Toltec.Ts");
    const auto in_phase = file.getVar("Data.Toltec.Is");
    const auto quadrature = file.getVar("Data.Toltec.Qs");
    EXPECT_NO_THROW(nc_contract::require_legacy_toltec_timing_schema(
        timestamps, in_phase, quadrature));
    EXPECT_THROW(nc_contract::require_legacy_toltec_timing_schema(
                     file.getVar("wrong_axis"), in_phase, quadrature),
                 DataIOError);
    EXPECT_THROW(nc_contract::require_legacy_toltec_timing_schema(
                     file.getVar("wrong_fingerprint"), in_phase,
                     quadrature),
                 DataIOError);
    EXPECT_THROW(nc_contract::require_legacy_toltec_timing_schema(
                     timestamps, in_phase, file.getVar("wrong_q_type")),
                 DataIOError);
    EXPECT_THROW(nc_contract::require_legacy_toltec_timing_schema(
                     timestamps, in_phase,
                     file.getVar("wrong_q_detector")),
                 DataIOError);
}

Eigen::VectorXd assigned_times(double phase, double cadence,
                               std::initializer_list<int> slots) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(slots.size()));
    Eigen::Index row = 0;
    for (const int slot : slots) {
        result[row++] = phase + static_cast<double>(slot) * cadence;
    }
    return result;
}

TEST(sci_align_packet_slot_contract,
     admits_only_packet_progression_identical_to_assigned_slot_progression) {
    constexpr double phase = 1000.0;
    constexpr double cadence = 0.008192;

    const auto contiguous = assigned_times(phase, cadence, {0, 1, 2});
    const auto no_gap =
        citlali::engine_detail::require_packet_slot_consistency(
            "toltec0", contiguous, {}, phase, cadence);
    EXPECT_EQ(no_gap.gap_event_count, 0u);
    EXPECT_EQ(no_gap.missing_packet_count, 0u);

    EXPECT_THROW(
        citlali::engine_detail::require_packet_slot_consistency(
            "toltec0", contiguous, {{0, 11, 1}}, phase, cadence),
        std::runtime_error);

    const auto one_missing = assigned_times(phase, cadence, {0, 2, 3});
    EXPECT_THROW(
        citlali::engine_detail::require_packet_slot_consistency(
            "toltec0", one_missing, {}, phase, cadence),
        std::runtime_error);

    const auto matched_gap =
        citlali::engine_detail::require_packet_slot_consistency(
            "toltec0", one_missing, {{0, 11, 1}}, phase, cadence);
    EXPECT_EQ(matched_gap.gap_event_count, 1u);
    EXPECT_EQ(matched_gap.missing_packet_count, 1u);
}

TEST(sci_align_telescope_header_snapshot,
     preserves_native_numeric_type_scalar_vector_shape_values_and_units) {
    TemporaryInputContractFile input;
    {
        netCDF::NcFile file(input.path.string(), netCDF::NcFile::replace);
        const auto pair_dimension =
            file.addDim("Header.Source.Ra_xlen", 2);
        auto scalar = file.addVar("Header.Source.CoordSys", netCDF::ncInt);
        const int scalar_value = 7;
        scalar.putVar(&scalar_value);
        auto vector = file.addVar("Header.Source.Ra", netCDF::ncDouble,
                                  pair_dimension);
        vector.putAtt("units", "rad");
        const double vector_values[] = {1.25, -2.5};
        vector.putVar(vector_values);
        file.close();
    }

    std::map<std::string,
             citlali::pipeline::sci_align::TelescopeHeaderSnapshot>
        snapshots;
    {
        netCDF::NcFile file(input.path.string(), netCDF::NcFile::read);
        snapshots.emplace(
            "Header.Source.CoordSys",
            nc_contract::read_numeric_telescope_header(
                nc_contract::require_variable(
                    file, "Header.Source.CoordSys"),
                "Header.Source.CoordSys"));
        snapshots.emplace(
            "Header.Source.Ra",
            nc_contract::read_numeric_telescope_header(
                nc_contract::require_variable(file, "Header.Source.Ra"),
                "Header.Source.Ra"));
        file.close();
    }

    TemporaryInputContractFile output;
    write_netcdf_atomic(output.path.string(), [&](netCDF::NcFile &file) {
        citlali::pipeline::add_telescope_header_vars(file, snapshots);
    });
    {
        netCDF::NcFile file(output.path.string(), netCDF::NcFile::read);
        const auto scalar = file.getVar("Header.Source.CoordSys");
        ASSERT_FALSE(scalar.isNull());
        EXPECT_EQ(scalar.getType().getTypeClass(),
                  netCDF::NcType::nc_INT);
        EXPECT_EQ(scalar.getDimCount(), 0u);
        int scalar_value = 0;
        scalar.getVar(&scalar_value);
        EXPECT_EQ(scalar_value, 7);

        const auto vector = file.getVar("Header.Source.Ra");
        ASSERT_FALSE(vector.isNull());
        EXPECT_EQ(vector.getType().getTypeClass(),
                  netCDF::NcType::nc_DOUBLE);
        ASSERT_EQ(vector.getDimCount(), 1u);
        EXPECT_EQ(vector.getDim(0).getName(), "Header.Source.Ra_xlen");
        EXPECT_EQ(vector.getDim(0).getSize(), 2u);
        std::vector<double> values(2);
        vector.getVar(values.data());
        EXPECT_EQ(values, (std::vector<double>{1.25, -2.5}));
        std::string units;
        vector.getAtt("units").getValues(units);
        EXPECT_EQ(units, "rad");
        file.close();
    }
}

TEST(sci_align_telescope_header_snapshot,
     malformed_required_output_fails_atomically_without_truncation) {
    using citlali::pipeline::sci_align::TelescopeHeaderDimensionSnapshot;
    using citlali::pipeline::sci_align::TelescopeHeaderNumericType;
    using citlali::pipeline::sci_align::TelescopeHeaderSnapshot;

    TelescopeHeaderSnapshot malformed;
    malformed.type = TelescopeHeaderNumericType::int32;
    malformed.dimensions = {
        TelescopeHeaderDimensionSnapshot{"header_vector", 2}};
    malformed.values = std::vector<int>{1};

    TemporaryInputContractFile output;
    const std::map<std::string, TelescopeHeaderSnapshot> snapshots{
        {"Header.Required.Vector", malformed}};
    EXPECT_THROW(
        write_netcdf_atomic(output.path.string(),
                            [&](netCDF::NcFile &file) {
                                citlali::pipeline::
                                    add_telescope_header_vars(file,
                                                              snapshots);
                            }),
        DataIOError);
    EXPECT_FALSE(std::filesystem::exists(output.path));
    EXPECT_FALSE(std::filesystem::exists(output.path.string() + ".tmp"));

    TelescopeHeaderSnapshot lossy;
    lossy.type = TelescopeHeaderNumericType::uint64;
    lossy.values =
        std::vector<unsigned long long>{9007199254740992ULL};
    EXPECT_THROW(
        citlali::pipeline::sci_align::telescope_header_legacy_double_view(
            lossy, "Header.Lossy"),
        std::invalid_argument);
}

TEST(sci_align_output_contract,
     atomic_replacement_preserves_the_last_complete_artifact) {
    TemporaryInputContractFile output;
    {
        netCDF::NcFile file(output.path.string(), netCDF::NcFile::replace);
        auto sentinel = file.addVar("sentinel", netCDF::ncInt);
        const int value = 17;
        sentinel.putVar(&value);
        file.close();
    }

    write_netcdf_atomic(output.path.string(), [&](netCDF::NcFile &file) {
        auto replacement = file.addVar("replacement", netCDF::ncInt);
        const int value = 23;
        replacement.putVar(&value);
    });
    {
        netCDF::NcFile file(output.path.string(), netCDF::NcFile::read);
        EXPECT_TRUE(file.getVar("sentinel").isNull());
        const auto replacement = file.getVar("replacement");
        ASSERT_FALSE(replacement.isNull());
        int value = 0;
        replacement.getVar(&value);
        EXPECT_EQ(value, 23);
        file.close();
    }

    EXPECT_THROW(
        write_netcdf_atomic(output.path.string(),
                            [&](netCDF::NcFile &file) {
                                file.addVar("incomplete", netCDF::ncInt);
                                throw DataIOError{"injected writer failure"};
                            }),
        DataIOError);
    EXPECT_FALSE(std::filesystem::exists(output.path.string() + ".tmp"));
    {
        netCDF::NcFile file(output.path.string(), netCDF::NcFile::read);
        const auto replacement = file.getVar("replacement");
        ASSERT_FALSE(replacement.isNull());
        EXPECT_TRUE(file.getVar("incomplete").isNull());
        int value = 0;
        replacement.getVar(&value);
        EXPECT_EQ(value, 23);
        file.close();
    }
}

}  // namespace
