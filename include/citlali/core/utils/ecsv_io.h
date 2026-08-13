#pragma once

#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/atomic_yaml_output.h>

#include <tula/algorithm/ei_stats.h>
#include <tula/algorithm/index.h>
#include <tula/container.h>
#include <tula/datatable.h>
#include <tula/filename.h>

#include <tula/ecsv/core.h>
#include <csv_parser/parser.hpp>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <tula/ecsv/table.h>
#include <tula/formatter/container.h>
#include <tula/formatter/matrix.h>
#include <yaml-cpp/node/emit.h>

inline double parse_uniform_float64_ecsv_token(const std::string &token) {
    if (token.empty()) {
        throw datatable::ParseError("empty float64 field in ECSV data");
    }
    errno = 0;
    char *end = nullptr;
    const double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str() ||
        end != token.c_str() + token.size() || errno == ERANGE) {
        throw datatable::ParseError(
            "malformed float64 field in ECSV data: " + token);
    }
    return value;
}

inline auto read_uniform_float64_ecsv(const std::string &filepath) {
    using namespace tula::ecsv;

    try {
        std::ifstream stream(filepath);
        if (!stream) {
            throw datatable::ParseError(
                "unable to open ECSV file " + filepath);
        }
        auto ecsv_header = ECSVHeader::read(stream);
        if (!check_uniform_dtype<double>(ecsv_header.datatypes())) {
            throw datatable::ParseError(
                "ECSV table is not uniformly float64");
        }
        auto header = tula::container_utils::to_stdvec(
            ecsv_header.colnames());
        if (header.empty()) {
            throw datatable::ParseError("ECSV table has no columns");
        }

        std::vector<std::vector<double>> rows;
        auto parser = aria::csv::CsvParser(stream).delimiter(
            ecsv_header.delimiter());
        for (const auto &row : parser) {
            if (row.size() != header.size()) {
                throw datatable::ParseError(
                    "ECSV data row width does not match its header");
            }
            std::vector<double> values;
            values.reserve(row.size());
            for (const auto &field : row) {
                values.push_back(parse_uniform_float64_ecsv_token(field));
            }
            rows.push_back(std::move(values));
        }

        Eigen::MatrixXd table(
            static_cast<Eigen::Index>(rows.size()),
            static_cast<Eigen::Index>(header.size()));
        for (Eigen::Index row = 0; row < table.rows(); ++row) {
            for (Eigen::Index column = 0; column < table.cols(); ++column) {
                table(row, column) = rows[static_cast<std::size_t>(row)]
                    [static_cast<std::size_t>(column)];
            }
        }
        YAML::Node meta = ecsv_header.meta();
        return std::tuple{table, header, meta};
    }
    catch (const datatable::ParseError &) {
        throw;
    }
    catch (const std::exception &error) {
        throw datatable::ParseError(
            "unable to parse uniform float64 ECSV " + filepath + ": " +
            error.what());
    }
}

// create Eigen::Matrix from ecsv file
inline auto to_matrix_from_ecsv(std::string filepath) {
    namespace fs = std::filesystem;

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    std::vector<std::string> header;
    Eigen::MatrixXd table;

    YAML::Node meta_;

    try {
        std::tie(table, header, meta_) =
            read_uniform_float64_ecsv(filepath);

    } catch (datatable::ParseError &e) {
        logger->warn("unable to read apt table file as ECSV {}: {}", filepath,
                    e.what());
        try {
            table = datatable::read<double, datatable::Format::ascii>(filepath,
                                                                      &header);
        } catch (datatable::ParseError &e) {
            logger->warn("unable to read apt table file as ASCII {}: {}",
                        filepath, e.what());
            throw e;
        }
    }
    return std::tuple {table, header, meta_};
}

template <typename Derived, class Validator>
inline void to_ecsv_from_matrix_validated(
    std::string filepath, Eigen::DenseBase<Derived> &table,
    std::vector<std::string> header, YAML::Node meta,
    Validator &&validator) {
    namespace fs = std::filesystem;
    const fs::path final_path(filepath + ".ecsv");
    const fs::path temp_path(final_path.string() + ".tmp");
    std::error_code ec;
    fs::remove(temp_path, ec);
    try {
        datatable::write<datatable::Format::ecsv>(
            temp_path.string(), table, header, std::vector<int>{}, meta);
        citlali::pipeline::atomic_output::synchronize_file(temp_path);

        auto [reopened_table, reopened_header, reopened_meta] =
            read_uniform_float64_ecsv(temp_path.string());
        if (reopened_table.rows() != table.rows() ||
            reopened_table.cols() != table.cols() ||
            reopened_header != header) {
            throw std::runtime_error(
                "reopened ECSV structure does not match the staged table");
        }
        for (Eigen::Index row = 0; row < table.rows(); ++row) {
            for (Eigen::Index column = 0; column < table.cols(); ++column) {
                const double expected =
                    static_cast<double>(table.derived()(row, column));
                const double actual = reopened_table(row, column);
                if (expected != actual &&
                    !(std::isnan(expected) && std::isnan(actual))) {
                    throw std::runtime_error(
                        "reopened ECSV values do not match the staged table");
                }
            }
        }
        if (!citlali::pipeline::atomic_output::yaml_nodes_equivalent(
                meta, reopened_meta)) {
            throw std::runtime_error(
                "reopened ECSV metadata does not match the staged table");
        }
        std::invoke(std::forward<Validator>(validator),
                    reopened_table, reopened_header, reopened_meta);
        citlali::pipeline::atomic_output::replace_atomically(
            temp_path, final_path);
    } catch (const std::exception &e) {
        ec.clear();
        fs::remove(temp_path, ec);
        throw citlali::error::output(
            "failed to write required ECSV output " + final_path.string() +
            ": " + e.what());
    } catch (...) {
        ec.clear();
        fs::remove(temp_path, ec);
        throw citlali::error::output(
            "failed to write required ECSV output " + final_path.string());
    }
}

// create ecsv file from Eigen::Matrix
template <typename Derived>
inline void to_ecsv_from_matrix(
    std::string filepath, Eigen::DenseBase<Derived> &table,
    std::vector<std::string> header, YAML::Node meta) {
    to_ecsv_from_matrix_validated(
        std::move(filepath), table, std::move(header), std::move(meta),
        [](const Eigen::MatrixXd &, const std::vector<std::string> &,
           const YAML::Node &) {});
}

inline auto to_map_from_ecsv_mixted_type(std::string filepath) {
    using namespace tula::ecsv;

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // vector to hold header
    std::vector<std::string> header;

    // std map for holding data
    std::map<std::string, Eigen::VectorXd> table;

    // hold str meta
    std::map<std::string, std::string> map_with_strs;

    // to hold meta data
    YAML::Node meta{};

    std::ifstream fo(filepath);
    try {
        // read in header
        auto hdr = ECSVHeader::read(fo);
        // create table
        auto tbl = ECSVTable(hdr);
        // parse the contents
        auto parser = aria::csv::CsvParser(fo).delimiter(tbl.header().delimiter());
        // load rows
        tbl.load_rows(parser);

        // get header colnames
        for (Eigen::Index i=0; i<tbl.header().colnames().size(); i++) {
            header.push_back(tbl.header().colnames()[i]);
        }

        const auto map_with_bools =
            meta_to_map<std::string, bool>(hdr.meta(), &meta);

        map_with_strs =
            meta_to_map<std::string, std::string>(meta, &meta);

        // get ints
        auto int_colnames = tbl.array_data<int>().colnames();
        for (auto & col : int_colnames) {
            table[col] = tbl.col<int>(col).template cast<double> ();
        }

        // get int16
        auto int16_colnames = tbl.array_data<int16_t>().colnames();
        for (auto & col : int16_colnames) {
            table[col] = tbl.col<int16_t>(col).template cast<double> ();
        }

        // get int64
        auto int64_colnames = tbl.array_data<int64_t>().colnames();
        for (auto & col : int64_colnames) {
            table[col] = tbl.col<int64_t>(col).template cast<double> ();
        }

        // get bools
        auto bool_colnames = tbl.array_data<bool>().colnames();
        for (auto & col : bool_colnames) {
            table[col] = tbl.col<bool>(col).template cast<double> ();
        }

        // get floats
        auto float_colnames = tbl.array_data<float>().colnames();
        for (auto & col : float_colnames) {
            table[col] = tbl.col<float>(col).template cast<double> ();
        }

        // get doubles
        auto dbl_colnames = tbl.array_data<double>().colnames();
        for (auto & col : dbl_colnames) {
            table[col] = tbl.col<double>(col);
        }
    }
    catch(const std::exception &error) {
        throw citlali::error::io(
            "cannot open input table " + filepath + ": " + error.what());
    }
    catch(...) {
        throw citlali::error::io("cannot open input table " + filepath);
    }

    // return map and header
    return std::tuple {table, header, map_with_strs};
}
