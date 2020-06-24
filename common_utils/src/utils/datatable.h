#pragma once

#include "ecsv.h"
#include "enum/meta_enum.h"
#include "logging.h"
#include <Eigen/Core>
#include <fstream>
#include <iomanip>
#include <regex>
#include <yaml-cpp/yaml.h>

namespace datatable {

using Index = Eigen::Index;

/// @brief Throw when there is an error in parsing data table.
struct ParseError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// @brief Throw when there is an error when dump data table.
struct DumpError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

#ifdef DOXYGEN
/**
 * @enum Format
 * @brief The format of data table.
 * @var ecsv
 *  Astropy ECSV file.
 * @var ascii
 *  Delim-separated Ascii table.
 * @var memdump
 *  Raw C-style array memory dump.
 */
enum class Format : int { Ascii, Memdump };
#else
META_ENUM(Format, int, ecsv, ascii, memdump);
#endif

/**
 * @brief Data table IO handler.
 */
template <Format format> struct IO {
    /**
     * @brief Parse input stream as the specified format.
     * To be implemented for each format specilization.
     * @see IO<Format::Ascii>::parse, IO<Format::Memdump>::parse
     * @tparam Scalar The numeric type of data table.
     * @tparam IStream The input stream type, e.g., std::ifstream.
     */
    template <typename Scalar, typename IStream> static void parse(IStream) {}
};

/**
 * @brief Ascii table IO.
 */
template <> struct IO<Format::ascii> {

    /**
     * @brief Dump as ascii table.
     * @param os_ Output stream to dump the data to
     * @param data The data to be dumped
     * @param colnames The column names to use.
     * @param usecols The indexes of columns to include in the output.
     *  Python-style negative indexes are supported.
     * @param delim Delimiter characters. Default is space.
     */
    template <typename OStream, typename Derived>
    static decltype(auto)
    dump(OStream &os_, const Eigen::EigenBase<Derived> &data,
         const std::vector<std::string> &colnames = {},
         const std::vector<int> &usecols = {}, char delim = ' ',
         std::optional<char> comment = '#') {
        using Scalar = typename Derived::Scalar;
        std::stringstream os;
        os << std::setprecision(std::numeric_limits<Scalar>::digits10 + 1);
        char newline = '\n';
        auto nrows = data.rows();
        auto ncols = data.cols();
        auto ncols_use = ncols;
        SPDLOG_TRACE("dump as ascii, nrows={}, ncols={}, usecols={} "
                     "delim=\"{}\" comment=\"{}\"",
                     nrows, ncols, usecols, delim);
        // get the usecols
        if (!usecols.empty()) {
            for (auto i : usecols) {
                if ((i < -ncols) || (i >= ncols))
                    throw ParseError(fmt::format(
                        "invalid column index {} for table of ncols={}", i,
                        ncols));
            }
            ncols_use = usecols.size();
            SPDLOG_TRACE("using {} cols out of {}", ncols_use, ncols);
        }
        std::string row_prefix{comment.has_value() ? " " : ""};
        // write header
        if (!colnames.empty()) {
            if (colnames.size() != ncols_use) {
                throw DumpError(fmt::format(
                    "number of colnames {} does not match number of columns {}",
                    colnames.size(), ncols_use));
            }
            if (comment.has_value()) {
                os << comment.value() << " ";
            }
            for (std::size_t j = 0; j < colnames.size(); ++j) {
                if (j == 0) {
                    os << row_prefix;
                } else {
                    os << delim;
                }
                os << colnames[j];
            }
            os << newline;
        }
        // write body
        for (Index i = 0; i < nrows; ++i) {
            if (usecols.empty()) {
                for (std::size_t j = 0; j < ncols; ++j) {
                    if (j == 0) {
                        os << row_prefix;
                    } else {
                        os << delim;
                    }
                    os << data.derived().coeff(i, j);
                }
            } else {
                for (std::size_t j = 0; j < usecols.size(); ++j) {
                    auto v = usecols[j];
                    if (v < 0) {
                        v += ncols;
                    }
                    if (j == 0) {
                        os << row_prefix;
                    } else {
                        os << delim;
                    }
                    os << data.derived().coeff(i, v);
                }
            }
            os << newline;
        }
        os_ << os.str();
        return os_;
    }

    /**
     * @brief Parse as Ascii table.
     * @param is Input stream to be parsed.
     * @param header Column names.
     * @param meta Additional commented lines.
     * @param usecols The indexes of columns to include in the result.
     *  Python-style negative indexes are supported.
     * @param delim Delimiter characters. Default is space or tab.
     */
    template <typename Scalar, typename IStream>
    static decltype(auto)
    parse(IStream &is, std::vector<std::string> *header = nullptr,
          std::vector<std::vector<std::string>> *meta = nullptr,
          const std::vector<int> &usecols = {},
          const std::string &delim = " \t") {
        SPDLOG_TRACE("parse as ascii, usecols={} delim=\"{}\"", usecols, delim);
        // is.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        std::string line;
        std::string strnum;
        std::vector<std::vector<Scalar>> data;
        std::vector<std::vector<std::string>> nondata;
        // parse line by line
        while (std::getline(is, line)) {
            std::vector<Scalar> buf_data;
            std::vector<std::string> buf_nondata;
            bool is_data = true;
            for (auto i = line.begin(); i != line.end(); ++i) {
                if (!isascii(*i)) {
                    throw ParseError("not an ascii file");
                }
                // If i is not a delim, then append it to strnum
                if (delim.find(*i) == std::string::npos) {
                    strnum += *i;
                    // continue to next char unless this is the last one
                    if (i + 1 != line.end())
                        continue;
                }
                // if strnum is still empty, it means the previous char is also
                // a delim (several delims appear together). Ignore this char.
                if (strnum.empty())
                    continue;
                // If we reach here, we got something.
                // store a backup in the header in case it is not a number
                buf_nondata.push_back(strnum);
                // try convert it to number.
                if (is_data) {
                    std::istringstream ss(strnum);
                    Scalar number;
                    ss >> number;
                    if (ss.fail()) {
                        // not a number, treat as header
                        is_data = false;
                    }
                    if (is_data) {
                        buf_data.push_back(number);
                    }
                }
                strnum.clear();
            }
            if (is_data) {
                data.emplace_back(buf_data);
            } else {
                nondata.emplace_back(buf_nondata);
            }
        }
        // handle nondata
        if (header != nullptr) {
            *header = nondata[0];
            if (header[0][0] == "#") {
                header->erase(header->begin());
            }
        }
        if (meta != nullptr) {
            *meta = nondata;
        }
        // convert to Eigen matrix
        auto nrows = static_cast<Index>(data.size());
        auto ncols = static_cast<Index>(data[0].size());
        auto ncols_use = ncols;
        SPDLOG_TRACE("shape of table ({}, {})", nrows, ncols);
        // get the usecols
        if (!usecols.empty()) {
            for (auto i : usecols) {
                if ((i < -ncols) || (i >= ncols))
                    throw ParseError(fmt::format(
                        "invalid column index {} for table of ncols={}", i,
                        ncols));
            }
            ncols_use = meta::size_cast<Index>(usecols.size());
            SPDLOG_TRACE("using {} cols out of {}", ncols_use, ncols);
        }
        using Eigen::Dynamic;
        using Eigen::Map;
        using Eigen::Matrix;
        Matrix<Scalar, Dynamic, Dynamic> ret(nrows, ncols_use);
        for (Index i = 0; i < nrows; ++i) {
            if (usecols.empty()) {
                ret.row(i) =
                    Map<Matrix<Scalar, Dynamic, 1>>(&data[i][0], ncols_use);
            } else {
                for (std::size_t j = 0; j < usecols.size(); ++j) {
                    auto v = usecols[j];
                    if (v < 0)
                        v += ncols;
                    // SPDLOG_TRACE("get table data {} {} to return {} {}", i,
                    // v, i, j);
                    ret(i, j) = data[i][v];
                }
            }
            data[i].clear();
        }
        return ret;
    }
};

/**
 * @brief ECSV table IO.
 */
template <> struct IO<Format::ecsv> {
    /**
     * @brief Dump as EVSV table.
     * @param os_ Output stream to dump the data to
     * @param data The data to be dumped
     * @param colnames The column names to use.
     * @param usecols The indexes of columns to include in the output.
     *  Python-style negative indexes are supported.
     */
    template <typename OStream, typename Derived>
    static decltype(auto)
    dump(OStream &os_, const Eigen::EigenBase<Derived> &data,
         const std::vector<std::string> &colnames = {},
         const std::vector<int> &usecols = {}, YAML::Node meta = {}) {
        std::stringstream os;
        using Scalar = typename Derived::Scalar;
        /*
        re2::RE2 re_ecsv_header_prefix("(?m)^");
        auto write_ecsv_header_content = [&](auto &os, auto &&content) {
            // search and replace all newline with the header prefix
            std::string s(FWD(content));
            re2::RE2::GlobalReplace(&s, re_ecsv_header_prefix,
                                    ecsv::spec::ECSV_HEADER_PREFIX);
            os << s << "\n";
        };
        // create yaml header
        auto dtype = ecsv::dtype_str<Scalar>;

        YAML::Node header;
        header.SetStyle(YAML::EmitterStyle::Block);
        for (const auto &c : colnames) {
            YAML::Node n;
            n.SetStyle(YAML::EmitterStyle::Flow);
            n["name"] = c;
            n["datatype"] = dtype;
            header["datatype"].push_back(n);
        }
        if (!meta.IsNull()) {
            header["meta"] = std::move(meta);
        }
        YAML::Emitter ye;
        ye << header;
        SPDLOG_INFO("ecsv header: {}", ye.c_str());
        write_ecsv_header_content(os,
                                  fmt::format("%ECSV {}\n---\n{}",
                                              ecsv::spec::ECSV_VERSION,
                                              ye.c_str()));
        */
        // dump header
        ecsv::dump_header(os, colnames, std::tuple<Scalar>{}, std::move(meta));
        // write data
        IO<Format::ascii>::dump(os, data, colnames, usecols,
                                ecsv::spec::ECSV_DELIM_CHAR, std::nullopt);
        os_ << os.str();
        return os_;
    }

    /**
     * @brief Parse as ECSV table.
     * @param is Input stream to be parsed.
     * @param colnames Vector to capture column names.
     * @param meta YAML node to capture meta data.
     * @param usecols The indexes of columns to include in the result.
     *  Python-style negative indexes are supported.
     */
    template <typename Scalar, typename IStream>
    static decltype(auto)
    parse(IStream &is, std::vector<std::string> *colnames = nullptr,
          YAML::Node *meta = nullptr, const std::vector<int> &usecols = {}) {
        SPDLOG_TRACE("parse as ECSV, usecols={}", usecols);
        // is.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        std::vector<std::string> lines_read;
        // this will read the in stream and parse it until the end of the header
        // block
        const auto &[colnames_, meta_] = [&]() {
            try {
                auto [colnames_, dtypes, meta_] =
                    ecsv::parse_header(is, &lines_read);
                // parse the data section
                if (!ecsv::check_uniform_dtype<Scalar>(dtypes)) {
                    throw ParseError(fmt::format(
                        "the current implementation does not support "
                        "non-uniform datatype in header {dtypes}",
                        dtypes));
                }
                SPDLOG_TRACE("parsed ecsv "
                             "header:\ncolnames:\n{}\ndtypes:\n{}\nmeta:\n{}",
                             colnames_, dtypes, meta_);
                return std::tuple{std::move(colnames_), std::move(meta_)};
            } catch (ecsv::ParseError) {
                throw ParseError(fmt::format(
                    "unable to parse as ECSV lines: {}", lines_read));
            }
        }();
        // proceed with data of type Scalar
        std::string line;
        std::string strnum; // hold the number
        std::vector<std::vector<Scalar>> data;
        while (std::getline(is, line)) {
            std::vector<Scalar> buf_data;
            for (auto it = line.begin(); it != line.end(); ++it) {
                if (!isascii(*it)) {
                    throw ParseError("not an ASCII file");
                }
                // If i is not a delim, then append it to strnum
                if (ecsv::spec::ECSV_DELIM_CHAR != *it) {
                    strnum += *it;
                    // continue to next char unless this is the last one
                    if (it + 1 != line.end())
                        continue;
                }
                // if strnum is still empty, it means the previous char is also
                // a delim (several delims appear together). Ignore this char.
                if (strnum.empty()) {
                    continue;
                }
                // If we reach here, we got something.
                // try convert it to number.
                std::istringstream ss(strnum);
                Scalar number;
                ss >> number;
                if (ss.fail()) {
                    // not a number, treat as header
                    throw ParseError("wrong data type found in data.");
                }
                buf_data.push_back(number);
                strnum.clear();
            }
            data.emplace_back(buf_data);
        }
        if (colnames != nullptr) {
            *colnames = colnames_;
        }
        if (meta != nullptr) {
            *meta = meta_;
        }
        // convert to Eigen matrix
        auto nrows = static_cast<Index>(data.size());
        auto ncols = static_cast<Index>(data[0].size());
        auto ncols_use = ncols;
        SPDLOG_TRACE("shape of table ({}, {})", nrows, ncols);
        // get the usecols
        if (!usecols.empty()) {
            for (auto i : usecols) {
                if ((i < -ncols) || (i >= ncols))
                    throw ParseError(fmt::format(
                        "invalid column index {} for table of ncols={}", i,
                        ncols));
            }
            ncols_use = meta::size_cast<Index>(usecols.size());
            SPDLOG_TRACE("using {} cols out of {}", ncols_use, ncols);
        }
        using Eigen::Dynamic;
        using Eigen::Map;
        using Eigen::Matrix;
        Matrix<Scalar, Dynamic, Dynamic> ret(nrows, ncols_use);
        for (Index i = 0; i < nrows; ++i) {
            if (usecols.empty()) {
                ret.row(i) =
                    Map<Matrix<Scalar, Dynamic, 1>>(&data[i][0], ncols_use);
            } else {
                for (std::size_t j = 0; j < usecols.size(); ++j) {
                    auto v = usecols[j];
                    if (v < 0)
                        v += ncols;
                    // SPDLOG_TRACE("get table data {} {} to return {} {}", i,
                    // v, i, j);
                    ret(i, j) = data[i][v];
                }
            }
            data[i].clear();
        }
        return ret;
    };
}; // namespace datatable

/**
 * @brief C-style array memory dump IO.
 */
template <> struct IO<Format::memdump> {

    /**
     * @brief Prase as C-style array memory dump.
     * @param is Input stream to be parsed.
     * @param nrows The number of rows of the table.
     * Default is Eigen::Dynamic, in which case the number of rows is
     * determined from the size of the data and \p ncols.
     * @param ncols The number of columns of the table.
     * Default is Eigen::Dynamic, in which case, the number of columns is
     * determined from the size of the data and \p nrows if \p nrows is not
     * Eigen::Dynamic, or 1 if nrows is Eigen::Dynamic.
     * @param order The storage order of the memdump.
     * Default is Eigen::ColMajor.
     */
    template <typename Scalar, typename IStream>
    static decltype(auto) parse(IStream &is, Index nrows = Eigen::Dynamic,
                                Index ncols = Eigen::Dynamic,
                                Eigen::StorageOptions order = Eigen::ColMajor) {
        using Eigen::Dynamic;
        using Eigen::Matrix;
        using Eigen::RowMajor;
        // throw everything
        is.exceptions(std::ios_base::failbit | std::ios_base::badbit |
                      std::ios_base::eofbit);
        // get file size
        is.seekg(0, std::ios_base::end);
        auto filesize = is.tellg();
        auto size = filesize / sizeof(Scalar);
        SPDLOG_TRACE("memdump size={} nelem={}", filesize, size);
        // validate based on nrows and ncols
        if (ncols == Dynamic) {
            ncols = 1;
        } else if (size % ncols != 0) {
            throw ParseError(fmt::format(
                "memdump size {} inconsistent with ncols={}", size, ncols));
        }
        if (nrows == Dynamic) {
            nrows = size / ncols;
        } else if (nrows * ncols != size) {
            throw ParseError(fmt::format(
                "memdump size {} inconsistent with nrows={} ncols={}", size,
                nrows, ncols));
        }
        SPDLOG_TRACE("memdump shape ({}, {})", nrows, ncols);
        // this is colmajor
        Matrix<Scalar, Dynamic, Dynamic> ret{nrows, ncols};
        auto get = [&](auto *data) {
            is.seekg(0, std::ios_base::beg);
            is.read(reinterpret_cast<char *>(data), filesize);
        };
        if (order & Eigen::RowMajor) {
            Matrix<Scalar, Dynamic, Dynamic, RowMajor> tmp{nrows, ncols};
            get(tmp.data());
            ret = tmp;
        } else {
            get(ret.data());
        }
        SPDLOG_TRACE("memdump data{:r10c10}", ret);
        return ret;
    }
};

/**
 * @brief Read data table file.
 * @tparam Scalar The numeric type of the data.
 * @tparam Format The expected file format from \p Format.
 * @param filepath The path to the table file.
 * @param args The arguments forwarded to call \ref IO<format>::parse().
 */
template <typename Scalar, Format format, typename... Args>
auto read(const std::string &filepath, Args &&... args) {
    SPDLOG_TRACE("read data from {}", filepath);
    std::ifstream fo;
    fo.open(filepath, std::ios_base::binary);
    return IO<format>::template parse<Scalar>(
        fo, std::forward<decltype(args)>(args)...);
}

/**
 * @brief Write table data to file.
 * @tparam Format The file format from Format.
 * @param filepath The path of the output file.
 * @param args The arguments forwarded to call IO<format>::dump().
 */
template <Format format, typename... Args>
void write(const std::string &filepath, Args &&... args) {
    SPDLOG_TRACE("write data to {}", filepath);
    std::ofstream fo;
    fo.open(filepath, std::ios_base::binary);
    IO<format>::dump(fo, std::forward<decltype(args)>(args)...);
}

} // namespace datatable
