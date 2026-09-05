#pragma once

#include <citlali/core/fruit/response_intervention.h>
#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>

namespace citlali::fruit {

using ResponseCountMatrix = Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic>;

struct ResponseAccumulators {
    Eigen::MatrixXd n, c, q, abs_n, abs_c;
    ResponseCountMatrix count, unique;
    ResponseAccumulators() = default;
    ResponseAccumulators(Eigen::Index rows, Eigen::Index cols);
};

struct ResponseDeletion {
    ResponseScore score;
    ResponseAccumulators target;
    Eigen::MatrixXd difference, leverage, contrast, identity_residual;
    Eigen::MatrixXd n_error, c_error, q_error, signal_error, coefficient_error;
    ResponseCountMatrix conditioned, support, footprint;
};

// Observation/iteration-owned exact final-PTC occurrence ledger. It never runs
// RTC/PTC, chooses a UID, or reads a future/control/injection result. Creation
// and output are cold boundaries. append() is buffered fixed-size binary I/O.
// Only the serial raw-observation JINC science pass may append occurrences.
// Binary records are eight native int64 values followed by five binary64
// values. Endianness and field identities are recorded in the NetCDF receipt.
class ResponseOccurrenceLedger {
public:
    using KernelBank = std::map<int, std::vector<Eigen::MatrixXd>>;
    static constexpr std::uint64_t spool_limit = std::uint64_t{32} << 30;
    static constexpr std::size_t record_bytes = 8 * sizeof(std::int64_t) + 5 * sizeof(double);

    ResponseOccurrenceLedger(std::filesystem::path path, std::string observation,
                             int iteration, Eigen::Index rows, Eigen::Index cols,
                             std::vector<int> map_arrays, KernelBank kernels,
                             KernelBank squared_kernels);
    ResponseOccurrenceLedger(const ResponseOccurrenceLedger &) = delete;
    ResponseOccurrenceLedger &operator=(const ResponseOccurrenceLedger &) = delete;

    void begin_scan(int scan);
    void register_detector(int uid, int array);
    void append(int scan, int uid, int array, int row, int col, int kernel,
                std::int64_t sample, double n_scale, double c_scale,
                double q_scale, double processed_signal, double multiplier);
    void capture_totals(const std::vector<Eigen::MatrixXd> &n,
                        const std::vector<Eigen::MatrixXd> &c,
                        const std::vector<Eigen::MatrixXd> &q);
    void finish(ResponseInterventionState &state,
                std::vector<ResponseCandidate> census,
                const std::vector<Eigen::MatrixXd> &signal,
                const std::vector<Eigen::MatrixXd> &weight,
                const std::vector<Eigen::MatrixXd> &formal_weight,
                double coverage_cut);
    void write_receipt(const std::filesystem::path &path,
                       const ResponseInterventionState &state,
                       double pixel_size_rad, const std::string &unit) const;

    const auto &totals() const { return totals_; }
    const std::string &observation() const { return observation_; }
    const auto &deletions() const { return deletions_; }
    const auto &joint_deletions() const { return joint_; }
    std::uint64_t occurrence_count() const { return occurrences_; }
    std::uint64_t spool_bytes() const { return bytes_; }

private:
    struct Record {
        // kind (0 scan boundary, 1 occurrence), scan, UID, array, row, col,
        // kernel index, zero-based final-PTC sample index.
        std::array<std::int64_t, 8> identity{};
        // numerator scale, signed-coefficient scale, quadratic scale,
        // processed mJy/beam value, independently recorded map multiplier.
        std::array<double, 5> values{};
    };
    std::filesystem::path path_;
    std::string observation_;
    int iteration_;
    Eigen::Index rows_, cols_;
    std::vector<int> map_arrays_;
    KernelBank kernels_, squared_kernels_;
    std::map<int, std::map<int, std::size_t>> uid_slots_;
    std::set<int> scans_;
    int current_scan_ = -1;
    std::unique_ptr<char[]> io_buffer_;
    std::ofstream spool_;
    std::uint64_t bytes_ = 0, occurrences_ = 0;
    bool captured_ = false, finished_ = false;
    double coverage_cut_ = 0.0;
    std::map<int, ResponseAccumulators> totals_;
    std::map<ResponseKey, ResponseDeletion> deletions_;
    std::map<int, ResponseDeletion> joint_;
    std::map<int, Eigen::MatrixXd> signal_, weight_, formal_weight_;
    std::map<int, ResponseCountMatrix> science_support_;
    void write_record(const Record &record);
    void reconstruct(const std::vector<ResponseCandidate> &census);
    ResponseDeletion evaluate(int array, ResponseAccumulators target,
                              double coverage_cut, double map_scale) const;
};

}  // namespace citlali::fruit
