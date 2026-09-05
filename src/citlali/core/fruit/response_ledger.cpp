#include <citlali/core/fruit/response_ledger.h>
#include <citlali/core/utils/netcdf_io.h>

#include <netcdf>
#include <bit>
#include <cstring>
#include <limits>
#include <numeric>

namespace citlali::fruit {
namespace {
constexpr double u = 0x1p-53;
constexpr double finalization_safety = 16.0;
constexpr double identity_safety = 64.0;

double gamma(double count) {
    const double product = std::max(1.0, count) * u;
    if (product >= 1.0) throw std::runtime_error("EL-F12 unbounded rounding error");
    return product / (1.0 - product);
}

double difference_bound(double total, double target, double total_abs,
                        double target_abs, double hits, double target_hits) {
    return gamma(hits) * total_abs + gamma(target_hits) * target_abs +
           gamma(std::max(hits - target_hits, 0.0)) * std::max(total_abs - target_abs, 0.0) +
           u * (std::abs(total) + std::abs(target));
}

double median(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("EL-F12 empty scientific map support");
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

double threshold(const Eigen::MatrixXd &weight, double cut) {
    std::vector<double> values;
    for (Eigen::Index i = 0; i < weight.size(); ++i)
        if (std::isfinite(weight(i)) && weight(i) > 0) values.push_back(weight(i));
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto index = (static_cast<std::size_t>(std::floor(0.75 * values.size())) + values.size()) / 2;
    return values.at(index) * cut;
}

struct Finalized {
    Eigen::MatrixXd signal, weight;
    ResponseCountMatrix support;
};

Finalized finalize(const ResponseAccumulators &value, double cut) {
    Finalized result;
    result.weight = Eigen::MatrixXd::Zero(value.n.rows(), value.n.cols());
    result.signal = result.weight;
    result.support = ResponseCountMatrix::Zero(value.n.rows(), value.n.cols());
    for (Eigen::Index i = 0; i < value.n.size(); ++i) {
        if (!std::isfinite(value.n(i)) || !std::isfinite(value.c(i)) || !std::isfinite(value.q(i)))
            throw std::runtime_error("EL-F12 nonfinite accumulator");
        if (std::abs(value.c(i)) > 1e-8 && value.q(i) > 0)
            result.weight(i) = value.c(i) * value.c(i) / std::max(value.q(i), 1e-30);
        if (!std::isfinite(result.weight(i))) throw std::runtime_error("EL-F12 coefficient overflow");
    }
    const double limit = threshold(result.weight, cut / 10.0);
    for (Eigen::Index i = 0; i < result.weight.size(); ++i) {
        const bool valid = result.weight(i) > 0.0 && result.weight(i) >= limit;
        result.support(i) = valid;
        if (valid) result.signal(i) = value.n(i) / value.c(i);
        else result.weight(i) = 0.0;
    }
    return result;
}

template <class Matrix>
void plane(netCDF::NcGroup &group, const std::string &name, const Matrix &matrix,
           const netCDF::NcDim &rows, const netCDF::NcDim &cols) {
    constexpr bool integral = std::is_integral_v<typename Matrix::Scalar>;
    const netCDF::NcType &type = integral ? static_cast<const netCDF::NcType &>(netCDF::ncInt64)
                                         : static_cast<const netCDF::NcType &>(netCDF::ncDouble);
    auto variable = group.addVar(name, type, {rows, cols});
    set_netcdf_chunking_and_compression(variable,
        {std::min<std::size_t>(128, rows.getSize()), std::min<std::size_t>(128, cols.getSize())});
    std::vector<typename Matrix::Scalar> values;
    values.reserve(static_cast<std::size_t>(matrix.size()));
    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
        for (Eigen::Index col = 0; col < matrix.cols(); ++col) values.push_back(matrix(row, col));
    variable.putVar(values.data());
}

void accumulator_planes(netCDF::NcGroup &group, const ResponseAccumulators &value,
                        const netCDF::NcDim &rows, const netCDF::NcDim &cols) {
    plane(group, "N", value.n, rows, cols);
    plane(group, "C", value.c, rows, cols);
    plane(group, "Q", value.q, rows, cols);
    plane(group, "absolute_N_terms", value.abs_n, rows, cols);
    plane(group, "absolute_C_terms", value.abs_c, rows, cols);
    plane(group, "occurrence_pixel_count", value.count, rows, cols);
    plane(group, "unique_detector_count", value.unique, rows, cols);
}
}  // namespace

ResponseAccumulators::ResponseAccumulators(Eigen::Index rows, Eigen::Index cols) {
    n = Eigen::MatrixXd::Zero(rows, cols);
    c = q = abs_n = abs_c = n;
    count = ResponseCountMatrix::Zero(rows, cols);
    unique = count;
}

ResponseOccurrenceLedger::ResponseOccurrenceLedger(
    std::filesystem::path path, std::string observation, int iteration,
    Eigen::Index rows, Eigen::Index cols, std::vector<int> map_arrays,
    KernelBank kernels, KernelBank squared_kernels)
    : path_(std::move(path)), observation_(std::move(observation)), iteration_(iteration),
      rows_(rows), cols_(cols), map_arrays_(std::move(map_arrays)),
      kernels_(std::move(kernels)), squared_kernels_(std::move(squared_kernels)),
      io_buffer_(std::make_unique<char[]>(1 << 20)) {
    if (observation_.empty() || iteration < 0 || iteration > 6 || rows <= 0 || cols <= 0 ||
        map_arrays_.empty() || map_arrays_.size() > 3 ||
        std::set<int>(map_arrays_.begin(), map_arrays_.end()).size() != map_arrays_.size())
        throw std::runtime_error("EL-F12 invalid ledger identity/shape");
    for (const int array : map_arrays_) {
        if (array < 0 || array > 2 || !kernels_.count(array) || kernels_.at(array).empty() ||
            !squared_kernels_.count(array) || kernels_.at(array).size() != squared_kernels_.at(array).size())
            throw std::runtime_error("EL-F12 missing JINC kernel bank");
        for (std::size_t i = 0; i < kernels_.at(array).size(); ++i) {
            const auto &kernel = kernels_.at(array)[i];
            const auto &square = squared_kernels_.at(array)[i];
            if (kernel.rows() <= 0 || kernel.cols() <= 0 || kernel.rows() % 2 != 1 || kernel.cols() % 2 != 1 ||
                square.rows() != kernel.rows() || square.cols() != kernel.cols() ||
                !kernel.allFinite() || !square.allFinite() || (square.array() < 0).any() ||
                !(square.array() == kernel.array().square()).all())
                throw std::runtime_error("EL-F12 malformed JINC kernel bank");
        }
        totals_.emplace(array, ResponseAccumulators(rows, cols));
    }
    if (std::filesystem::exists(path_)) throw std::runtime_error("EL-F12 refuses to overwrite an occurrence spool");
    spool_.rdbuf()->pubsetbuf(io_buffer_.get(), 1 << 20);
    spool_.open(path_, std::ios::binary | std::ios::out);
    if (!spool_) throw std::runtime_error("EL-F12 cannot create required occurrence spool");
}

void ResponseOccurrenceLedger::write_record(const Record &record) {
    if (finished_ || bytes_ > spool_limit - record_bytes)
        throw std::runtime_error("EL-F12 occurrence spool limit exceeded or already finalized");
    spool_.write(reinterpret_cast<const char *>(record.identity.data()), sizeof(record.identity));
    spool_.write(reinterpret_cast<const char *>(record.values.data()), sizeof(record.values));
    if (!spool_) throw std::runtime_error("EL-F12 required occurrence spool write failed");
    bytes_ += record_bytes;
}

void ResponseOccurrenceLedger::begin_scan(int scan) {
    if (scan < 0 || !scans_.insert(scan).second)
        throw std::runtime_error("EL-F12 duplicate science scan (noise passes cannot append)");
    current_scan_ = scan;
    Record marker;
    marker.identity[1] = scan;
    write_record(marker);
}

void ResponseOccurrenceLedger::register_detector(int uid, int array) {
    if (uid < 0 || !totals_.count(array)) throw std::runtime_error("EL-F12 invalid occurrence UID/array");
    for (const auto &[other_array, slots] : uid_slots_)
        if (other_array != array && slots.count(uid)) throw std::runtime_error("EL-F12 UID changed arrays");
    auto &slots = uid_slots_[array];
    slots.try_emplace(uid, slots.size());
}

void ResponseOccurrenceLedger::append(
    int scan, int uid, int array, int row, int col, int kernel, std::int64_t sample,
    double n_scale, double c_scale, double q_scale, double processed_signal, double multiplier) {
    if (scan != current_scan_ || !uid_slots_.count(array) || !uid_slots_.at(array).count(uid) ||
        row < 0 || col < 0 || row >= rows_ || col >= cols_ || kernel < 0 ||
        static_cast<std::size_t>(kernel) >= kernels_.at(array).size() || sample < 0 ||
        !std::isfinite(n_scale) || !std::isfinite(c_scale) || c_scale <= 0 ||
        !std::isfinite(q_scale) || q_scale <= 0 || !std::isfinite(processed_signal) ||
        (multiplier != 1.0 && multiplier != 0.5))
        throw std::runtime_error("EL-F12 malformed admitted occurrence");
    Record record;
    record.identity = {1, scan, uid, array, row, col, kernel, sample};
    record.values = {n_scale, c_scale, q_scale, processed_signal, multiplier};
    write_record(record);
    ++occurrences_;
}

void ResponseOccurrenceLedger::capture_totals(
    const std::vector<Eigen::MatrixXd> &n, const std::vector<Eigen::MatrixXd> &c,
    const std::vector<Eigen::MatrixXd> &q) {
    if (captured_ || n.size() != map_arrays_.size() || c.size() != n.size() || q.size() != n.size())
        throw std::runtime_error("EL-F12 missing/duplicate total accumulator capture");
    for (std::size_t slot = 0; slot < n.size(); ++slot) {
        for (const auto *value : {&n[slot], &c[slot], &q[slot]})
            if (value->rows() != rows_ || value->cols() != cols_ || !value->allFinite())
                throw std::runtime_error("EL-F12 invalid total accumulator shape/value");
        auto &total = totals_.at(map_arrays_[slot]);
        total.n = n[slot]; total.c = c[slot]; total.q = q[slot];
    }
    captured_ = true;
}

void ResponseOccurrenceLedger::reconstruct(const std::vector<ResponseCandidate> &census) {
    std::map<int, ResponseAccumulators> reconstructed, scan_totals, joint_scan;
    std::map<int, std::vector<std::uint64_t>> unique_bits, joint_bits;
    std::map<int, std::map<int, std::size_t>> joint_slots;
    const auto pixels = static_cast<std::size_t>(rows_ * cols_);
    for (const int array : map_arrays_) {
        reconstructed.emplace(array, ResponseAccumulators(rows_, cols_));
        scan_totals.emplace(array, ResponseAccumulators(rows_, cols_));
        unique_bits[array].resize(pixels * ((uid_slots_[array].size() + 63) / 64));
    }
    for (const auto &candidate : census) {
        if (candidate.disposition != "eligible") continue;
        ResponseDeletion deletion;
        deletion.target = ResponseAccumulators(rows_, cols_);
        if (!deletions_.emplace(candidate.key, std::move(deletion)).second)
            throw std::runtime_error("EL-F12 duplicate eligible ledger key");
        auto &slots = joint_slots[candidate.key.array];
        slots.try_emplace(candidate.key.uid, slots.size());
        if (!joint_.count(candidate.key.array)) {
            joint_[candidate.key.array].target = ResponseAccumulators(rows_, cols_);
            joint_scan.emplace(candidate.key.array, ResponseAccumulators(rows_, cols_));
            joint_bits[candidate.key.array].resize(pixels);
        }
    }
    // Preserve the ordinary two-level accumulation: per-scan planes, then
    // ordered commits into observation totals. Target/union terms retain
    // their original occurrence order, without normalized-map summation.
    auto commit_scan = [&] {
        for (const int array : map_arrays_) {
            auto &full = reconstructed.at(array);
            auto &part = scan_totals.at(array);
            full.n += part.n; full.c += part.c; full.q += part.q;
            part.n.setZero(); part.c.setZero(); part.q.setZero();
        }
        for (auto &[array, part] : joint_scan) {
            auto &full = joint_.at(array).target;
            full.n += part.n; full.c += part.c; full.q += part.q;
            part.n.setZero(); part.c.setZero(); part.q.setZero();
        }
    };
    auto mark_unique = [&](ResponseAccumulators &value, std::vector<std::uint64_t> &bits,
                           std::size_t words, std::size_t slot, Eigen::Index row, Eigen::Index col) {
        const auto pixel = static_cast<std::size_t>(row * cols_ + col);
        auto &word = bits.at(pixel * words + slot / 64);
        const auto bit = std::uint64_t{1} << (slot % 64);
        if (!(word & bit)) { word |= bit; ++value.unique(row, col); }
    };
    std::ifstream input(path_, std::ios::binary);
    if (!input) throw std::runtime_error("EL-F12 cannot read occurrence spool");
    int scan = -1;
    std::uint64_t read_bytes = 0, read_occurrences = 0;
    std::set<int> seen_scans;
    std::set<std::pair<int, int>> seen_detector_blocks;
    int previous_uid = -1;
    std::int64_t previous_sample = -1;
    while (read_bytes < bytes_) {
        Record record;
        input.read(reinterpret_cast<char *>(record.identity.data()), sizeof(record.identity));
        input.read(reinterpret_cast<char *>(record.values.data()), sizeof(record.values));
        if (!input) throw std::runtime_error("EL-F12 truncated occurrence spool");
        read_bytes += record_bytes;
        const auto &id = record.identity;
        if (id[0] == 0) {
            if (scan >= 0) commit_scan();
            scan = static_cast<int>(id[1]);
            if (!seen_scans.insert(scan).second) throw std::runtime_error("EL-F12 repeated spool scan");
            previous_uid = -1; previous_sample = -1;
            continue;
        }
        if (id[0] != 1 || scan < 0 || id[1] != scan)
            throw std::runtime_error("EL-F12 invalid spool sequence");
        const int uid = static_cast<int>(id[2]), array = static_cast<int>(id[3]);
        if (uid != previous_uid) {
            if (!seen_detector_blocks.emplace(scan, uid).second)
                throw std::runtime_error("EL-F12 duplicate/noncontiguous detector occurrence block");
            previous_uid = uid; previous_sample = -1;
        }
        if (id[7] <= previous_sample) throw std::runtime_error("EL-F12 duplicate/unordered sample identity");
        previous_sample = id[7];
        ++read_occurrences;
        const auto &kernel = kernels_.at(array).at(static_cast<std::size_t>(id[6]));
        const auto &square = squared_kernels_.at(array).at(static_cast<std::size_t>(id[6]));
        const int original_row = static_cast<int>(id[4] - (kernel.rows() - 1) / 2);
        const int original_col = static_cast<int>(id[5] - (kernel.cols() - 1) / 2);
        const int lower_row = std::max(0, original_row), lower_col = std::max(0, original_col);
        const int size_rows = std::min<int>(rows_, original_row + kernel.rows()) - lower_row;
        const int size_cols = std::min<int>(cols_, original_col + kernel.cols()) - lower_col;
        const auto block = kernel.block(lower_row - original_row, lower_col - original_col, size_rows, size_cols);
        const auto square_block = square.block(lower_row - original_row, lower_col - original_col, size_rows, size_cols);
        auto &part = scan_totals.at(array);
        part.n.block(lower_row, lower_col, size_rows, size_cols) += (block * record.values[0]).eval();
        part.c.block(lower_row, lower_col, size_rows, size_cols).array() += block.array() * record.values[1];
        part.q.block(lower_row, lower_col, size_rows, size_cols).array() += square_block.array() * record.values[2];
        const ResponseKey key{observation_, array, uid, scan};
        auto found = deletions_.find(key);
        auto &full = reconstructed.at(array);
        for (int r = 0; r < size_rows; ++r) for (int c = 0; c < size_cols; ++c) {
            const int row = lower_row + r, col = lower_col + c;
            const double n = block(r, c) * record.values[0];
            const double coeff = block(r, c) * record.values[1];
            const double q = square_block(r, c) * record.values[2];
            full.abs_n(row, col) += std::abs(n); full.abs_c(row, col) += std::abs(coeff);
            ++full.count(row, col);
            mark_unique(full, unique_bits.at(array), (uid_slots_.at(array).size() + 63) / 64,
                        uid_slots_.at(array).at(uid), row, col);
            if (found == deletions_.end()) continue;
            auto &joint_part = joint_scan.at(array);
            joint_part.n(row, col) += n; joint_part.c(row, col) += coeff; joint_part.q(row, col) += q;
            auto &target = found->second.target;
            target.n(row, col) += n; target.c(row, col) += coeff; target.q(row, col) += q;
            for (auto *target : {&found->second.target, &joint_.at(array).target}) {
                target->abs_n(row, col) += std::abs(n); target->abs_c(row, col) += std::abs(coeff);
                ++target->count(row, col);
            }
            found->second.target.unique(row, col) = 1;
            mark_unique(joint_.at(array).target, joint_bits.at(array), 1,
                        joint_slots.at(array).at(uid), row, col);
        }
    }
    if (scan >= 0) commit_scan();
    if (input.peek() != std::char_traits<char>::eof() || read_occurrences != occurrences_ || seen_scans != scans_)
        throw std::runtime_error("EL-F12 occurrence spool count/scan closure failed");
    for (const int array : map_arrays_) {
        auto &actual = totals_.at(array);
        auto &full = reconstructed.at(array);
        if (!(actual.n.array() == full.n.array()).all() || !(actual.c.array() == full.c.array()).all() ||
            !(actual.q.array() == full.q.array()).all())
            throw std::runtime_error("EL-F12 spool does not exactly reconstruct actual JINC totals");
        actual.abs_n = std::move(full.abs_n); actual.abs_c = std::move(full.abs_c);
        actual.count = std::move(full.count); actual.unique = std::move(full.unique);
    }
}

ResponseDeletion ResponseOccurrenceLedger::evaluate(
    int array, ResponseAccumulators target, double coverage_cut, double map_scale) const {
    ResponseDeletion result;
    result.target = std::move(target);
    const auto &total = totals_.at(array), &part = result.target;
    ResponseAccumulators without(rows_, cols_);
    without.n = total.n - part.n; without.c = total.c - part.c; without.q = total.q - part.q;
    const auto finalized = finalize(without, coverage_cut);
    const double science_threshold = threshold(finalized.weight, coverage_cut);
    result.support = ((finalized.weight.array() > 0) && (finalized.weight.array() >= science_threshold)).cast<std::int64_t>();
    result.conditioned = ResponseCountMatrix::Zero(rows_, cols_);
    result.footprint = result.conditioned;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    result.difference = Eigen::MatrixXd::Constant(rows_, cols_, nan);
    result.leverage = result.contrast = result.identity_residual = result.difference;
    result.n_error = result.c_error = result.q_error = result.signal_error = result.coefficient_error = result.difference;
    result.score.map_scale = map_scale;
    long double squared_response = 0;
    for (Eigen::Index i = 0; i < total.n.size(); ++i) {
        const bool original_support = science_support_.at(array)(i);
        const bool footprint = original_support && part.abs_c(i) > 0;
        result.footprint(i) = footprint;
        result.score.footprint += footprint;
        result.score.lost += original_support && !result.support(i);
        result.score.gained += !original_support && result.support(i);
        const double nb = difference_bound(total.n(i), part.n(i), total.abs_n(i), part.abs_n(i), total.count(i), part.count(i));
        const double cb = difference_bound(total.c(i), part.c(i), total.abs_c(i), part.abs_c(i), total.count(i), part.count(i));
        const double qb = difference_bound(total.q(i), part.q(i), std::abs(total.q(i)), std::abs(part.q(i)), total.count(i), part.count(i));
        result.n_error(i) = nb; result.c_error(i) = cb; result.q_error(i) = qb;
        const double cmargin = std::abs(without.c(i)) - cb, qmargin = without.q(i) - qb;
        const bool conditioned = cmargin > 1e-8 && qmargin > 0 &&
            std::abs(total.c(i)) > 1e-8 && std::abs(part.c(i)) > 1e-8;
        if (original_support && part.abs_c(i) > 0 && !conditioned) result.score.support_risk = true;
        if (!conditioned) continue;
        const double all_signal = total.n(i) / total.c(i), target_signal = part.n(i) / part.c(i);
        const double deleted_signal = without.n(i) / without.c(i);
        const double difference = deleted_signal - all_signal;
        const double leverage = part.c(i) / total.c(i), contrast = deleted_signal - target_signal;
        const double predicted = leverage * contrast, residual = difference - predicted;
        const double bound = identity_safety * u * std::max({1.0, std::abs(difference), std::abs(predicted),
            std::abs(all_signal), std::abs(target_signal), std::abs(deleted_signal)});
        if (!std::isfinite(residual) || std::abs(residual) > bound)
            throw std::runtime_error("EL-F12 signed deletion identity exceeds the approved bound");
        result.difference(i) = difference; result.leverage(i) = leverage;
        result.contrast(i) = contrast; result.identity_residual(i) = residual;
        result.signal_error(i) = finalization_safety * ((nb + std::abs(deleted_signal) * cb) / cmargin +
            u * std::max(1.0, std::abs(deleted_signal)));
        const double raw_coefficient = without.c(i) * without.c(i) / without.q(i);
        result.coefficient_error(i) = finalization_safety * (
            (2 * std::abs(without.c(i)) * cb + cb * cb) / qmargin +
            std::pow(std::abs(without.c(i)) + cb, 2) * qb / (without.q(i) * qmargin) +
            u * std::max(1.0, std::abs(raw_coefficient)));
        result.conditioned(i) = footprint && result.support(i);
        if (result.conditioned(i)) {
            ++result.score.conditioned;
            squared_response += static_cast<long double>(difference) * difference;
        }
    }
    result.score.support_risk = result.score.support_risk || result.score.lost > 0;
    if (result.score.conditioned > 0)
        result.score.ratio = std::sqrt(static_cast<double>(squared_response / result.score.conditioned)) / map_scale;
    else if (result.score.footprint > 0) result.score.available = false;
    result.score.validate();
    return result;
}

void ResponseOccurrenceLedger::finish(
    ResponseInterventionState &state, std::vector<ResponseCandidate> census,
    const std::vector<Eigen::MatrixXd> &signal, const std::vector<Eigen::MatrixXd> &weight,
    const std::vector<Eigen::MatrixXd> &formal_weight, double coverage_cut) {
    if (!captured_ || finished_ || state.iteration() != iteration_ || state.completed() ||
        signal.size() != map_arrays_.size() || weight.size() != signal.size() ||
        (!formal_weight.empty() && formal_weight.size() != signal.size()) ||
        !std::isfinite(coverage_cut) || coverage_cut < 0)
        throw std::runtime_error("EL-F12 incomplete map/ledger boundary");
    std::size_t eligible = 0;
    std::set<ResponseKey> keys;
    for (const auto &candidate : census) {
        candidate.key.validate();
        if (candidate.key.observation != observation_ || !totals_.count(candidate.key.array) ||
            !keys.insert(candidate.key).second) throw std::runtime_error("EL-F12 census identity mismatch");
        eligible += candidate.disposition == "eligible";
    }
    if (eligible > ResponseInterventionState::boundary_limit)
        throw std::runtime_error("EL-F12 full eligible population exceeds 16");
    spool_.close();
    if (!spool_ || std::filesystem::file_size(path_) != bytes_)
        throw std::runtime_error("EL-F12 occurrence spool close/size failure");
    reconstruct(census);
    std::map<int, double> scales;
    for (std::size_t slot = 0; slot < map_arrays_.size(); ++slot) {
        const int array = map_arrays_[slot];
        for (const auto *value : {&signal[slot], &weight[slot]})
            if (value->rows() != rows_ || value->cols() != cols_)
                throw std::runtime_error("EL-F12 final map shape mismatch");
        const auto finalized = finalize(totals_.at(array), coverage_cut);
        if (!(finalized.signal.array() == signal[slot].array()).all())
            throw std::runtime_error("EL-F12 total re-finalization does not reproduce the science map");
        if (!formal_weight.empty() && (formal_weight[slot].rows() != rows_ || formal_weight[slot].cols() != cols_ ||
            !(formal_weight[slot].array() == finalized.weight.array()).all()))
            throw std::runtime_error("EL-F12 total re-finalization does not reproduce formal weights");
        signal_[array] = signal[slot]; weight_[array] = weight[slot]; formal_weight_[array] = finalized.weight;
        const double limit = threshold(weight[slot], coverage_cut);
        auto &support = science_support_[array];
        support = ((weight[slot].array().isFinite()) && (weight[slot].array() > 0) &&
                   (weight[slot].array() >= limit) && signal[slot].array().isFinite()).cast<std::int64_t>();
        std::vector<double> values;
        for (Eigen::Index i = 0; i < support.size(); ++i) if (support(i)) values.push_back(signal[slot](i));
        const double center = median(values);
        for (auto &value : values) value = std::abs(value - center);
        scales[array] = std::max(1.0, 1.4826 * median(values));
        if (!std::isfinite(scales[array])) throw std::runtime_error("EL-F12 nonfinite descriptive map scale");
    }
    for (auto &[key, deletion] : deletions_)
        deletion = evaluate(key.array, std::move(deletion.target), coverage_cut, scales.at(key.array));
    std::map<int, ResponseScore> joint_scores;
    for (auto &[array, deletion] : joint_) {
        deletion = evaluate(array, std::move(deletion.target), coverage_cut, scales.at(array));
        joint_scores[array] = deletion.score;
    }
    for (auto &candidate : census)
        if (candidate.disposition == "eligible") candidate.response = deletions_.at(candidate.key).score;
    state.resolve(std::move(census), std::move(joint_scores));
    coverage_cut_ = coverage_cut;
    finished_ = true;
}

void ResponseOccurrenceLedger::write_receipt(
    const std::filesystem::path &path, const ResponseInterventionState &state,
    double pixel_size_rad, const std::string &unit) const {
    if (!finished_ || !state.completed() || state.iteration() != iteration_ || unit != "mJy/beam" ||
        !std::isfinite(pixel_size_rad) || pixel_size_rad <= 0 || std::filesystem::exists(path))
        throw std::runtime_error("EL-F12 invalid or colliding required ledger output");
    write_netcdf_atomic(path.string(), [&](netCDF::NcFile &file) {
        file.putAtt("schema", "SCI-FRUIT-EL-F12-OCCURRENCE-LEDGER-R0.1");
        file.putAtt("diagnostic_only", netCDF::ncInt, 1);
        file.putAtt("observation", observation_);
        file.putAtt("iteration", netCDF::ncInt, iteration_);
        file.putAtt("pixel_size_rad", netCDF::ncDouble, pixel_size_rad);
        file.putAtt("signal_unit", unit);
        file.putAtt("coverage_cut", netCDF::ncDouble, coverage_cut_);
        file.putAtt("indexing", "zero-based map row/column and scan; same grid as companion raw-observation FITS");
        file.putAtt("frame", "AZOFFSET/ELOFFSET; exact WCS bound by companion FITS and registration");
        file.putAtt("action_horizon", netCDF::ncInt, ResponseInterventionState::horizon);
        file.putAtt("half_multiplier", netCDF::ncDouble, 0.5);
        file.putAtt("selector_cutoff", netCDF::ncDouble, 1.0);
        file.putAtt("map_scale_floor_mJy_per_beam", netCDF::ncDouble, 1.0);
        file.putAtt("state", state.serialize());
        file.putAtt("spool_path", path_.string());
        file.putAtt("spool_record_bytes", netCDF::ncInt, static_cast<int>(record_bytes));
        file.putAtt("spool_records", netCDF::ncInt64, static_cast<long long>(bytes_ / record_bytes));
        file.putAtt("science_occurrences", netCDF::ncInt64, static_cast<long long>(occurrences_));
        file.putAtt("spool_byte_order", std::endian::native == std::endian::little ? "little" : "big");
        file.putAtt("spool_int64_fields", "kind scan uid array row col kernel sample");
        file.putAtt("spool_binary64_fields", "N_scale C_scale Q_scale processed_signal multiplier");
        const auto rows = file.addDim("map_row", static_cast<std::size_t>(rows_));
        const auto cols = file.addDim("map_col", static_cast<std::size_t>(cols_));
        for (const auto &[array, total] : totals_) {
            auto group = file.addGroup("array_" + std::to_string(array));
            accumulator_planes(group, total, rows, cols);
            plane(group, "signal", signal_.at(array), rows, cols);
            plane(group, "weight", weight_.at(array), rows, cols);
            plane(group, "formal_weight", formal_weight_.at(array), rows, cols);
            plane(group, "science_support", science_support_.at(array), rows, cols);
            for (std::size_t i = 0; i < kernels_.at(array).size(); ++i) {
                auto kg = group.addGroup("kernel_" + std::to_string(i));
                const auto kr = kg.addDim("kernel_row", kernels_.at(array)[i].rows());
                const auto kc = kg.addDim("kernel_col", kernels_.at(array)[i].cols());
                plane(kg, "coefficient", kernels_.at(array)[i], kr, kc);
                plane(kg, "coefficient_square", squared_kernels_.at(array)[i], kr, kc);
            }
        }
        auto write_deletion = [&](netCDF::NcGroup group, const ResponseDeletion &deletion) {
            accumulator_planes(group, deletion.target, rows, cols);
            plane(group, "deletion_response", deletion.difference, rows, cols);
            plane(group, "signed_leverage", deletion.leverage, rows, cols);
            plane(group, "signal_contrast", deletion.contrast, rows, cols);
            plane(group, "identity_residual", deletion.identity_residual, rows, cols);
            plane(group, "N_error_bound", deletion.n_error, rows, cols);
            plane(group, "C_error_bound", deletion.c_error, rows, cols);
            plane(group, "Q_error_bound", deletion.q_error, rows, cols);
            plane(group, "signal_error_bound", deletion.signal_error, rows, cols);
            plane(group, "coefficient_error_bound", deletion.coefficient_error, rows, cols);
            plane(group, "conditioned_footprint", deletion.conditioned, rows, cols);
            plane(group, "deletion_science_support", deletion.support, rows, cols);
            plane(group, "target_footprint", deletion.footprint, rows, cols);
        };
        int index = 0;
        for (const auto &[key, deletion] : deletions_) {
            auto group = file.addGroup("candidate_" + std::to_string(index++));
            group.putAtt("array", netCDF::ncInt, key.array);
            group.putAtt("uid", netCDF::ncInt, key.uid);
            group.putAtt("scan", netCDF::ncInt, key.scan);
            write_deletion(group, deletion);
        }
        for (const auto &[array, deletion] : joint_)
            write_deletion(file.addGroup("joint_array_" + std::to_string(array)), deletion);
    });
}

}  // namespace citlali::fruit
