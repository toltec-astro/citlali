#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace citlali::fruit {

// SCI-FRUIT EL-F12 + CAP-001. Reduction-owned, serial execution only.
// UID/array are scientific identities; scan and iteration are zero-based.
// Scores are dimensionless descriptive response, never benefit or probability.
struct ResponseKey {
    std::string observation;
    int array = -1;
    int uid = -1;
    int scan = -1;
    auto identity() const { return std::tie(observation, array, uid, scan); }
    bool operator<(const ResponseKey &other) const { return identity() < other.identity(); }
    bool operator==(const ResponseKey &other) const { return identity() == other.identity(); }
    void validate() const {
        if (observation.empty() || array < 0 || array > 2 || uid < 0 || scan < 0)
            throw std::runtime_error("EL-F12 invalid observation/array/UID/scan identity");
    }
};

struct ResponseScore {
    bool available = true;
    bool support_risk = false;
    double ratio = 0.0;
    double map_scale = 1.0;  // mJy/beam, robust scatter with the approved floor
    long long footprint = 0;
    long long conditioned = 0;
    long long lost = 0;
    long long gained = 0;
    bool selected() const { return support_risk || (available && ratio >= 1.0); }
    void validate() const {
        if (!std::isfinite(ratio) || ratio < 0 || !std::isfinite(map_scale) ||
            map_scale < 1.0 || footprint < 0 || conditioned < 0 || lost < 0 ||
            gained < 0 || conditioned > footprint || (!available && ratio != 0.0) ||
            (lost > 0 && !support_risk) || (footprint == 0 && ratio != 0.0) ||
            (!available && conditioned != 0))
            throw std::runtime_error("EL-F12 invalid response score/support receipt");
    }
};

struct ResponseCandidate {
    ResponseKey key;
    // Non-eligible records remain in the census with an explicit reason.
    std::string disposition = "eligible";
    ResponseScore response;
    bool selected = false;
};

struct ResponseStageReceipt {
    // 0 unvisited, 1 inapplicable, 2 absent, 3 rejected, 4 permitted.
    int status = 0;
    double proposed_fraction = 0.0;
    double cap = 0.0;
    bool suppressed = false;
    bool independent_reason = false;
    long long ordinary_new_samples = 0;
};

class ResponseInterventionState {
public:
    static constexpr int horizon = 6;
    static constexpr std::size_t boundary_limit = 16;
    static constexpr std::size_t assignment_limit = 64;
    static constexpr const char *version = "SCI-FRUIT-EL-F12-STATE-R0.1+CAP-001";

    static bool valid_arm(const std::string &arm) {
        return arm == "disabled" || arm == "H" || arm == "Half" || arm == "Hold";
    }
    bool enabled() const { return arm_ != "disabled"; }
    const std::string &arm() const { return arm_; }
    int iteration() const { return iteration_; }
    bool completed() const { return completed_; }
    const auto &assignments() const { return assignments_; }
    const auto &census() const { return census_; }
    const auto &unions() const { return unions_; }
    const auto &receipts() const { return receipts_; }
    const auto &first_applications() const { return first_application_; }

    void configure(const std::string &arm) {
        if (!valid_arm(arm)) throw std::runtime_error("EL-F12 unknown action arm");
        *this = ResponseInterventionState{};
        arm_ = arm;
    }

    void begin(int iteration, bool application_active) {
        if (!enabled()) return;
        if (iteration < 0 || iteration > horizon ||
            (iteration_ < 0 ? iteration != 0 : (!completed_ || iteration != iteration_ + 1)))
            throw std::runtime_error("EL-F12 missing boundary state or unsupported continuation");
        iteration_ = iteration;
        completed_ = false;
        census_.clear();
        unions_.clear();
        receipts_.clear();
        for (const auto &[key, source] : assignments_) {
            (void) source;
            auto &stages = receipts_[key];
            if (!application_active) for (auto &stage : stages) stage.status = 1;
        }
    }

    bool assigned(const ResponseKey &key) const { return assignments_.count(key) != 0; }
    bool suppresses(const ResponseKey &key) const {
        return enabled() && arm_ != "H" && active(key);
    }

    void record_stage(const ResponseKey &key, int stage, int status,
                      double proposed_fraction, double cap,
                      bool suppressed = false, bool independent_reason = false,
                      long long ordinary_new_samples = 0) {
        if (!active(key)) return;
        if (stage < 0 || stage > 1 || status < 1 || status > 4 ||
            !std::isfinite(proposed_fraction) || proposed_fraction < 0.0 ||
            proposed_fraction > 1.0 || !std::isfinite(cap) || cap < 0.0 || ordinary_new_samples < 0 ||
            (suppressed && (status != 4 || independent_reason || arm_ == "H")))
            throw std::runtime_error("EL-F12 malformed stage receipt");
        auto &receipt = receipts_.at(key).at(static_cast<std::size_t>(stage));
        if (receipt.status != 0)
            throw std::runtime_error("EL-F12 duplicate or inapplicable stage evaluation");
        if ((status == 4 && cap > 0.0 && proposed_fraction > cap) ||
            (status == 3 && !(cap > 0.0 && proposed_fraction > cap)))
            throw std::runtime_error("EL-F12 stage receipt contradicts the ordinary cap");
        receipt = {status, proposed_fraction, cap, suppressed, independent_reason, ordinary_new_samples};
        if ((suppressed || (arm_ == "H" && status == 4)) && ordinary_new_samples > 0) mark_applied(key);
    }

    void mark_applied(const ResponseKey &key) {
        if (!active(key)) throw std::runtime_error("EL-F12 action outside its assignment horizon");
        auto &first = first_application_.at(key);
        if (first < 0) first = iteration_;
    }

    double coefficient(const ResponseKey &key) const {
        if (!enabled() || !active(key)) return 1.0;
        const auto &stages = receipts_.at(key);
        if (stages[0].status == 0 || stages[1].status == 0)
            throw std::runtime_error("EL-F12 map action precedes its stage receipts");
        if (arm_ != "Half") return 1.0;
        return stages[0].status == 4 || stages[1].status == 4 ? 0.5 : 1.0;
    }

    // All eligible keys and one combined score per represented array are
    // supplied by the causal ledger at this completed map boundary.
    void resolve(std::vector<ResponseCandidate> census,
                 std::map<int, ResponseScore> unions) {
        if (!enabled()) return;
        if (iteration_ < 0 || completed_)
            throw std::runtime_error("EL-F12 duplicate/unstarted decision boundary");
        for (const auto &[key, stages] : receipts_) {
            (void) key;
            if (stages[0].status == 0 || stages[1].status == 0)
                throw std::runtime_error("EL-F12 incomplete application-stage census");
        }
        std::sort(census.begin(), census.end(), [](const auto &a, const auto &b) {
            return a.key < b.key;
        });
        std::set<int> eligible_arrays;
        std::size_t eligible_count = 0;
        auto next_assignments = assignments_;
        for (std::size_t i = 0; i < census.size(); ++i) {
            auto &candidate = census[i];
            candidate.key.validate();
            if (i && candidate.key == census[i - 1].key)
                throw std::runtime_error("EL-F12 duplicate candidate key");
            if (!valid_disposition(candidate.disposition))
                throw std::runtime_error("EL-F12 unknown census disposition");
            candidate.selected = false;
            if (candidate.disposition != "eligible") continue;
            if (iteration_ == horizon || assigned(candidate.key))
                throw std::runtime_error("EL-F12 ineligible assignment at decision boundary");
            ++eligible_count;
            eligible_arrays.insert(candidate.key.array);
            candidate.response.validate();
        }
        if (eligible_count > boundary_limit || unions.size() != eligible_arrays.size())
            throw std::runtime_error("EL-F12 candidate cap exceeded or incomplete joint census");
        for (const int array : eligible_arrays) unions.at(array).validate();
        for (auto &candidate : census) {
            if (candidate.disposition != "eligible") continue;
            candidate.selected = candidate.response.selected() || unions.at(candidate.key.array).selected();
            if (candidate.selected) next_assignments.emplace(candidate.key, iteration_);
        }
        if (next_assignments.size() > assignment_limit)
            throw std::runtime_error("EL-F12 assignment cap exceeded; no truncation allowed");
        // Commit the whole decision only after all accounting/limits pass.
        assignments_ = std::move(next_assignments);
        for (const auto &[key, source] : assignments_) { (void) source; first_application_.try_emplace(key, -1); }
        census_ = std::move(census);
        unions_ = std::move(unions);
        completed_ = true;
    }

    std::string serialize() const {
        if (!enabled() || !completed_) throw std::runtime_error("EL-F12 cannot checkpoint unfinished state");
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::setprecision(17) << version << '\n' << arm_ << ' ' << iteration_ << '\n';
        out << assignments_.size() << '\n';
        for (const auto &[key, source] : assignments_) {
            write_key(out, key); out << source << ' ' << first_application_.at(key) << '\n';
        }
        out << census_.size() << '\n';
        for (const auto &candidate : census_) {
            write_key(out, candidate.key);
            out << candidate.disposition << ' ' << candidate.selected << ' ';
            write_score(out, candidate.response);
        }
        out << unions_.size() << '\n';
        for (const auto &[array, score] : unions_) { out << array << ' '; write_score(out, score); }
        out << receipts_.size() << '\n';
        for (const auto &[key, stages] : receipts_) {
            write_key(out, key);
            for (const auto &stage : stages) out << stage.status << ' ' << stage.proposed_fraction << ' ' << stage.cap << ' '
                << stage.suppressed << ' ' << stage.independent_reason << ' ' << stage.ordinary_new_samples << ' ';
            out << '\n';
        }
        if (!out) throw std::runtime_error("EL-F12 state serialization failed");
        return out.str();
    }

    static ResponseInterventionState restore(const std::string &text,
                                            const std::string &expected_arm,
                                            int completed_iteration) {
        std::istringstream in(text);
        in.imbue(std::locale::classic());
        std::string schema, arm;
        int iteration = -1;
        in >> schema >> arm >> iteration;
        if (!in || schema != version || arm != expected_arm || arm == "disabled" ||
            iteration != completed_iteration || iteration < 0 || iteration >= horizon)
            throw std::runtime_error("EL-F12 incompatible checkpoint or continuation beyond iteration 6");
        ResponseInterventionState state;
        state.configure(arm);
        state.iteration_ = iteration;
        for (std::size_t n = read_count(in, assignment_limit); n; --n) {
            auto key = read_key(in);
            int source = -1, first = -1; in >> source >> first;
            if (source < 0 || source > iteration || source >= horizon ||
                (first != -1 && (first <= source || first > iteration)) ||
                !state.assignments_.emplace(key, source).second)
                throw std::runtime_error("EL-F12 invalid checkpoint assignment");
            state.first_application_[key] = first;
        }
        for (std::size_t n = read_count(in, 200000); n; --n) {
            ResponseCandidate candidate;
            candidate.key = read_key(in);
            in >> candidate.disposition >> candidate.selected;
            candidate.response = read_score(in);
            if (!valid_disposition(candidate.disposition) ||
                (!state.census_.empty() && !(state.census_.back().key < candidate.key)))
                throw std::runtime_error("EL-F12 invalid checkpoint census");
            state.census_.push_back(candidate);
        }
        for (std::size_t n = read_count(in, 3); n; --n) {
            int array = -1; in >> array;
            auto score = read_score(in);
            if (array < 0 || array > 2 || !state.unions_.emplace(array, score).second)
                throw std::runtime_error("EL-F12 invalid checkpoint joint response");
        }
        for (std::size_t n = read_count(in, assignment_limit); n; --n) {
            auto key = read_key(in);
            std::array<ResponseStageReceipt, 2> stages;
            for (auto &stage : stages) {
                in >> stage.status >> stage.proposed_fraction >> stage.cap
                   >> stage.suppressed >> stage.independent_reason >> stage.ordinary_new_samples;
                if (stage.status < 1 || stage.status > 4 || !std::isfinite(stage.proposed_fraction) ||
                    stage.proposed_fraction < 0 || stage.proposed_fraction > 1 ||
                    !std::isfinite(stage.cap) || stage.cap < 0 || stage.ordinary_new_samples < 0 ||
                    (stage.suppressed && (stage.status != 4 || stage.independent_reason || arm == "H")) ||
                    (stage.status == 4 && stage.cap > 0 && stage.proposed_fraction > stage.cap) ||
                    (stage.status == 3 && !(stage.cap > 0 && stage.proposed_fraction > stage.cap)))
                    throw std::runtime_error("EL-F12 invalid checkpoint stage receipt");
            }
            if (!state.assignments_.count(key) || state.assignments_.at(key) >= iteration ||
                !state.receipts_.emplace(key, stages).second)
                throw std::runtime_error("EL-F12 checkpoint receipt identity mismatch");
        }
        if (!in) throw std::runtime_error("EL-F12 truncated checkpoint state");
        in >> std::ws;
        if (!in.eof()) throw std::runtime_error("EL-F12 trailing checkpoint state");
        for (const auto &[key, source] : state.assignments_) {
            if (source < iteration && !state.receipts_.count(key))
                throw std::runtime_error("EL-F12 missing checkpoint stage receipt");
        }
        // Recompute selection from the stored descriptive scores. No final
        // result, sky truth, UID ranking or other arm is an input.
        auto prior = state.assignments_;
        for (auto it = state.assignments_.begin(); it != state.assignments_.end();) {
            if (it->second == iteration) it = state.assignments_.erase(it); else ++it;
        }
        const auto original_census = state.census_;
        state.resolve(state.census_, state.unions_);
        if (state.assignments_ != prior || state.census_.size() != original_census.size())
            throw std::runtime_error("EL-F12 checkpoint selection mismatch");
        for (std::size_t i = 0; i < original_census.size(); ++i)
            if (state.census_[i].selected != original_census[i].selected)
                throw std::runtime_error("EL-F12 checkpoint selection flag mismatch");
        if (state.serialize() != text) throw std::runtime_error("EL-F12 noncanonical checkpoint state");
        return state;
    }

private:
    std::string arm_ = "disabled";
    int iteration_ = -1;
    bool completed_ = false;
    std::map<ResponseKey, int> assignments_;
    std::map<ResponseKey, int> first_application_;
    std::vector<ResponseCandidate> census_;
    std::map<int, ResponseScore> unions_;
    std::map<ResponseKey, std::array<ResponseStageReceipt, 2>> receipts_;

    bool active(const ResponseKey &key) const {
        const auto it = assignments_.find(key);
        return it != assignments_.end() && iteration_ > it->second && iteration_ <= horizon;
    }
    static bool valid_disposition(const std::string &value) {
        return value == "eligible" || value == "not_new" || value == "already_assigned" ||
               value == "independent_exclusion" || value == "entry_hard_exclusion" || value == "horizon";
    }
    static void write_key(std::ostream &out, const ResponseKey &key) {
        out << std::quoted(key.observation) << ' ' << key.array << ' ' << key.uid << ' ' << key.scan << ' ';
    }
    static ResponseKey read_key(std::istream &in) {
        ResponseKey key;
        in >> std::quoted(key.observation) >> key.array >> key.uid >> key.scan;
        key.validate();
        return key;
    }
    static std::size_t read_count(std::istream &in, std::size_t limit) {
        long long count = -1; in >> count;
        if (!in || count < 0 || static_cast<unsigned long long>(count) > limit)
            throw std::runtime_error("EL-F12 invalid checkpoint cardinality");
        return static_cast<std::size_t>(count);
    }
    static void write_score(std::ostream &out, const ResponseScore &score) {
        out << score.available << ' ' << score.support_risk << ' ' << score.ratio << ' ' << score.map_scale << ' '
            << score.footprint << ' ' << score.conditioned << ' ' << score.lost << ' ' << score.gained << '\n';
    }
    static ResponseScore read_score(std::istream &in) {
        ResponseScore score;
        in >> score.available >> score.support_risk >> score.ratio >> score.map_scale
           >> score.footprint >> score.conditioned >> score.lost >> score.gained;
        score.validate();
        return score;
    }
};

}  // namespace citlali::fruit
