#pragma once

// Reusable authority for issuing opaque artifact occurrences and publishing
// one canonical artifact with one envelope-bound receipt. Scientific codecs
// remain responsible for canonical serialization and reread validation.

#include <citlali/core/utils/sha256.h>

#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline::canonical_artifact_publication {

inline constexpr std::string_view receipt_schema_v1 =
    "citlali-canonical-apt-publication-receipt-v1";

class PublicationError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct OpaqueIssuance {
    std::string occurrence;
    std::string event_reference;

    friend bool operator==(const OpaqueIssuance &,
                           const OpaqueIssuance &) = default;
};

using IssuanceFactory = std::function<OpaqueIssuance()>;

namespace detail {

inline void require_single_line_value(std::string_view value,
                                      std::string_view label,
                                      bool allow_empty = false) {
    if ((!allow_empty && value.empty()) ||
        value.find('\0') != std::string_view::npos ||
        value.find('\r') != std::string_view::npos ||
        value.find('\n') != std::string_view::npos) {
        throw PublicationError("canonical artifact " + std::string(label) +
                               " is empty or not single-line text");
    }
}

inline bool is_sha256_reference(std::string_view value) noexcept {
    constexpr std::string_view prefix = "sha256:";
    if (!value.starts_with(prefix) || value.size() != prefix.size() + 64U) {
        return false;
    }
    for (const char ch : value.substr(prefix.size())) {
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) {
            return false;
        }
    }
    return true;
}

inline std::uint64_t exact_byte_count(std::string_view bytes) {
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
        if (bytes.size() > std::numeric_limits<std::uint64_t>::max()) {
            throw PublicationError(
                "canonical artifact byte count is not representable");
        }
    }
    return static_cast<std::uint64_t>(bytes.size());
}

inline bool path_exists(const std::filesystem::path &path) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
        throw PublicationError(
            "failed to inspect canonical artifact publication path " +
            path.string() + ": " + error.message());
    }
    return exists;
}

inline std::string read_binary_file(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw PublicationError("failed to open canonical artifact file: " +
                               path.string());
    }
    std::string bytes{std::istreambuf_iterator<char>(stream),
                      std::istreambuf_iterator<char>()};
    if (stream.bad()) {
        throw PublicationError("failed to read canonical artifact file: " +
                               path.string());
    }
    return bytes;
}

inline void write_binary_file(const std::filesystem::path &path,
                              std::string_view bytes) {
    std::ofstream stream(path,
                         std::ios::binary | std::ios::out | std::ios::trunc);
    if (!stream) {
        throw PublicationError(
            "failed to create staged canonical artifact file: " +
            path.string());
    }
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream.flush();
    if (!stream) {
        throw PublicationError(
            "failed to write staged canonical artifact file: " +
            path.string());
    }
    stream.close();
    if (!stream) {
        throw PublicationError(
            "failed to close staged canonical artifact file: " +
            path.string());
    }
}

inline void make_publication_source_read_only(
    const std::filesystem::path &path) {
    std::error_code error;
    std::filesystem::permissions(
        path,
        std::filesystem::perms::owner_read |
            std::filesystem::perms::group_read |
            std::filesystem::perms::others_read,
        std::filesystem::perm_options::replace, error);
    if (error) {
        throw PublicationError(
            "failed to protect staged canonical artifact publication source: " +
            error.message());
    }
}

inline bool remove_if_owned_hard_link(
    const std::filesystem::path &owner,
    const std::filesystem::path &published) noexcept {
    std::error_code existence_error;
    if (!std::filesystem::exists(published, existence_error) &&
        !existence_error) {
        return true;
    }
    std::error_code equivalent_error;
    const bool owned = std::filesystem::exists(owner, equivalent_error) &&
        !equivalent_error &&
        std::filesystem::exists(published, equivalent_error) &&
        !equivalent_error &&
        std::filesystem::equivalent(owner, published, equivalent_error) &&
        !equivalent_error;
    if (!owned) {
        return false;
    }
    std::error_code remove_error;
    return std::filesystem::remove(published, remove_error) && !remove_error;
}

inline void publish_no_replace_hard_link(
    const std::filesystem::path &source,
    const std::filesystem::path &published) {
    std::error_code error;
    std::filesystem::create_hard_link(source, published, error);
    if (error) {
        throw PublicationError(
            "canonical artifact no-overwrite publication failed: " +
            published.string() + ": " + error.message());
    }
}

inline void require_owned_publication_alias(
    const std::filesystem::path &alias,
    const std::filesystem::path &owner) {
    std::error_code error;
    if (!std::filesystem::exists(alias, error) || error ||
        !std::filesystem::exists(owner, error) || error ||
        !std::filesystem::equivalent(alias, owner, error) || error) {
        throw PublicationError(
            "canonical artifact staged publication entry changed after validation");
    }
}

}  // namespace detail

inline std::string make_entropy_reference(std::string_view prefix) {
    detail::require_single_line_value(prefix, "entropy prefix", true);
    std::array<unsigned char, 32> bytes{};
    std::ifstream entropy("/dev/urandom", std::ios::binary);
    if (!entropy) {
        throw PublicationError(
            "canonical artifact OS entropy source is unavailable");
    }
    entropy.read(reinterpret_cast<char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (entropy.gcount() != static_cast<std::streamsize>(bytes.size()) ||
        !entropy) {
        throw PublicationError(
            "canonical artifact OS entropy source returned a short read");
    }
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << prefix;
    for (const auto byte : bytes) {
        stream << std::hex << std::nouppercase << std::setfill('0')
               << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return stream.str();
}

inline OpaqueIssuance make_entropy_issuance(
    std::string_view occurrence_prefix,
    std::string_view event_reference_prefix) {
    return {make_entropy_reference(occurrence_prefix),
            make_entropy_reference(event_reference_prefix)};
}

inline OpaqueIssuance issue_opaque(const IssuanceFactory &factory) {
    if (!factory) {
        throw PublicationError(
            "canonical artifact has no opaque issuance factory");
    }
    auto issuance = factory();
    detail::require_single_line_value(issuance.occurrence,
                                      "occurrence");
    detail::require_single_line_value(issuance.event_reference,
                                      "event reference");
    return issuance;
}

struct ReceiptBinding {
    std::string schema;
    std::string scope;
    std::string envelope_sha256;
    std::string byte_sha256;
    std::uint64_t byte_count = 0;

    friend bool operator==(const ReceiptBinding &,
                           const ReceiptBinding &) = default;
};

inline void validate_receipt_contract(const ReceiptBinding &receipt) {
    detail::require_single_line_value(receipt.schema, "receipt schema");
    detail::require_single_line_value(receipt.scope, "receipt scope");
    if (!detail::is_sha256_reference(receipt.envelope_sha256) ||
        !detail::is_sha256_reference(receipt.byte_sha256)) {
        throw PublicationError(
            "canonical artifact receipt has an invalid SHA-256 reference");
    }
}

inline ReceiptBinding make_receipt_binding(
    std::string schema, std::string scope, std::string envelope_sha256,
    std::string_view artifact_bytes) {
    ReceiptBinding result{
        std::move(schema), std::move(scope), std::move(envelope_sha256),
        "sha256:" + citlali::utils::sha256(artifact_bytes),
        detail::exact_byte_count(artifact_bytes)};
    validate_receipt_contract(result);
    return result;
}

inline std::string canonical_receipt_bytes(const ReceiptBinding &receipt) {
    validate_receipt_contract(receipt);
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << receipt.schema << '\n'
           << "scope=" << receipt.scope << '\n'
           << "envelope_sha256=" << receipt.envelope_sha256 << '\n'
           << "byte_sha256=" << receipt.byte_sha256 << '\n'
           << "byte_count=" << receipt.byte_count << '\n';
    return stream.str();
}

inline ReceiptBinding parse_canonical_receipt(
    std::string_view bytes, std::string_view expected_schema,
    std::string_view expected_scope) {
    detail::require_single_line_value(expected_schema,
                                      "expected receipt schema");
    detail::require_single_line_value(expected_scope,
                                      "expected receipt scope");
    if (bytes.empty() || bytes.back() != '\n' ||
        bytes.find('\r') != std::string_view::npos ||
        bytes.find('\0') != std::string_view::npos) {
        throw PublicationError(
            "canonical artifact receipt must be exact LF-terminated text");
    }
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    while (start < bytes.size()) {
        const auto end = bytes.find('\n', start);
        lines.push_back(bytes.substr(start, end - start));
        start = end + 1;
    }
    if (lines.size() != 5 || lines[0] != expected_schema) {
        throw PublicationError(
            "canonical artifact receipt schema or field count is invalid");
    }
    const auto value = [](std::string_view line,
                          std::string_view prefix) {
        if (!line.starts_with(prefix) || line.size() == prefix.size()) {
            throw PublicationError(
                "canonical artifact receipt has a missing or misordered field");
        }
        return line.substr(prefix.size());
    };
    ReceiptBinding result;
    result.schema = lines[0];
    result.scope = value(lines[1], "scope=");
    result.envelope_sha256 = value(lines[2], "envelope_sha256=");
    result.byte_sha256 = value(lines[3], "byte_sha256=");
    const auto count = value(lines[4], "byte_count=");
    const auto [end, error] = std::from_chars(
        count.data(), count.data() + count.size(), result.byte_count, 10);
    if (error != std::errc{} || end != count.data() + count.size() ||
        result.scope != expected_scope) {
        throw PublicationError(
            "canonical artifact receipt scope or byte count is invalid");
    }
    validate_receipt_contract(result);
    if (canonical_receipt_bytes(result) != bytes) {
        throw PublicationError(
            "canonical artifact receipt is not exact canonical text");
    }
    return result;
}

inline void validate_receipt_binding(
    std::string_view artifact_bytes, const ReceiptBinding &receipt) {
    validate_receipt_contract(receipt);
    if (receipt.byte_count != detail::exact_byte_count(artifact_bytes) ||
        receipt.byte_sha256 !=
            "sha256:" + citlali::utils::sha256(artifact_bytes)) {
        throw PublicationError(
            "canonical artifact bytes disagree with the publication receipt");
    }
}

enum class PublicationStage {
    artifact_staged = 0,
    ecsv_staged = artifact_staged,
    artifact_validated = 1,
    ecsv_validated = artifact_validated,
    receipt_staged = 2,
    receipt_validated = 3,
    before_artifact_publish = 4,
    before_ecsv_publish = before_artifact_publish,
    artifact_published = 5,
    ecsv_published = artifact_published,
    before_receipt_publish = 6,
};

struct PublicationHooks {
    std::function<void(PublicationStage, const std::filesystem::path &,
                       const std::filesystem::path &)>
        on_stage;
};

using PublicationValidator =
    std::function<void(std::string_view, std::string_view)>;

struct PublicationPlan {
    std::filesystem::path artifact_path;
    std::filesystem::path receipt_path;
    std::string artifact_bytes;
    std::string receipt_bytes;
    PublicationValidator validate;
};

struct PublicationResult {
    std::filesystem::path artifact_path;
    std::filesystem::path receipt_path;
};

static_assert(std::is_nothrow_move_constructible_v<PublicationResult>);

inline void notify_publication_stage(
    const PublicationHooks &hooks, PublicationStage stage,
    const std::filesystem::path &staged_artifact,
    const std::filesystem::path &staged_receipt) {
    if (hooks.on_stage) {
        hooks.on_stage(stage, staged_artifact, staged_receipt);
    }
}

class StagingDirectoryGuard {
public:
    explicit StagingDirectoryGuard(
        const std::filesystem::path &output_path) {
        const auto parent = output_path.parent_path();
        if (parent.empty() || !std::filesystem::is_directory(parent)) {
            throw PublicationError(
                "canonical artifact output parent directory does not exist");
        }
        for (int attempt = 0; attempt < 16; ++attempt) {
            const auto suffix = make_entropy_reference("");
            path_ = parent /
                ("." + output_path.filename().string() + ".stage-" + suffix);
            std::error_code error;
            if (std::filesystem::create_directory(path_, error)) {
                return;
            }
            if (error && error != std::errc::file_exists) {
                throw PublicationError(
                    "failed to reserve canonical artifact staging directory: " +
                    error.message());
            }
        }
        throw PublicationError(
            "failed to reserve a unique canonical artifact staging directory");
    }

    StagingDirectoryGuard(const StagingDirectoryGuard &) = delete;
    StagingDirectoryGuard &operator=(const StagingDirectoryGuard &) = delete;

    ~StagingDirectoryGuard() { (void)cleanup(); }

    const std::filesystem::path &path() const noexcept { return path_; }

    bool cleanup() noexcept {
        if (path_.empty()) {
            return true;
        }
        std::error_code permission_error;
        std::filesystem::permissions(
            path_, std::filesystem::perms::owner_all,
            std::filesystem::perm_options::add, permission_error);
        std::error_code cleanup_error;
        std::filesystem::remove_all(path_, cleanup_error);
        std::error_code existence_error;
        const bool remains = std::filesystem::exists(path_, existence_error);
        return !permission_error && !cleanup_error && !existence_error &&
            !remains;
    }

private:
    std::filesystem::path path_;
};

inline PublicationResult publish_canonical_artifact(
    const PublicationPlan &plan, const PublicationHooks &hooks = {}) {
    if (plan.artifact_path.empty() || plan.receipt_path.empty() ||
        plan.artifact_path == plan.receipt_path ||
        plan.artifact_path.filename().empty() ||
        plan.receipt_path.filename().empty()) {
        throw PublicationError(
            "canonical artifact publication paths are empty or ambiguous");
    }
    const auto parent = plan.artifact_path.parent_path();
    if (parent.empty() || plan.receipt_path.parent_path() != parent ||
        !std::filesystem::is_directory(parent)) {
        throw PublicationError(
            "canonical artifact and receipt require one existing parent directory");
    }
    if (!plan.validate) {
        throw PublicationError(
            "canonical artifact publication requires a reread validator");
    }
    if (detail::path_exists(plan.artifact_path) ||
        detail::path_exists(plan.receipt_path)) {
        throw PublicationError(
            "canonical artifact refuses to overwrite an existing artifact or receipt");
    }
    PublicationResult publication_result{plan.artifact_path,
                                         plan.receipt_path};
    plan.validate(plan.artifact_bytes, plan.receipt_bytes);

    StagingDirectoryGuard staging(plan.artifact_path);
    const auto staged_artifact =
        staging.path() / plan.artifact_path.filename();
    const auto staged_receipt =
        staging.path() / plan.receipt_path.filename();
    const auto artifact_owner = staging.path() / ".owner-artifact";
    const auto receipt_owner = staging.path() / ".owner-receipt";
    bool artifact_published = false;
    bool receipt_published = false;
    try {
        detail::write_binary_file(staged_artifact, plan.artifact_bytes);
        notify_publication_stage(hooks, PublicationStage::artifact_staged,
                                 staged_artifact, staged_receipt);
        const auto staged_artifact_bytes =
            detail::read_binary_file(staged_artifact);
        if (staged_artifact_bytes != plan.artifact_bytes) {
            throw PublicationError(
                "staged canonical artifact bytes changed before validation");
        }
        plan.validate(staged_artifact_bytes, plan.receipt_bytes);
        notify_publication_stage(hooks, PublicationStage::artifact_validated,
                                 staged_artifact, staged_receipt);

        // A hook is an injected failure/tamper seam. Re-read after every hook
        // that follows validation so a same-process mutation cannot ride the
        // previously validated byte snapshot into publication.
        const auto post_artifact_hook_bytes =
            detail::read_binary_file(staged_artifact);
        if (post_artifact_hook_bytes != plan.artifact_bytes) {
            throw PublicationError(
                "staged canonical artifact changed after validation");
        }
        plan.validate(post_artifact_hook_bytes, plan.receipt_bytes);

        detail::write_binary_file(staged_receipt, plan.receipt_bytes);
        notify_publication_stage(hooks, PublicationStage::receipt_staged,
                                 staged_artifact, staged_receipt);
        const auto staged_receipt_bytes =
            detail::read_binary_file(staged_receipt);
        if (staged_receipt_bytes != plan.receipt_bytes) {
            throw PublicationError(
                "staged canonical artifact receipt changed before validation");
        }
        const auto receipt_stage_artifact_bytes =
            detail::read_binary_file(staged_artifact);
        if (receipt_stage_artifact_bytes != plan.artifact_bytes) {
            throw PublicationError(
                "staged canonical artifact changed while staging its receipt");
        }
        plan.validate(receipt_stage_artifact_bytes, staged_receipt_bytes);
        notify_publication_stage(hooks, PublicationStage::receipt_validated,
                                 staged_artifact, staged_receipt);

        const auto protected_artifact_bytes =
            detail::read_binary_file(staged_artifact);
        const auto protected_receipt_bytes =
            detail::read_binary_file(staged_receipt);
        if (protected_artifact_bytes != plan.artifact_bytes ||
            protected_receipt_bytes != plan.receipt_bytes) {
            throw PublicationError(
                "staged canonical artifact or receipt changed after validation");
        }
        plan.validate(protected_artifact_bytes, protected_receipt_bytes);

        detail::make_publication_source_read_only(staged_artifact);
        detail::make_publication_source_read_only(staged_receipt);
        detail::publish_no_replace_hard_link(staged_artifact, artifact_owner);
        detail::publish_no_replace_hard_link(staged_receipt, receipt_owner);

        if (detail::path_exists(plan.artifact_path) ||
            detail::path_exists(plan.receipt_path)) {
            throw PublicationError(
                "canonical artifact destination appeared during staging");
        }
        notify_publication_stage(
            hooks, PublicationStage::before_artifact_publish,
            staged_artifact, staged_receipt);
        detail::require_owned_publication_alias(staged_artifact,
                                                artifact_owner);
        detail::require_owned_publication_alias(staged_receipt,
                                                receipt_owner);
        const auto owner_artifact_bytes =
            detail::read_binary_file(artifact_owner);
        const auto owner_receipt_prepublication_bytes =
            detail::read_binary_file(receipt_owner);
        if (owner_artifact_bytes != plan.artifact_bytes ||
            owner_receipt_prepublication_bytes != plan.receipt_bytes) {
            throw PublicationError(
                "canonical artifact staging owner changed before publication");
        }
        plan.validate(owner_artifact_bytes,
                      owner_receipt_prepublication_bytes);
        detail::publish_no_replace_hard_link(artifact_owner,
                                             plan.artifact_path);
        artifact_published = true;
        notify_publication_stage(hooks, PublicationStage::artifact_published,
                                 staged_artifact, staged_receipt);

        const auto published_artifact_bytes =
            detail::read_binary_file(plan.artifact_path);
        const auto owner_receipt_bytes =
            detail::read_binary_file(receipt_owner);
        if (published_artifact_bytes != plan.artifact_bytes ||
            owner_receipt_bytes != plan.receipt_bytes) {
            throw PublicationError(
                "published canonical artifact disagrees with staged intent");
        }
        plan.validate(published_artifact_bytes, owner_receipt_bytes);
        notify_publication_stage(
            hooks, PublicationStage::before_receipt_publish,
            staged_artifact, staged_receipt);

        detail::require_owned_publication_alias(staged_artifact,
                                                artifact_owner);
        detail::require_owned_publication_alias(staged_receipt,
                                                receipt_owner);
        detail::require_owned_publication_alias(plan.artifact_path,
                                                artifact_owner);
        const auto final_artifact_bytes =
            detail::read_binary_file(plan.artifact_path);
        const auto final_receipt_bytes =
            detail::read_binary_file(receipt_owner);
        if (final_artifact_bytes != plan.artifact_bytes ||
            final_receipt_bytes != plan.receipt_bytes) {
            throw PublicationError(
                "canonical artifact or receipt changed before completion publication");
        }
        plan.validate(final_artifact_bytes, final_receipt_bytes);
        detail::publish_no_replace_hard_link(receipt_owner,
                                             plan.receipt_path);
        receipt_published = true;

        // Receipt visibility is the sole completion transition and the last
        // fallible operation. Cleanup is deliberately nonthrowing afterward.
        (void)staging.cleanup();
        return publication_result;
    } catch (...) {
        bool rollback_ok = true;
        if (receipt_published) {
            bool receipt_rollback = detail::remove_if_owned_hard_link(
                receipt_owner, plan.receipt_path);
            if (!receipt_rollback) {
                receipt_rollback = detail::remove_if_owned_hard_link(
                    staged_receipt, plan.receipt_path);
            }
            rollback_ok = receipt_rollback && rollback_ok;
        }
        if (artifact_published) {
            bool artifact_rollback = detail::remove_if_owned_hard_link(
                artifact_owner, plan.artifact_path);
            if (!artifact_rollback) {
                artifact_rollback = detail::remove_if_owned_hard_link(
                    staged_artifact, plan.artifact_path);
            }
            rollback_ok = artifact_rollback && rollback_ok;
        }
        const bool cleanup_ok = staging.cleanup();
        if (!rollback_ok || !cleanup_ok) {
            throw PublicationError(
                "canonical artifact publication failed and owned output cleanup was incomplete");
        }
        throw;
    }
}

inline PublicationResult validate_published_canonical_artifact(
    const std::filesystem::path &artifact_path,
    const std::filesystem::path &receipt_path,
    const PublicationValidator &validator) {
    if (!validator) {
        throw PublicationError(
            "canonical artifact publication validation requires a validator");
    }
    const auto artifact_bytes = detail::read_binary_file(artifact_path);
    const auto receipt_bytes = detail::read_binary_file(receipt_path);
    validator(artifact_bytes, receipt_bytes);
    return {artifact_path, receipt_path};
}

enum class BundlePublicationStage {
    members_staged,
    members_validated,
    receipt_staged,
    receipt_validated,
    before_members_publish,
    member_published,
    before_receipt_publish,
};

struct BundlePublicationHooks {
    std::function<void(BundlePublicationStage,
                       const std::filesystem::path &,
                       const std::filesystem::path &)>
        on_stage;
};

using BundlePublicationValidator = std::function<void(
    const std::vector<std::pair<std::filesystem::path, std::string>> &,
    std::string_view)>;

struct BundlePublicationPlan {
    // Members are published in this exact order. Canonical APT v2 places all
    // content-addressed members first and manifest.ecsv last.
    std::vector<std::pair<std::filesystem::path, std::string>> members;
    std::filesystem::path receipt_path;
    std::string receipt_bytes;
    BundlePublicationValidator validate;
};

struct BundlePublicationResult {
    std::vector<std::filesystem::path> member_paths;
    std::filesystem::path receipt_path;
};

inline BundlePublicationResult publish_canonical_bundle(
    const BundlePublicationPlan &plan,
    const BundlePublicationHooks &hooks = {}) {
    if (plan.members.empty() || plan.receipt_path.empty() || !plan.validate) {
        throw PublicationError(
            "canonical bundle publication plan is incomplete");
    }
    const auto parent = plan.receipt_path.parent_path();
    if (parent.empty() || !std::filesystem::is_directory(parent)) {
        throw PublicationError(
            "canonical bundle publication parent does not exist");
    }
    std::set<std::filesystem::path> names;
    names.insert(plan.receipt_path.filename());
    for (const auto &[path, bytes] : plan.members) {
        (void)bytes;
        if (path.parent_path() != parent || path.filename().empty() ||
            !names.insert(path.filename()).second) {
            throw PublicationError(
                "canonical bundle member paths are not unique siblings");
        }
        if (detail::path_exists(path)) {
            throw PublicationError(
                "canonical bundle refuses to overwrite an existing member");
        }
    }
    if (detail::path_exists(plan.receipt_path)) {
        throw PublicationError(
            "canonical bundle refuses to overwrite an existing receipt");
    }
    plan.validate(plan.members, plan.receipt_bytes);
    BundlePublicationResult result;
    result.receipt_path = plan.receipt_path;
    for (const auto &[path, bytes] : plan.members) {
        (void)bytes;
        result.member_paths.push_back(path);
    }

    // Stage beside the bundle directory, never inside it. Thus receipt
    // visibility cannot race a guardian that rejects extra directory entries.
    StagingDirectoryGuard staging(parent);
    std::vector<std::filesystem::path> staged_members;
    std::vector<std::filesystem::path> owner_members;
    std::vector<std::filesystem::path> published_members;
    const auto staged_receipt =
        staging.path() / plan.receipt_path.filename();
    const auto receipt_owner = staging.path() / ".owner-receipt";
    bool receipt_published = false;
    try {
        for (std::size_t index = 0; index < plan.members.size(); ++index) {
            const auto staged =
                staging.path() / plan.members[index].first.filename();
            const auto owner =
                staging.path() / (".owner-member-" + std::to_string(index));
            detail::write_binary_file(staged, plan.members[index].second);
            staged_members.push_back(staged);
            owner_members.push_back(owner);
        }
        if (hooks.on_stage) {
            hooks.on_stage(BundlePublicationStage::members_staged,
                           staging.path(), staged_receipt);
        }
        const auto reread_members = [&]() {
            std::vector<std::pair<std::filesystem::path, std::string>> value;
            value.reserve(staged_members.size());
            for (std::size_t index = 0; index < staged_members.size(); ++index) {
                value.emplace_back(plan.members[index].first,
                                   detail::read_binary_file(
                                       staged_members[index]));
                if (value.back().second != plan.members[index].second) {
                    throw PublicationError(
                        "staged canonical bundle member changed");
                }
            }
            return value;
        };
        auto checked_members = reread_members();
        plan.validate(checked_members, plan.receipt_bytes);
        if (hooks.on_stage) {
            hooks.on_stage(BundlePublicationStage::members_validated,
                           staging.path(), staged_receipt);
        }
        checked_members = reread_members();
        plan.validate(checked_members, plan.receipt_bytes);

        detail::write_binary_file(staged_receipt, plan.receipt_bytes);
        if (hooks.on_stage) {
            hooks.on_stage(BundlePublicationStage::receipt_staged,
                           staging.path(), staged_receipt);
        }
        auto checked_receipt = detail::read_binary_file(staged_receipt);
        if (checked_receipt != plan.receipt_bytes) {
            throw PublicationError(
                "staged canonical bundle receipt changed");
        }
        checked_members = reread_members();
        plan.validate(checked_members, checked_receipt);
        if (hooks.on_stage) {
            hooks.on_stage(BundlePublicationStage::receipt_validated,
                           staging.path(), staged_receipt);
        }
        checked_members = reread_members();
        checked_receipt = detail::read_binary_file(staged_receipt);
        plan.validate(checked_members, checked_receipt);

        for (std::size_t index = 0; index < staged_members.size(); ++index) {
            detail::make_publication_source_read_only(staged_members[index]);
            detail::publish_no_replace_hard_link(staged_members[index],
                                                 owner_members[index]);
        }
        detail::make_publication_source_read_only(staged_receipt);
        detail::publish_no_replace_hard_link(staged_receipt, receipt_owner);
        if (hooks.on_stage) {
            hooks.on_stage(BundlePublicationStage::before_members_publish,
                           staging.path(), staged_receipt);
        }
        checked_members = reread_members();
        checked_receipt = detail::read_binary_file(staged_receipt);
        plan.validate(checked_members, checked_receipt);

        for (std::size_t index = 0; index < plan.members.size(); ++index) {
            detail::require_owned_publication_alias(staged_members[index],
                                                    owner_members[index]);
            detail::publish_no_replace_hard_link(owner_members[index],
                                                 plan.members[index].first);
            published_members.push_back(plan.members[index].first);
            if (hooks.on_stage) {
                hooks.on_stage(BundlePublicationStage::member_published,
                               plan.members[index].first, staged_receipt);
            }
        }
        std::vector<std::pair<std::filesystem::path, std::string>> final_members;
        final_members.reserve(plan.members.size());
        for (const auto &[path, expected] : plan.members) {
            auto value = detail::read_binary_file(path);
            if (value != expected) {
                throw PublicationError(
                    "published canonical bundle member changed");
            }
            final_members.emplace_back(path, std::move(value));
        }
        checked_receipt = detail::read_binary_file(receipt_owner);
        plan.validate(final_members, checked_receipt);
        if (hooks.on_stage) {
            hooks.on_stage(BundlePublicationStage::before_receipt_publish,
                           staging.path(), staged_receipt);
        }
        for (std::size_t index = 0; index < plan.members.size(); ++index) {
            detail::require_owned_publication_alias(plan.members[index].first,
                                                    owner_members[index]);
        }
        detail::require_owned_publication_alias(staged_receipt, receipt_owner);
        plan.validate(final_members, detail::read_binary_file(receipt_owner));
        detail::publish_no_replace_hard_link(receipt_owner,
                                             plan.receipt_path);
        receipt_published = true;

        // Receipt visibility is the sole success transition. No fallible
        // product mutation or validation occurs after this point.
        (void)staging.cleanup();
        return result;
    } catch (...) {
        bool rollback_ok = true;
        if (receipt_published) {
            rollback_ok = detail::remove_if_owned_hard_link(
                receipt_owner, plan.receipt_path);
        }
        for (std::size_t index = published_members.size(); index > 0; --index) {
            rollback_ok = detail::remove_if_owned_hard_link(
                              owner_members[index - 1],
                              published_members[index - 1]) &&
                rollback_ok;
        }
        const bool cleanup_ok = staging.cleanup();
        if (!rollback_ok || !cleanup_ok) {
            throw PublicationError(
                "canonical bundle publication failed and owned cleanup was incomplete");
        }
        throw;
    }
}

}  // namespace citlali::pipeline::canonical_artifact_publication
