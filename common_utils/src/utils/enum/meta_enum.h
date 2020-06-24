#pragma once
#include <array>
#include <optional>
#include <string_view>

namespace meta_enum {

// We use function traits to allow traits in different namespaces
template <typename T>
struct type_t {};
/// @brief Traits to test if enum has meta
template <typename T>
std::false_type enum_has_meta(type_t<T>);
/// @brief Traits to obtain the meta type of enum
template <typename T>
void enum_meta_type(type_t<T>);

/// @brief Class to store metadata of enum member
template <typename EnumType>
struct MetaEnumMember {
    EnumType value{};
    std::string_view name{};
    std::string_view string{};
    std::size_t index{0};
};

/// @brief Class to store metadata of enum type
template <typename EnumType, typename _UnderlyingType, std::size_t size>
struct MetaEnum {
    using UnderlyingType = _UnderlyingType;
    std::string_view name{};
    std::string_view string{};
    std::array<MetaEnumMember<EnumType>, size> members{};
};

namespace internal {

// facilities to parse the meta enum macro body
constexpr bool isNested(std::size_t brackets, bool quote) {
    return brackets != 0 || quote;
}

constexpr std::size_t nextEnumCommaOrEnd(std::size_t start,
                                         std::string_view enumString) {
    std::size_t brackets = 0; //()[]{}
    bool quote = false;       //""
    char lastChar = '\0';
    char nextChar = '\0';

    auto feedCounters = [&brackets, &quote, &lastChar, &nextChar](char c) {
        if (quote) {
            if (lastChar != '\\' && c == '"') // ignore " if they are
                                              // backslashed
                quote = false;
            return;
        }

        switch (c) {
        case '"':
            if (lastChar != '\\') // ignore " if they are backslashed
                quote = true;
            break;
        case '(':
        case '<':
            if (lastChar == '<' || nextChar == '<')
                break;
            [[fallthrough]];
        case '{':
            ++brackets;
            break;
        case ')':
        case '>':
            if (lastChar == '>' || nextChar == '>')
                break;
            [[fallthrough]];
        case '}':
            --brackets;
            break;
        default:
            break;
        }
    };

    std::size_t current = start;
    for (; current < enumString.size() &&
           (isNested(brackets, quote) || (enumString[current] != ','));
         ++current) {
        feedCounters(enumString[current]);
        lastChar = enumString[current];
        nextChar =
            current + 2 < enumString.size() ? enumString[current + 2] : '\0';
    }

    return current;
}

constexpr bool isAllowedIdentifierChar(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

constexpr std::string_view parseEnumMemberName(std::string_view memberString) {
    std::size_t nameStart = 0;
    while (!isAllowedIdentifierChar(memberString[nameStart])) {
        ++nameStart;
    }

    std::size_t nameSize = 0;

    while (isAllowedIdentifierChar(memberString[nameStart + nameSize])) {
        ++nameSize;
    }

    return std::string_view(memberString.data() + nameStart, nameSize);
}

template <typename EnumType, typename UnderlyingType, std::size_t size>
constexpr MetaEnum<EnumType, UnderlyingType, size>
parseMetaEnum(std::string_view name, std::string_view in,
              const std::array<EnumType, size> &values) {
    MetaEnum<EnumType, UnderlyingType, size> result;
    result.name = name;
    result.string = in;

    std::array<std::string_view, size> memberStrings;
    std::size_t amountFilled = 0;

    std::size_t currentStringStart = 0;

    while (amountFilled < size) {
        std::size_t currentStringEnd =
            nextEnumCommaOrEnd(currentStringStart + 1, in);
        std::size_t currentStringSize = currentStringEnd - currentStringStart;

        if (currentStringStart != 0) {
            ++currentStringStart;
            --currentStringSize;
        }

        memberStrings[amountFilled] =
            std::string_view(in.data() + currentStringStart, currentStringSize);
        ++amountFilled;
        currentStringStart = currentStringEnd;
    }

    for (std::size_t i = 0; i < memberStrings.size(); ++i) {
        result.members[i].name = parseEnumMemberName(memberStrings[i]);
        result.members[i].string = memberStrings[i];
        result.members[i].value = values[i];
        result.members[i].index = i;
    }

    return result;
}

template <typename EnumUnderlyingType>
struct IntWrapper {
    constexpr IntWrapper() : value(0) {}
    constexpr IntWrapper(EnumUnderlyingType in) : value(in), empty(false) {}
    constexpr IntWrapper &operator=(EnumUnderlyingType in) {
        value = in;
        empty = false;
        return *this;
    }
    EnumUnderlyingType value;
    bool empty = true;
    // allow composition of members in the definition
    constexpr friend IntWrapper operator|(const IntWrapper &lhs,
                                          const IntWrapper &rhs) {
        return IntWrapper(lhs.value | rhs.value);
    }
};

template <typename EnumType, typename EnumUnderlyingType, std::size_t size>
constexpr std::array<EnumType, size> resolveEnumValuesArray(
    const std::initializer_list<IntWrapper<EnumUnderlyingType>> &in) {
    std::array<EnumType, size> result{};

    EnumUnderlyingType nextValue = 0;
    for (std::size_t i = 0; i < size; ++i) {
        auto wrapper = *(in.begin() + i);
        EnumUnderlyingType newValue = wrapper.empty ? nextValue : wrapper.value;
        nextValue = newValue + 1;
        result[i] = static_cast<EnumType>(newValue);
    }

    return result;
}
} // namespace internal

} // namespace meta_enum

#define META_ENUM_IMPL(Type, UnderlyingType, ...)                              \
    struct Type##_meta {                                                       \
        constexpr static auto internal_size = []() constexpr {                 \
            using IntWrapperType =                                             \
                meta_enum::internal::IntWrapper<UnderlyingType>;               \
            IntWrapperType __VA_ARGS__;                                        \
            return std::initializer_list<IntWrapperType>{__VA_ARGS__}.size();  \
        };                                                                     \
        constexpr static auto meta = meta_enum::internal::parseMetaEnum<       \
            Type, UnderlyingType, internal_size()>(#Type, #__VA_ARGS__, []() { \
            using IntWrapperType =                                             \
                meta_enum::internal::IntWrapper<UnderlyingType>;               \
            IntWrapperType __VA_ARGS__;                                        \
            return meta_enum::internal::resolveEnumValuesArray<                \
                Type, UnderlyingType, internal_size()>({__VA_ARGS__});         \
        }());                                                                  \
        [[maybe_unused]] constexpr static auto to_name = [](Type e) {          \
            for (const auto &member : meta.members) {                          \
                if (member.value == e)                                         \
                    return member.name;                                        \
            }                                                                  \
            return std::string_view("__INVALID__");                            \
        };                                                                     \
        [[maybe_unused]] constexpr static auto from_name =                     \
            [](std::string_view s)                                             \
            -> std::optional<meta_enum::MetaEnumMember<Type>> {                \
            for (const auto &member : meta.members) {                          \
                if (member.name == s)                                          \
                    return member;                                             \
            }                                                                  \
            return std::nullopt;                                               \
        };                                                                     \
        [[maybe_unused]] constexpr static auto from_value =                    \
            [](Type v) -> std::optional<meta_enum::MetaEnumMember<Type>> {     \
            for (const auto &member : meta.members) {                          \
                if (member.value == v)                                         \
                    return member;                                             \
            }                                                                  \
            return std::nullopt;                                               \
        };                                                                     \
        [[maybe_unused]] constexpr static auto from_index =                    \
            [](std::size_t i) {                                                \
                std::optional<meta_enum::MetaEnumMember<Type>> result;         \
                if (i < meta.members.size())                                   \
                    result = meta.members[i];                                  \
                return result;                                                 \
            };                                                                 \
        constexpr static auto &name = meta.name;                               \
        constexpr static auto &string = meta.string;                           \
        constexpr static auto &members = meta.members;                         \
    }

// meta enum out of line
#define META_ENUM_(Type, UnderlyingType, ...)                                  \
    enum class Type : UnderlyingType { __VA_ARGS__ };                          \
    META_ENUM_IMPL(Type, UnderlyingType, __VA_ARGS__);                         \
    REGISTER_META_ENUM(Type)

// meta enum in line
#define META_ENUM(Type, UnderlyingType, ...)                                   \
    enum class Type : UnderlyingType { __VA_ARGS__ };                          \
    META_ENUM_IMPL(Type, UnderlyingType, __VA_ARGS__)

#define ENUM_META(Type) Type##_meta::meta

#define REGISTER_META_ENUM(Type)                                               \
    constexpr Type##_meta enum_meta_type(meta_enum::type_t<Type>);             \
    constexpr std::true_type enum_has_meta(meta_enum::type_t<Type>)
