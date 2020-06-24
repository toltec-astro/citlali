#pragma once

/*
    Bitmask
    =======

    A generic implementation of the BitmaskType C++ concept
    http://en.cppreference.com/w/cpp/concept/BitmaskType

    Version: 1.1.2

    Latest version and documentation:
        https://github.com/oliora/bitmask

    Copyright (c) 2016-2017 Andrey Upadyshev (oliora@gmail.com)

    Distributed under the Boost Software License, Version 1.0.
    See http://www.boost.org/LICENSE_1_0.txt

    Changes history
    ---------------
    v1.1.2:
        - Fix: Can not define bitmask for a class local enum (https://github.com/oliora/bitmask/issues/3)
    v1.1.1:
        - Added missed `<limits>` header
        - README and code comments updated
    v1.1:
        - Change namespace from `boost` to `bitmask`
        - Add CMake package
          https://github.com/oliora/bitmask/issues/1
    v1.0:
        - Initial release

    Zhiyuan's fork
    --------------
    Allow in-class use
 */

#include <type_traits>
#include <functional>  // for std::hash
#include <limits>  // for std::numeric_limits
#include <cassert>


namespace bitmask {

    // We use function traits to allow traits in different namespaces
    template <typename T> struct type_t {};
    // Traits to return the value mask
    template <typename T> std::false_type value_mask_traits(type_t<T>);

    namespace bitmask_detail {
        using std::void_t;
        using std::underlying_type;
        using std::underlying_type_t;

        template<class T, T MaxElement = T::_bitmask_max_element>
        struct mask_from_max_element
        {
            static constexpr underlying_type_t<T> max_element_value_ =
                static_cast<underlying_type_t<T>>(MaxElement);

            static_assert(max_element_value_ >= 0,
                          "Max element is negative");

            // If you really have to define a bitmask that uses the highest bit of signed type (i.e. the sign bit) then
            // define the value mask rather than the max element.
            static_assert(max_element_value_ <= (std::numeric_limits<typename std::underlying_type<T>::type>::max() >> 1) + 1,
                          "Max element is greater than the underlying type's highest bit");

            // `((value - 1) << 1) + 1` is used rather that simpler `(value << 1) - 1`
            // because latter overflows in case if `value` is the highest bit of the underlying type.
            static constexpr underlying_type_t<T> value =
                max_element_value_ ? ((max_element_value_ - 1) << 1) + 1 : 0;
        };

        template<class, class = void_t<>>
        struct has_max_element : std::false_type {};

        template<class T>
        struct has_max_element<T, void_t<decltype(T::_bitmask_max_element)>> : std::true_type {};

#if !defined _MSC_VER
        template<class, class = void_t<>>
        struct has_value_mask : std::false_type {};

        template<class T>
        struct has_value_mask<T, void_t<decltype(T::_bitmask_value_mask)>> : std::true_type {};
#else
        // MS Visual Studio 2015 (even Update 3) has weird support for expressions SFINAE
        // so I can't get a real check for `has_value_mask` to compile.
        template<class T>
        struct has_value_mask: std::integral_constant<bool, !has_max_element<T>::value> {};
#endif

        template<class T>
        struct is_valid_enum_definition : std::integral_constant<bool,
            !(has_value_mask<T>::value && has_max_element<T>::value)> {};

        template<class, class = void>
        struct enum_mask;

        template<class T>
        struct enum_mask<T, typename std::enable_if<has_max_element<T>::value>::type>
            : std::integral_constant<underlying_type_t<T>, mask_from_max_element<T>::value> {};

        template<class T>
        struct enum_mask<T, typename std::enable_if<has_value_mask<T>::value>::type>
            : std::integral_constant<underlying_type_t<T>, static_cast<underlying_type_t<T>>(T::_bitmask_value_mask)> {};

        template<class Assert>
        inline void constexpr_assert_failed(Assert&& a) noexcept { a(); }

        // When evaluated at compile time emits a compilation error if condition is not true.
        // Invokes the standard assert at run time.
        #define bitmask_constexpr_assert(cond) \
            ((void)((cond) ? 0 : (bitmask::bitmask_detail::constexpr_assert_failed([](){ assert(!#cond);}), 0)))

        template<class T>
        inline constexpr T checked_value(T value, T mask)
        {
            return bitmask_constexpr_assert((value & ~mask) == 0), value;
        }
    }

//     template<class T>
//     inline constexpr bitmask_detail::underlying_type_t<T> get_enum_mask(const T&) noexcept
//     {
//         static_assert(bitmask_detail::is_valid_enum_definition<T>::value,
//                       "Both of _bitmask_max_element and _bitmask_value_mask are specified");
//         return bitmask_detail::enum_mask<T>::value;
//     }
//
//
    template<class T>
    class bitmask
    {
    public:
        using value_type = T;
        using underlying_type = bitmask_detail::underlying_type_t<T>;

        static constexpr underlying_type mask_value = decltype(value_mask_traits(type_t<value_type>{}))::value;
                //get_enum_mask(static_cast<value_type>(0));

        constexpr bitmask() noexcept = default;
        constexpr bitmask(std::nullptr_t) noexcept: m_bits{0} {}

        constexpr bitmask(value_type value) noexcept
        : m_bits{bitmask_detail::checked_value(static_cast<underlying_type>(value), mask_value)} {}

        constexpr underlying_type bits() const noexcept { return m_bits; }

        constexpr explicit operator value_type () const noexcept { return static_cast<T>(bits()); }

        constexpr explicit operator bool() const noexcept { return bits(); }

        constexpr bitmask operator ~ () const noexcept
        {
            return bitmask{std::true_type{}, ~m_bits & mask_value};
        }

        constexpr bitmask operator & (const bitmask& r) const noexcept
        {
            return bitmask{std::true_type{}, m_bits & r.m_bits};
        }

        constexpr bitmask operator | (const bitmask& r) const noexcept
        {
            return bitmask{std::true_type{}, m_bits | r.m_bits};
        }

        constexpr bitmask operator ^ (const bitmask& r) const noexcept
        {
            return bitmask{std::true_type{}, m_bits ^ r.m_bits};
        }

        bitmask& operator |= (const bitmask& r) noexcept
        {
            m_bits |= r.m_bits;
            return *this;
        }

        bitmask& operator &= (const bitmask& r) noexcept
        {
            m_bits &= r.m_bits;
            return *this;
        }

        bitmask& operator ^= (const bitmask& r) noexcept
        {
            m_bits ^= r.m_bits;
            return *this;
        }

    private:
        template<class U>
        constexpr bitmask(std::true_type, U bits) noexcept
        : m_bits(static_cast<underlying_type>(bits)) {}

        underlying_type m_bits = 0;
    };

    template<class T>
    inline constexpr bitmask<T>
    operator & (T l, const bitmask<T>& r) noexcept { return r & l; }

    template<class T>
    inline constexpr bitmask<T>
    operator | (T l, const bitmask<T>& r) noexcept { return r | l; }

    template<class T>
    inline constexpr bitmask<T>
    operator ^ (T l, const bitmask<T>& r) noexcept { return r ^ l; }

    template<class T>
    inline constexpr bool
    operator != (const bitmask<T>& l, const bitmask<T>& r) noexcept { return l.bits() != r.bits(); }

    template<class T>
    inline constexpr bool
    operator == (const bitmask<T>& l, const bitmask<T>& r) noexcept { return ! operator != (l, r); }

    template<class T>
    inline constexpr bool
    operator != (T l, const bitmask<T>& r) noexcept { return static_cast<bitmask_detail::underlying_type_t<T>>(l) != r.bits(); }

    template<class T>
    inline constexpr bool
    operator == (T l, const bitmask<T>& r) noexcept { return ! operator != (l, r); }

    template<class T>
    inline constexpr bool
    operator != (const bitmask<T>& l, T r) noexcept { return l.bits() != static_cast<bitmask_detail::underlying_type_t<T>>(r); }

    template<class T>
    inline constexpr bool operator == (const bitmask<T>& l, T r) noexcept { return ! operator != (l, r); }

    template<class T>
    inline constexpr bool
    operator != (const bitmask_detail::underlying_type_t<T>& l, const bitmask<T>& r) noexcept { return l != r.bits(); }

    template<class T>
    inline constexpr bool
    operator == (const bitmask_detail::underlying_type_t<T>& l, const bitmask<T>& r) noexcept { return ! operator != (l, r); }

    template<class T>
    inline constexpr bool
    operator != (const bitmask<T>& l, const bitmask_detail::underlying_type_t<T>& r) noexcept { return l.bits() != r; }

    template<class T>
    inline constexpr bool
    operator == (const bitmask<T>& l, const bitmask_detail::underlying_type_t<T>& r) noexcept { return ! operator != (l, r); }

    // Allow `bitmask` to be be used as a map key
    template<class T>
    inline constexpr bool
    operator < (const bitmask<T>& l, const bitmask<T>& r) noexcept { return l.bits() < r.bits(); }

    template<class T>
    inline constexpr bitmask_detail::underlying_type_t<T>
    bits(const bitmask<T>& bm) noexcept { return bm.bits(); }
}


namespace std
{
    template<class T>
    struct hash<bitmask::bitmask<T>>
    {
        constexpr std::size_t operator() (const bitmask::bitmask<T>& op) const noexcept
        {
            using ut = typename bitmask::bitmask<T>::underlying_type;
            return std::hash<ut>{}(op.bits());
        }
    };
}

// Implementation detail macros
#define BITMASK_DETAIL_CONCAT_IMPL(a, b) a##b
#define BITMASK_DETAIL_CONCAT(a, b) BITMASK_DETAIL_CONCAT_IMPL(a, b)

#define BITMASK_DETAIL_DEFINE_OPS(value_type) \
    [[maybe_unused]] inline constexpr bitmask::bitmask<value_type> operator & (value_type l, value_type r) noexcept { return bitmask::bitmask<value_type>{l} & r; }  \
    [[maybe_unused]] inline constexpr bitmask::bitmask<value_type> operator | (value_type l, value_type r) noexcept { return bitmask::bitmask<value_type>{l} | r; }  \
    [[maybe_unused]] inline constexpr bitmask::bitmask<value_type> operator ^ (value_type l, value_type r) noexcept { return bitmask::bitmask<value_type>{l} ^ r; }  \
    [[maybe_unused]] inline constexpr bitmask::bitmask<value_type> operator ~ (value_type op) noexcept { return ~bitmask::bitmask<value_type>{op}; }                 \
    [[maybe_unused]] inline constexpr bitmask::bitmask<value_type>::underlying_type bits(value_type op) noexcept { return bitmask::bitmask<value_type>{op}.bits(); }

#define BITMASK_DETAIL_DEFINE_OPS_INLINE(value_type) \
    [[maybe_unused]] friend inline constexpr bitmask::bitmask<value_type> operator & (value_type l, value_type r) noexcept { return bitmask::bitmask<value_type>{l} & r; }  \
    [[maybe_unused]] friend inline constexpr bitmask::bitmask<value_type> operator | (value_type l, value_type r) noexcept { return bitmask::bitmask<value_type>{l} | r; }  \
    [[maybe_unused]] friend inline constexpr bitmask::bitmask<value_type> operator ^ (value_type l, value_type r) noexcept { return bitmask::bitmask<value_type>{l} ^ r; }  \
    [[maybe_unused]] friend inline constexpr bitmask::bitmask<value_type> operator ~ (value_type op) noexcept { return ~bitmask::bitmask<value_type>{op}; }                 \
    [[maybe_unused]] friend inline constexpr bitmask::bitmask<value_type>::underlying_type bits(value_type op) noexcept { return bitmask::bitmask<value_type>{op}.bits(); }

#define BITMASK_DETAIL_DEFINE_VALUE_MASK(value_type, value_mask) \
    std::integral_constant<bitmask::bitmask_detail::underlying_type_t<value_type>, value_mask> value_mask_traits(bitmask::type_t<value_type>);

#define BITMASK_DETAIL_DEFINE_MAX_ELEMENT(value_type, max_element) \
    bitmask::bitmask_detail::mask_from_max_element<value_type, value_type::max_element> value_mask_traits(bitmask::type_t<value_type>);

// Public macros

// Defines missing operations for a bit-mask elements enum 'value_type'
// Value mask is taken from 'value_type' definition. One should has either
// '_bitmask_value_mask' or '_bitmask_max_element' element defined.
#define BITMASK_DEFINE(value_type) \
    BITMASK_DETAIL_DEFINE_OPS(value_type)

// Defines missing operations and a value mask for
// a bit-mask elements enum 'value_type'
#define BITMASK_DEFINE_VALUE_MASK(value_type, value_mask) \
    BITMASK_DETAIL_DEFINE_VALUE_MASK(value_type, value_mask) \
    BITMASK_DETAIL_DEFINE_OPS(value_type)

// Defines missing operations and a value mask based on max element for
// a bit-mask elements enum 'value_type'
#define BITMASK_DEFINE_MAX_ELEMENT(value_type, max_element) \
    BITMASK_DETAIL_DEFINE_MAX_ELEMENT(value_type, max_element) \
    BITMASK_DETAIL_DEFINE_OPS(value_type)
