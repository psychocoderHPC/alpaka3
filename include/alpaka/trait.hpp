/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cstdint>

namespace alpaka
{
    namespace trait
    {
        template<typename T>
        struct GetDim
        {
            static constexpr uint32_t value = T::dim();
        };

        template<std::integral T>
        struct GetDim<T>
        {
            static constexpr uint32_t value = 1u;
        };

        template<typename T>
        constexpr uint32_t getDim_v = GetDim<T>::value;

        template<typename T>
        struct GetValueType
        {
            using type = typename T::value_type;
        };

        template<typename T>
        requires(std::is_fundamental_v<T>)
        struct GetValueType<T>
        {
            using type = T;
        };

        // resolve handles
        template<typename  T>
        requires requires(){ typename T::element_type; }
        struct GetValueType<T>
        {
            using type = typename GetValueType<typename T::element_type>::type;
        };

        template<typename T>
        using GetValueType_t = typename GetValueType<T>::type;

#if 0
        /** type to describe size
         *
         * ::type must be a scalar type
         *
         * @{
         */
        template<typename T>
        struct GetSizeType
        {
            using type = typename T::size_type;
        };

        template<typename T>
        requires(std::is_fundamental_v<T>)
        struct GetSizeType<T>
        {
            using type = size_t;
        };

        template<typename T>
        using GetSizeType_t = typename GetSizeType<T>::type;

        /** @} */

        /** type to describe extents
         *
         * type can be a alpaka::Vec
         *
         * @{
         */
        template<typename T>
        struct GetExtentType
        {
            using type = GetSizeType_t<T>;
        };

        template<typename T>
        requires(std::is_fundamental_v<T>)
        struct GetExtentType<T>
        {
            using type = size_t;
        };

        template<typename T>
        using GetExtentType_t = typename GetExtentType<T>::type;
        /** @} */
#endif
    } // namespace trait

    template<typename T>
    consteval uint32_t getDim(T const& any)
    {
        return trait::getDim_v<T>;
    }

} // namespace alpaka
