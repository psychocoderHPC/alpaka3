/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace alpaka::internal
{

    /** Convert a bool value into a mask type for SIMD
     *
     * @return
     */
    template<typename T>
    constexpr auto valueMaskCast(bool condition)
    {
        return condition;
    }

    /** specialization for 4 and 8 byte types
     *
     * @return value type where all bits are 1 if condition is true, else all bit are zero
     */
    template<typename T>
    requires(sizeof(T) == 4u || sizeof(T) == 8u)
    constexpr auto valueMaskCast(bool condition)
    {
        using ValueMaskType = std::conditional_t<sizeof(T) == 4u, uint32_t, uint64_t>;
        // if condition is true value will be 1 and negated to set all bits
        return -static_cast<ValueMaskType>(condition);
    }

    /** Check if a pointer is aligned to a specific byte boundary.
     *
     * @param alignment alignemnt in bytes
     * @return true if the pointer is aligned to aligment or larger, else false.
     */
    template<typename T>
    constexpr bool isAlignedPtr(T const* ptr, std::size_t alignment)
    {
        return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0u;
    }
} // namespace alpaka::internal
