/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 *
 * Defines the Alpaka3 warp traits for kernel calls for votes and shuffles.
 * Central helpers (all, any, ballot, shfl…) forward to trait::Op specializations.
 *
 * Dispatch now depends on api::X and deviceKind::Y tags, keeping the layer backend free.
 * Provides a single entry point that works for host and GPU backends alike.
 *
 * Legacy Alpaka preferred the dispatch through ConceptWarp implementations per accelerator and used
 * interface::ImplementationBase indirection to select the correct implementation.
 */

#pragma once

#include "alpaka/api/concepts/api.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>

namespace alpaka::onAcc::warp::internal
{
    template<concepts::Acc T_Acc>
    constexpr uint32_t getSize()
    {
        return T_Acc::getWarpSize();
    }

    /** Retrieve a bit-mask describing which warp lanes are active. */
    struct Activemask
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api api) const
            {
                static_assert(sizeof(T_Acc) && false, "Missing warp Activemask implementation for the accelerator.");
                return 0u;
            }
        };
    };

    struct GetLanIdx
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api api) const
            {
                static_assert(sizeof(T_Acc) && false, "Missing warp GetLanIdx implementation for the accelerator.");
                return 0u;
            }
        };
    };

    struct All
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr bool operator()(T_Acc const&, T_Api api, int32_t predicate) const
            {
                static_assert(sizeof(T_Acc) && false, "Missing warp All implementation for the accelerator.");
                return false;
            }
        };
    };

    struct Any
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr bool operator()(T_Acc const&, T_Api api, int32_t predicate) const
            {
                static_assert(sizeof(T_Acc) && false, "Missing warp Any implementation for the accelerator.");
                return false;
            }
        };
    };

    struct Ballot
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api api, int32_t predicate) const
            {
                static_assert(sizeof(T_Acc) && false, "Missing warp Ballot implementation for the accelerator.");
                return 0;
            }
        };
    };

    struct Shfl
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api, typename T>
        struct Op
        {
            constexpr T operator()(T_Acc const&, T_Api api, T const& value, uint32_t srcLane, uint32_t width) const
            {
                static_assert(sizeof(T_Acc) && false, "Missing warp Shfl implementation for the accelerator.");
                return T{};
            }
        };
    };

    struct ShflDown
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api, typename T>
        struct Op
        {
            constexpr T operator()(T_Acc const&, T_Api api, T const& value, uint32_t delta, uint32_t width) const
            {
                static_assert(sizeof(T_Acc) && false, "Missing warp ShflDown implementation for the accelerator.");
                return T{};
            }
        };
    };

#if 0

    /** Vote function returning true if all active lanes satisfy the predicate. */
    struct All
    {
        template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
        struct Op
        {
            template<typename T_Predicate>
            ALPAKA_FN_HOST_ACC constexpr bool operator()(T_Api const, T_DeviceKind const, T_Predicate const&) const
            {
                static_assert(sizeof(T_Api) && false, "Missing warp All implementation for API/device pair.");
                return false;
            }
        };
    };

    /** Vote function returning true if any active lane satisfies the predicate. */
    struct Any
    {
        template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
        struct Op
        {
            template<typename T_Predicate>
            ALPAKA_FN_HOST_ACC constexpr bool operator()(T_Api const, T_DeviceKind const, T_Predicate const&) const
            {
                static_assert(sizeof(T_Api) && false, "Missing warp Any implementation for API/device pair.");
                return false;
            }
        };
    };

    /** Vote function that returns a bit-mask of lanes where the predicate evaluates to true. */
    struct Ballot
    {
        template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
        struct Op
        {
            template<typename T_Predicate>
            ALPAKA_FN_HOST_ACC constexpr std::uint64_t operator()(T_Api const, T_DeviceKind const, T_Predicate const&)
                const
            {
                static_assert(sizeof(T_Api) && false, "Missing warp Ballot implementation for API/device pair.");
                return 0u;
            }
        };
    };

    /** Shuffle function selecting the value from a specific source lane. */
    struct Shfl
    {
        template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind, typename T_Value>
        struct Op
        {
            ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
                T_Api const,
                T_DeviceKind const,
                T_Value const& value,
                std::uint32_t,
                std::uint32_t) const
            {
                static_assert(sizeof(T_Api) && false, "Missing warp Shfl implementation for API/device pair.");
                return value;
            }
        };
    };

    /** Shuffle function shifting values toward higher lane indices. */
    struct ShflDown
    {
        template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind, typename T_Value>
        struct Op
        {
            ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
                T_Api const,
                T_DeviceKind const,
                T_Value const& value,
                std::uint32_t,
                std::uint32_t) const
            {
                static_assert(sizeof(T_Api) && false, "Missing warp ShflDown implementation for API/device pair.");
                return value;
            }
        };
    };

    /** Shuffle function shifting values toward lower lane indices. */
    struct ShflUp
    {
        template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind, typename T_Value>
        struct Op
        {
            ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
                T_Api const,
                T_DeviceKind const,
                T_Value const& value,
                std::uint32_t,
                std::uint32_t) const
            {
                static_assert(sizeof(T_Api) && false, "Missing warp ShflUp implementation for API/device pair.");
                return value;
            }
        };
    };

    /** Shuffle function exchanging values based on an XOR lane mask. */
    struct ShflXor
    {
        template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind, typename T_Value>
        struct Op
        {
            ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
                T_Api const,
                T_DeviceKind const,
                T_Value const& value,
                std::uint32_t,
                std::uint32_t) const
            {
                static_assert(sizeof(T_Api) && false, "Missing warp ShflXor implementation for API/device pair.");
                return value;
            }
        };
    };

    /** Warp-wide predicate: true if all active lanes satisfy the predicate. */
    ALPAKA_FN_HOST_ACC constexpr bool all(
        alpaka::concepts::Api auto const api,
        alpaka::concepts::DeviceKind auto const device,
        auto const& predicate)
    {
        return trait::All::Op<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(device)>{}(api, device, predicate);
    }

    /** Warp-wide predicate: true if any active lane satisfies the predicate. */
    ALPAKA_FN_HOST_ACC constexpr bool any(
        alpaka::concepts::Api auto const api,
        alpaka::concepts::DeviceKind auto const device,
        auto const& predicate)
    {
        return trait::Any::Op<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(device)>{}(api, device, predicate);
    }

    /** Warp-wide ballot producing a bit-mask of lanes where the predicate is true. */
    ALPAKA_FN_HOST_ACC constexpr std::uint64_t ballot(
        alpaka::concepts::Api auto const api,
        alpaka::concepts::DeviceKind auto const device,
        auto const& predicate)
    {
        return trait::Ballot::Op<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(device)>{}(api, device, predicate);
    }

    /** Return the value held by a specific source lane. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shfl(
        alpaka::concepts::Api auto const api,
        alpaka::concepts::DeviceKind auto const device,
        T_Value const& value,
        std::uint32_t srcLane,
        std::uint32_t width)
    {
        return trait::Shfl::Op<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(device), T_Value>{}(
            api,
            device,
            value,
            srcLane,
            width);
    }

    /** Shift values toward higher lane indices. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflDown(
        alpaka::concepts::Api auto const api,
        alpaka::concepts::DeviceKind auto const device,
        T_Value const& value,
        std::uint32_t delta,
        std::uint32_t width)
    {
        return trait::ShflDown::Op<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(device), T_Value>{}(
            api,
            device,
            value,
            delta,
            width);
    }

    /** Shift values toward lower lane indices. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflUp(
        alpaka::concepts::Api auto const api,
        alpaka::concepts::DeviceKind auto const device,
        T_Value const& value,
        std::uint32_t delta,
        std::uint32_t width)
    {
        return trait::ShflUp::Op<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(device), T_Value>{}(
            api,
            device,
            value,
            delta,
            width);
    }

    /** Exchange values based on an XOR lane mask. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflXor(
        alpaka::concepts::Api auto const api,
        alpaka::concepts::DeviceKind auto const device,
        T_Value const& value,
        std::uint32_t laneMask,
        std::uint32_t width)
    {
        return trait::ShflXor::Op<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(device), T_Value>{}(
            api,
            device,
            value,
            laneMask,
            width);
    }
#endif
} // namespace alpaka::onAcc::warp::internal
