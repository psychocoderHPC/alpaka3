/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 *
 * Bridges runtime accelerator instances to the trait-based warp intrinsics so kernels can call them without tags.
 * Exposes device-safe `alpaka::onAcc::warp::*` wrappers for ballots, shuffles, and lane queries.
 * Reuses the compile-time warp trait specialisations instead of duplicating backend-specific logic in kernels.
 * Supplies a uniform warp API across CUDA, HIP, SYCL, and host-emulation accelerators.
 *
 * Some example usages:
 * - consteval `alpaka::getWarpSize(api::Cuda{}, deviceKind::NvidiaGpu{})` for tag-driven compile-time logic.
 * - device-side `alpaka::onAcc::warp::getSize(acc)` to query the active warp inside a kernel.
 * - device-side `alpaka::onAcc::warp::shfl(acc, 42, 0u) == 42`
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/internal/warp.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>

namespace alpaka::onAcc::warp
{
    /** Return the bit-mask of active lanes for the warp associated with the accelerator.
     *
     * @return bit mask where the Nth bit is set to 1 if the corresponding thread is participating the call. The return
     * type can be 64bit or 32bit depending on the API.
     */
    template<alpaka::onAcc::concepts::Acc T_Acc>
    constexpr auto activemask(T_Acc const& acc) -> std::conditional_t<T_Acc::getWarpSize() <= 32u, uint32_t, uint64_t>
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::Activemask::Op<Acc, Api>{}(acc, Api{});
    }

    /** Return the lane index of the current thread within its warp. */
    constexpr uint32_t getLaneIdx(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::GetLanIdx::Op<Acc, Api>{}(acc, Api{});
    }

    /** Vote function returning true if all active lanes satisfy the predicate. */


    /** Evaluates predicate for all active threads of the warp
     *
     * It follows the logic of __all_sync(__activemask(), predicate) in CUDA but returns a boolean.
     *
     * Note:
     * * The programmer must ensure that all threads calling this function are executing
     *   the same line of code. In particular, it is not portable to write
     *   if(a) {all} else {all}.
     *
     * @param predicate The predicate value for current thread.
     * @return true if all threads predicate non zero, else false
     */
    constexpr bool all(alpaka::onAcc::concepts::Acc auto const& acc, int32_t predicate)
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::All::Op<Acc, Api>{}(acc, Api{}, predicate);
    }

    /** Evaluates predicate for all active threads of the warp.
     *
     * It follows the logic of __any_sync(__activemask(), predicate) in CUDA but returns a boolean.
     *
     * Note:
     * * The programmer must ensure that all threads calling this function are executing
     *   the same line of code. In particular, it is not portable to write
     *   if(a) {any} else {any}.
     *
     * @param predicate The predicate value for current thread.
     * @return true if at least one threads predicate is non zero, else false
     */
    constexpr bool any(alpaka::onAcc::concepts::Acc auto const& acc, int32_t predicate)
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::Any::Op<Acc, Api>{}(acc, Api{}, predicate);
    }

    /** Evaluates predicate for all non-exited threads in a warp and returns
     * a 32- or 64-bit unsigned integer (depending on the accelerator)
     * whose Nth bit is set if and only if predicate evaluates to non-zero
     * for the Nth thread of the warp and the Nth thread is active.
     *
     * It follows the logic of __ballot_sync(__activemask(), predicate) in CUDA.
     *
     * Note:
     * * The programmer must ensure that all threads calling this function are executing
     *   the same line of code. In particular, it is not portable to write
     *   if(a) {ballot} else {ballot}.
     *
     * @param predicate The predicate value for current thread.
     * @return bit mask where the Nth bit is set to 1 if the corresponding threads predicate was non zero. The return
     * type can be 64bit or 32bit depending on the API.
     */
    template<alpaka::onAcc::concepts::Acc T_Acc>
    constexpr auto ballot(T_Acc const& acc, int32_t predicate)
        -> std::conditional_t<T_Acc::getWarpSize() <= 32u, uint32_t, uint64_t>
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::Ballot::Op<Acc, Api>{}(acc, Api{}, predicate);
    }

    /** Return the warp size.
     *
     * A warp is a collection of threads which work in lock step (executing the same command).
     * The warp size can be larger than the number of threads executed in the kernel/ thread block.
     * @{
     */
    template<concepts::Acc T_Acc>
    constexpr uint32_t getSize()
    {
        return internal::getSize<T_Acc>();
    }

    template<concepts::Acc T_Acc>
    constexpr uint32_t getSize(T_Acc const& acc)
    {
        return T_Acc::getWarpSize();
    }

    /** @} */

    /** Exchange data between threads within a warp.
     *
     * Effectively executes:
     *
     *     __shared__ int32_t values[warpsize];
     *     values[threadIdx.x] = value;
     *     __syncthreads();
     *     return values[width*(threadIdx.x/width) + srcLane%width];
     *
     * However, it does not use shared memory.
     *
     *  Commonly used with width = warpsize (the default), (returns values[srcLane])
     *
     * This method supports to be called in diverging control flow branches if you only query values from threads
     * within the same branch path.
     *
     * @param  value   value to broadcast, only used if other thread is addressing the lane of this thread.
     * @param  srcLane source lane index within the group range [0; width).
     * @param  width   number of threads receiving a single value, must be a power of 2.
     * @return val from the thread index srcLane.
     */
    template<typename T, alpaka::onAcc::concepts::Acc T_Acc>
    constexpr T shfl(T_Acc const& acc, T const& value, uint32_t srcLane, uint32_t width = getSize<T_Acc>())
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::Shfl::Op<Acc, Api, T>{}(acc, Api{}, value, srcLane, width != 0u ? width : getSize<T_Acc>());
    }

    /** Exchange data between threads within a warp.
     *
     * It copies from a lane with higher ID relative to caller.
     * The lane ID is calculated by adding delta to the caller’s lane ID.
     *
     * Effectively executes:
     *
     *     __shared__ int32_t values[warpsize];
     *     values[threadIdx.x] = value;
     *     __syncthreads();
     *     return (threadIdx.x % width + delta < width) ? values[threadIdx.x + delta] : values[threadIdx.x];
     *
     * However, it does not use shared memory.
     *
     * Notes:
     * * The programmer must ensure that all threads calling this
     *   function (and the srcLane) are executing the same line of code.
     *   In particular it is not portable to write if(a) {shfl} else {shfl}.
     *
     * * Commonly used with width = warpsize (the default), (returns values[threadIdx.x+delta] if threadIdx.x+delta <
     * warpsize)
     *
     * * Width must be a power of 2.
     *
     * @param  value   value to broadcast
     * @param  delta  corresponds to the delta used to compute the lane ID
     * @param  width   size of the group participating in the shuffle operation
     * @return the value from the thread index lane ID + delta within the group build by width, else value.
     */
    template<typename T, alpaka::onAcc::concepts::Acc T_Acc>
    constexpr T shflDown(T_Acc const& acc, T const& value, uint32_t delta, uint32_t width = getSize<T_Acc>())
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return internal::ShflDown::Op<Acc, Api, T>{}(acc, Api{}, value, delta, width != 0u ? width : getSize<T_Acc>());
    }

#if 0

    namespace detail
    {
        template<typename T_CountVec, typename T_IdxVec>
        ALPAKA_FN_HOST_ACC constexpr std::uint32_t linearThreadIdx(
            T_CountVec const& threadCount,
            T_IdxVec const& threadIdx)
        {
            auto const linear = linearize(threadCount, threadIdx);
            return static_cast<std::uint32_t>(linear);
        }

        template<typename T_Api, typename T_DeviceKind>
        struct WarpFacade
        {
            T_Api api;
            T_DeviceKind device;
            std::uint32_t width;

            ALPAKA_FN_HOST_ACC constexpr std::uint32_t size() const
            {
                return width;
            }

            ALPAKA_FN_HOST_ACC constexpr std::uint64_t activemask() const
            {
                return alpaka::onAcc::warp::internal::activemask(api, device);
            }

            template<typename Predicate>
            ALPAKA_FN_HOST_ACC constexpr bool all(Predicate const& predicate) const
            {
                return alpaka::onAcc::warp::all(api, device, predicate);
            }

            template<typename Predicate>
            ALPAKA_FN_HOST_ACC constexpr bool any(Predicate const& predicate) const
            {
                return alpaka::onAcc::warp::any(api, device, predicate);
            }

            template<typename Predicate>
            ALPAKA_FN_HOST_ACC constexpr std::uint64_t ballot(Predicate const& predicate) const
            {
                return alpaka::onAcc::warp::ballot(api, device, predicate);
            }

            template<typename T_Value>
            ALPAKA_FN_HOST_ACC constexpr T_Value shfl(T_Value const& value, std::uint32_t srcLane) const
            {
                return alpaka::onAcc::warp::shfl(api, device, value, srcLane, width);
            }

            template<typename T_Value>
            ALPAKA_FN_HOST_ACC constexpr T_Value shflDown(T_Value const& value, std::uint32_t delta) const
            {
                return alpaka::onAcc::warp::shflDown(api, device, value, delta, width);
            }

            template<typename T_Value>
            ALPAKA_FN_HOST_ACC constexpr T_Value shflUp(T_Value const& value, std::uint32_t delta) const
            {
                return alpaka::onAcc::warp::shflUp(api, device, value, delta, width);
            }

            template<typename T_Value>
            ALPAKA_FN_HOST_ACC constexpr T_Value shflXor(T_Value const& value, std::uint32_t laneMask) const
            {
                return alpaka::onAcc::warp::shflXor(api, device, value, laneMask, width);
            }
        };
    } // namespace detail



    /** Return the warp index of the current thread within the block. */
    ALPAKA_FN_HOST_ACC constexpr std::uint32_t getWarpIdxInBlock(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        auto const& threadLayer = acc[alpaka::layer::thread];
        auto const linearIdx = detail::linearThreadIdx(threadLayer.count(), threadLayer.idx());
        auto const size = getSize(acc);
        return static_cast<std::uint32_t>(linearIdx / size);
    }

    /** Return the total number of warps required to cover the current block. */
    ALPAKA_FN_HOST_ACC constexpr std::uint32_t getNumWarps(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        auto const& threadLayer = acc[alpaka::layer::thread];
        auto const totalThreads = static_cast<std::uint32_t>(threadLayer.count().product());
        auto const size = getSize(acc);
        return static_cast<std::uint32_t>((totalThreads + size - 1u) / size);
    }

    /** True if the current lane is the first lane within the warp. */
    ALPAKA_FN_HOST_ACC constexpr bool isWarpLeader(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        return getLaneIdx(acc) == 0u;
    }


    /** True if all active lanes satisfy the predicate. */
    template<typename Predicate>
    ALPAKA_FN_HOST_ACC constexpr bool all(alpaka::onAcc::concepts::Acc auto const& acc, Predicate const& predicate)
    {
        return alpaka::onAcc::warp::all(acc.getApi(), acc.getDeviceKind(), predicate);
    }

    /** True if any active lane satisfies the predicate. */
    template<typename Predicate>
    ALPAKA_FN_HOST_ACC constexpr bool any(alpaka::onAcc::concepts::Acc auto const& acc, Predicate const& predicate)
    {
        return alpaka::onAcc::warp::any(acc.getApi(), acc.getDeviceKind(), predicate);
    }

    /** Bit-mask of lanes where the predicate evaluates to true. */
    template<typename Predicate>
    ALPAKA_FN_HOST_ACC constexpr std::uint64_t ballot(
        alpaka::onAcc::concepts::Acc auto const& acc,
        Predicate const& predicate)
    {
        return alpaka::onAcc::warp::ballot(acc.getApi(), acc.getDeviceKind(), predicate);
    }

    /** Broadcast the value from a specific source lane using the current warp width. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shfl(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t srcLane,
        std::uint32_t width)
    {
        return alpaka::onAcc::warp::shfl(acc.getApi(), acc.getDeviceKind(), value, srcLane, width);
    }

    /** Broadcast the value from a specific source lane using the configured warp size. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shfl(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t srcLane)
    {
        return shfl(acc, value, srcLane, getSize(acc));
    }

    /** Shift values toward higher lane indices. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflDown(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t delta,
        std::uint32_t width)
    {
        return alpaka::onAcc::warp::shflDown(acc.getApi(), acc.getDeviceKind(), value, delta, width);
    }

    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflDown(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t delta)
    {
        return shflDown(acc, value, delta, getSize(acc));
    }

    /** Shift values toward lower lane indices. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflUp(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t delta,
        std::uint32_t width)
    {
        return alpaka::onAcc::warp::shflUp(acc.getApi(), acc.getDeviceKind(), value, delta, width);
    }

    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflUp(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t delta)
    {
        return shflUp(acc, value, delta, getSize(acc));
    }

    /** Exchange values according to an XOR mask. */
    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflXor(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t laneMask,
        std::uint32_t width)
    {
        return alpaka::onAcc::warp::shflXor(acc.getApi(), acc.getDeviceKind(), value, laneMask, width);
    }

    template<typename T_Value>
    ALPAKA_FN_HOST_ACC constexpr T_Value shflXor(
        alpaka::onAcc::concepts::Acc auto const& acc,
        T_Value const& value,
        std::uint32_t laneMask)
    {
        return shflXor(acc, value, laneMask, getSize(acc));
    }

    /** Factory returning a lightweight warp helper bound to the accelerator. */
    ALPAKA_FN_HOST_ACC constexpr auto make(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        using ApiTag = ALPAKA_TYPEOF(acc.getApi());
        using DeviceTag = ALPAKA_TYPEOF(acc.getDeviceKind());
        return detail::WarpFacade<ApiTag, DeviceTag>{acc.getApi(), acc.getDeviceKind(), getSize(acc)};
    }
#endif
} // namespace alpaka::onAcc::warp
