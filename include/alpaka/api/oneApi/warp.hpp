/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/oneApi/Api.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/internal/warp.hpp"

#include <algorithm>
#include <cstdint>

#if ALPAKA_LANG_ONEAPI
#    include <sycl/sycl.hpp>

namespace alpaka::onAcc::warp::internal
{
    // GPU back-ends use native SYCL subgroup operations.
    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Activemask::Op<T_Acc, api::OneApi>
    {
        auto operator()(T_Acc const& acc, api::OneApi) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();

            return getMask(sg);
        }

        static auto getMask(auto const subGroup)
        {
            auto sgMask = sycl::ext::oneapi::group_ballot(subGroup, true);

            constexpr auto const warpSize = T_Acc::getWarpSize();
            using ReturnType = std::conditional_t<warpSize <= 32, uint32_t, uint64_t>;
            ReturnType mask;
            sgMask.extract_bits(mask, 0u);
            return mask;
        };
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetLanIdx::Op<T_Acc, api::OneApi>
    {
        constexpr auto operator()(T_Acc const& acc, api::OneApi) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            // lane id within the warp subgroup
            return sg.get_local_id()[0];
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct All::Op<T_Acc, api::OneApi>
    {
        bool operator()(T_Acc const& acc, api::OneApi, int32_t predicate) const
        {
            using DeviceKind = ALPAKA_TYPEOF(acc[object::deviceKind]);
            if constexpr(DeviceKind{} == alpaka::deviceKind::amdGpu)
            {
                /* Workaround for AMD GPUs: Sycl is taking the results of the non active threads into account
                 * and therefore even if all participating threads have a true predicate the result will be false.
                 * We vote with ballot and mask the result with the active thread mask.
                 */
                sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
                auto activeMask = Activemask::Op<T_Acc, api::OneApi>::getMask(sg);
                auto sgMask = sycl::ext::oneapi::group_ballot(sg, predicate != 0);

                constexpr auto const warpSize = T_Acc::getWarpSize();
                using ReturnType = std::conditional_t<warpSize <= 32, uint32_t, uint64_t>;
                ReturnType predicateMask;
                sgMask.extract_bits(predicateMask, 0u);
                return activeMask & predicateMask == activeMask;
            }
            else
            {
                sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
                return sycl::all_of_group(sg, predicate != 0);
            }
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Any::Op<T_Acc, api::OneApi>
    {
        bool operator()(T_Acc const& acc, api::OneApi, int32_t predicate) const
        {
            using DeviceKind = ALPAKA_TYPEOF(acc[object::deviceKind]);
            if constexpr(DeviceKind{} == alpaka::deviceKind::amdGpu)
            {
                /* Workaround for AMD GPUs: Sycl is taking the results of non active threads into account
                 * and therefore even if all participating threads have a false predicate the result will be true.
                 * We vote with ballot and mask the result with the active thread mask.
                 */
                sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
                auto activeMask = Activemask::Op<T_Acc, api::OneApi>::getMask(sg);
                auto sgMask = sycl::ext::oneapi::group_ballot(sg, predicate != 0);

                constexpr auto const warpSize = T_Acc::getWarpSize();
                using ReturnType = std::conditional_t<warpSize <= 32, uint32_t, uint64_t>;
                ReturnType predicateMask;
                sgMask.extract_bits(predicateMask, 0u);
                return activeMask & predicateMask;
            }
            else
            {
                sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
                return sycl::any_of_group(sg, predicate != 0);
            }
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct Ballot::Op<T_Acc, api::OneApi>
    {
        auto operator()(T_Acc const& acc, api::OneApi, int32_t predicate) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            auto sgMask = sycl::ext::oneapi::group_ballot(sg, predicate != 0);

            constexpr auto const warpSize = T_Acc::getWarpSize();
            using ReturnType = std::conditional_t<warpSize <= 32, uint32_t, uint64_t>;
            ReturnType mask;
            sgMask.extract_bits(mask, 0u);
            return mask;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct Shfl::Op<T_Acc, api::OneApi, T>
    {
        constexpr T operator()(T_Acc const& acc, api::OneApi, T const& value, uint32_t srcLane, uint32_t width) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            uint32_t laneIdxInWarp = sg.get_local_id()[0];
            uint32_t partitionOffset = (laneIdxInWarp / width) * width;
            uint32_t srcInPartitionLaneIdx = partitionOffset + (srcLane % width);

            return sycl::select_from_group(sg, value, srcInPartitionLaneIdx);
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflDown::Op<T_Acc, api::OneApi, T>
    {
        constexpr T operator()(T_Acc const& acc, api::OneApi, T const& value, uint32_t delta, uint32_t width) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();

            uint32_t laneIdxInWarp = sg.get_local_id()[0];
            uint32_t groupEndIdx = (laneIdxInWarp / width + 1) * width;

            T result = sycl::shift_group_left(sg, value, delta);
            if(laneIdxInWarp + delta >= groupEndIdx)
                result = value;
            return result;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc, typename T>
    struct ShflUp::Op<T_Acc, api::OneApi, T>
    {
        constexpr T operator()(T_Acc const& acc, api::OneApi, T const& value, uint32_t delta, uint32_t width) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();

            uint32_t laneIdxInWarp = sg.get_local_id()[0];
            uint32_t groupStartIdx = (laneIdxInWarp / width) * width;

            T result = sycl::shift_group_right(sg, value, delta);
            if(laneIdxInWarp - groupStartIdx < delta)
                result = value;
            return result;
        }
    };
} // namespace alpaka::onAcc::warp::internal
#endif

#if 0
namespace alpaka::onAcc::warp::detail
{
#    if ALPAKA_LANG_ONEAPI
    /** Return the active sub-group for the calling work-item.
     * Legacy alpaka was using  warp.m_item_warp.get_sub_group()
     * but here warp operation moved into trait specializations keyed by tag-types, without access to the warp object.
     */
    ALPAKA_FN_ACC inline auto currentSubGroup()
    {
        /**  Prefer the new helper added in oneAPI 2025.1+, but keep
         the legacy free-function for older toolchains that have not adopted
         the `this_work_item` namespace yet.
*/
#        if defined(__INTEL_LLVM_COMPILER)
#            if __INTEL_LLVM_COMPILER >= 20'250'100
        return sycl::ext::oneapi::this_work_item::get_sub_group();
#            else
        return sycl::ext::oneapi::this_sub_group();
#            endif
#        else
        static_assert(sizeof(void*) == 0, "SYCL implementation without subgroup accessor not supported");
#        endif
    }

    /** Reduce the requested logical width to the available sub-group width. */
    ALPAKA_FN_ACC inline std::uint32_t clampWidth(std::uint32_t requested, std::uint32_t subgroupWidth)
    {
        if(requested == 0u)
        {
            return subgroupWidth;
        }
        return std::min(requested, subgroupWidth);
    }

    /** Compute the mask that covers all lanes in the current logical warp partition. */
    ALPAKA_FN_ACC inline std::uint64_t fullPartitionMask(std::uint32_t width)
    {
        if(width >= 64u)
        {
            return ~std::uint64_t{0};
        }
        return (std::uint64_t{1} << width) - std::uint64_t{1};
    }
#    endif
} // namespace alpaka::onAcc::warp::detail

namespace alpaka::onAcc::warp::trait
{

    template<>
    struct All::Op<api::OneApi, deviceKind::Cpu>
    {
        template<typename Predicate>
        ALPAKA_FN_HOST_ACC constexpr bool operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            Predicate const& predicate) const
        {
            return SingleThread::all(predicate);
        }
    };

    template<>
    struct Any::Op<api::OneApi, deviceKind::Cpu>
    {
        template<typename Predicate>
        ALPAKA_FN_HOST_ACC constexpr bool operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            Predicate const& predicate) const
        {
            return SingleThread::any(predicate);
        }
    };

    template<>
    struct Ballot::Op<api::OneApi, deviceKind::Cpu>
    {
        template<typename Predicate>
        ALPAKA_FN_HOST_ACC constexpr std::uint64_t operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            Predicate const& predicate) const
        {
            return SingleThread::ballot(predicate);
        }
    };

    template<typename T_Value>
    struct Shfl::Op<api::OneApi, deviceKind::Cpu, T_Value>
    {
        ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            T_Value const& value,
            std::uint32_t srcLane,
            std::uint32_t width) const
        {
            return SingleThread::shfl(value, srcLane, width);
        }
    };

    template<typename T_Value>
    struct ShflDown::Op<api::OneApi, deviceKind::Cpu, T_Value>
    {
        ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            return SingleThread::shflDown(value, delta, width);
        }
    };

    template<typename T_Value>
    struct ShflUp::Op<api::OneApi, deviceKind::Cpu, T_Value>
    {
        ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            return SingleThread::shflUp(value, delta, width);
        }
    };

    template<typename T_Value>
    struct ShflXor::Op<api::OneApi, deviceKind::Cpu, T_Value>
    {
        ALPAKA_FN_HOST_ACC constexpr T_Value operator()(
            api::OneApi const,
            deviceKind::Cpu const,
            T_Value const& value,
            std::uint32_t laneMask,
            std::uint32_t width) const
        {
            return SingleThread::shflXor(value, laneMask, width);
        }
    };

#    if ALPAKA_LANG_ONEAPI
    // GPU back-ends use native SYCL subgroup operations.
    template<alpaka::concepts::DeviceKind T_DeviceKind>
    struct Activemask::Op<api::OneApi, T_DeviceKind>
    {
        constexpr auto operator()(api::OneApi const, T_DeviceKind const) const
        {
            sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
            auto sgMask = sycl::ext::oneapi::group_ballot(sg, true);

            using ReturnType = std::conditional<>
            uint64_t mask;
            sgMask.extract_bits(mask,0u);
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind>
    struct All::Op<api::OneApi, T_DeviceKind>
    {
        template<typename Predicate>
        ALPAKA_FN_ACC bool operator()(api::OneApi const, T_DeviceKind const, Predicate const& predicate) const
        {
            auto const subGroup = detail::currentSubGroup();
            return sycl::all_of_group(subGroup, static_cast<bool>(predicate));
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind>
    struct Any::Op<api::OneApi, T_DeviceKind>
    {
        template<typename Predicate>
        ALPAKA_FN_ACC bool operator()(api::OneApi const, T_DeviceKind const, Predicate const& predicate) const
        {
            auto const subGroup = detail::currentSubGroup();
            return sycl::any_of_group(subGroup, static_cast<bool>(predicate));
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind>
    struct Ballot::Op<api::OneApi, T_DeviceKind>
    {
        template<typename Predicate>
        ALPAKA_FN_ACC std::uint64_t operator()(api::OneApi const, T_DeviceKind const, Predicate const& predicate) const
        {
            auto const subGroup = detail::currentSubGroup();
            auto const mask = sycl::ext::oneapi::group_ballot(subGroup, static_cast<bool>(predicate));
            std::uint64_t bits = 0u;
            mask.extract_bits(bits);
            return bits;
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind, typename T_Value>
    struct Shfl::Op<api::OneApi, T_DeviceKind, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::OneApi const,
            T_DeviceKind const,
            T_Value const& value,
            std::uint32_t srcLane,
            std::uint32_t width) const
        {
            auto const subGroup = detail::currentSubGroup();
            auto const subgroupWidth = static_cast<std::uint32_t>(subGroup.get_local_linear_range());
            auto const partitionWidth = detail::clampWidth(width, subgroupWidth);
            auto const laneId = static_cast<std::uint32_t>(subGroup.get_local_linear_id());
            auto const partitionBase = (laneId / partitionWidth) * partitionWidth;
            auto const targetLane = partitionBase + (srcLane % partitionWidth);
            return sycl::select_from_group(subGroup, value, targetLane);
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind, typename T_Value>
    struct ShflDown::Op<api::OneApi, T_DeviceKind, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::OneApi const,
            T_DeviceKind const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            auto const subGroup = detail::currentSubGroup();
            auto const subgroupWidth = static_cast<std::uint32_t>(subGroup.get_local_linear_range());
            auto const partitionWidth = detail::clampWidth(width, subgroupWidth);
            auto const laneId = static_cast<std::uint32_t>(subGroup.get_local_linear_id());
            auto const partitionBase = (laneId / partitionWidth) * partitionWidth;
            auto const partitionEnd = partitionBase + partitionWidth;
            auto result = sycl::shift_group_left(subGroup, value, delta);
            if(laneId + delta >= partitionEnd)
            {
                result = value;
            }
            return result;
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind, typename T_Value>
    struct ShflUp::Op<api::OneApi, T_DeviceKind, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::OneApi const,
            T_DeviceKind const,
            T_Value const& value,
            std::uint32_t delta,
            std::uint32_t width) const
        {
            auto const subGroup = detail::currentSubGroup();
            auto const subgroupWidth = static_cast<std::uint32_t>(subGroup.get_local_linear_range());
            auto const partitionWidth = detail::clampWidth(width, subgroupWidth);
            auto const laneId = static_cast<std::uint32_t>(subGroup.get_local_linear_id());
            auto const partitionBase = (laneId / partitionWidth) * partitionWidth;
            auto result = sycl::shift_group_right(subGroup, value, delta);
            if(laneId < partitionBase + delta)
            {
                result = value;
            }
            return result;
        }
    };

    template<alpaka::concepts::GpuType T_DeviceKind, typename T_Value>
    struct ShflXor::Op<api::OneApi, T_DeviceKind, T_Value>
    {
        ALPAKA_FN_ACC T_Value operator()(
            api::OneApi const,
            T_DeviceKind const,
            T_Value const& value,
            std::uint32_t laneMask,
            std::uint32_t width) const
        {
            auto const subGroup = detail::currentSubGroup();
            auto const subgroupWidth = static_cast<std::uint32_t>(subGroup.get_local_linear_range());
            auto const partitionWidth = detail::clampWidth(width, subgroupWidth);
            auto const laneId = static_cast<std::uint32_t>(subGroup.get_local_linear_id());
            auto const partitionBase = (laneId / partitionWidth) * partitionWidth;
            auto const relativeId = laneId - partitionBase;
            auto const targetRelative = relativeId ^ (laneMask % partitionWidth);
            if(targetRelative < partitionWidth)
            {
                auto const targetLane = partitionBase + targetRelative;
                return sycl::select_from_group(subGroup, value, targetLane);
            }
            return value;
        }
    };
#    endif
} // namespace alpaka::onAcc::warp::trait
#endif
