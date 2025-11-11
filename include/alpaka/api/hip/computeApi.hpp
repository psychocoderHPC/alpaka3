/* Copyright 2025 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/unifiedCudaHip/ComputeApi.hpp"
#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_HIP

namespace alpaka::onAcc::unifiedCudaHip::internal
{

}; // namespace alpaka::onAcc::unionmy_union
{
    template<>
    struct WarpSize::Get<alpaka::deviceKind::AmdGpu>
    {
        constexpr auto operator()() const
        {
            // HIP/ROCm may have a wavefront of 32 or 64 depending on the target device
#    if defined(__GFX9__)
            // GCN 5.0 and CDNA GPUs have a wavefront size of 64
            return CVec<uint32_t, 64u>{};
#    elif defined(__GFX10__) or defined(__GFX11__) or defined(__GFX12__)
            // RDNA GPUs have a wavefront size of 32
            return CVec<uint32_t, 32u>{};
#    else
            // Unknown AMD GPU architecture
#        ifdef ALPAKA_DEFAULT_HIP_WAVEFRONT_SIZE
            return CVec<uint32_t, ALPAKA_DEFAULT_HIP_WAVEFRONT_SIZE>{};
#        else
#            error The current AMD GPU architucture is not supported by this version of alpaka. You can define a default wavefront size setting the preprocessor macro ALPAKA_DEFAULT_HIP_WAVEFRONT_SIZE
            // return 32 instead of zero to avoid errors due to possible devision by zero, the code will thow at this
            // point anyway therefore we can return what we want
            return CVec<uint32_t, 32u>{};
#        endif
#    endif
        }
    };
} // namespace alpaka::onAcc::internalCompute

#endif
