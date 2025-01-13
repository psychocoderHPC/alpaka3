/* Copyright 2024 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Handle.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>

namespace alpaka::onHost
{
    namespace mem
    {
        //! Calculate the pitches purely from the extents.
        template<typename T_Elem, alpaka::concepts::Vector T_Vec>
        constexpr auto calculatePitchesFromExtents(T_Vec const& extent)
        {
            constexpr auto dim = T_Vec::dim();
            using type = typename T_Vec::type;
            auto pitchBytes = typename T_Vec::UniVec{};
            if constexpr(dim > 0)
                pitchBytes.back() = static_cast<type>(sizeof(T_Elem));
            if constexpr(dim > 1)
                for(type i = dim - 1; i > 0; i--)
                    pitchBytes[i - 1] = extent[i] * pitchBytes[i];
            return pitchBytes;
        }

        //! Calculate the pitches purely from the extents.
        template<typename T_Elem, alpaka::concepts::Vector T_Vec>
        requires(T_Vec::dim() >= 2)
        constexpr auto calculatePitches(T_Vec const& extent, typename T_Vec::type const& rowPitchBytes)
        {
            constexpr auto dim = T_Vec::dim();
            using type = typename T_Vec::type;
            auto pitchBytes = typename T_Vec::UniVec{};
            pitchBytes.back() = static_cast<type>(sizeof(T_Elem));
            if constexpr(dim > 1)
                pitchBytes[dim - 2u] = rowPitchBytes;
            if constexpr(dim > 2)
                for(type i = dim - 2; i > 0; i--)
                    pitchBytes[i - 1] = extent[i] * pitchBytes[i];
            return pitchBytes;
        }
    } // namespace mem

    template<typename T_PtrType, alpaka::concepts::Vector T_Extents, alpaka::concepts::Vector T_Pitches>
    struct PitchedPtr
    {
    public:
        PitchedPtr(
            T_PtrType data,
            T_Extents const& extents,
            T_Pitches const& pitches,
            std::function<void(T_PtrType)> deleter = [](T_PtrType) {})
            : m_data(data)
            , m_extents(extents)
            , m_pitches(pitches)
            , m_deleter(deleter)
        {
        }

        template<typename T_DataContainer>
        PitchedPtr(
            T_DataContainer && dataContainer,
            std::function<void(T_PtrType)> deleter = [](T_PtrType) {})
            : m_data(onHost::data(dataContainer))
            , m_extents(onHost::getExtents(dataContainer))
            , m_pitches(onHost::getPitches(dataContainer))
            , m_deleter(deleter)
        {
        }

        PitchedPtr(PitchedPtr const&) = default;
        PitchedPtr(PitchedPtr&&) = default;
        PitchedPtr& operator=(PitchedPtr const&) = default;
        PitchedPtr& operator=(PitchedPtr&&) = default;

        ~PitchedPtr()
        {
        }

        using value_type = std::remove_pointer_t<T_PtrType>;

        T_Pitches getPitches() const
        {
            return m_pitches;
        }

        T_Extents getExtents() const
        {
            return m_extents;
        }

        T_PtrType data() const
        {
            return m_data;
        }

        T_PtrType data()
        {
            return m_data;
        }

        // frees the memory hold by the pointer
        void free()
        {
            // NOTE: m_pMem is allowed to be a nullptr here.
            m_deleter(m_data);
        }

    private:
        friend struct internal::Data;

        void _()
        {
        }

        T_PtrType m_data;
        T_Extents m_extents;
        T_Pitches m_pitches;
        std::function<void(T_PtrType)> m_deleter;

        friend struct alpaka::internal::GetApi;
    };

    template<
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Vector T_Pitches,
        typename T_DeleterFn>
    PitchedPtr(T_Type const, T_Extents, T_Pitches, T_DeleterFn) -> PitchedPtr<T_Type, T_Extents, T_Pitches>;

    template<typename T_DataContainer>
    PitchedPtr(T_DataContainer) -> PitchedPtr<
        ALPAKA_TYPEOF(onHost::data(std::declval<T_DataContainer>())),
        ALPAKA_TYPEOF(onHost::getExtents(std::declval<T_DataContainer>())),
        ALPAKA_TYPEOF(onHost::getPitches(std::declval<T_DataContainer>()))>;
} // namespace alpaka::onHost

namespace alpaka::trait
{
    template<typename T_Type, alpaka::concepts::Vector T_Extents, alpaka::concepts::Vector T_Pitches>
    struct GetValueType<onHost::PitchedPtr<T_Type, T_Extents, T_Pitches>>
    {
        using type = std::remove_pointer_t<T_Type>;
    };

    template<typename T_Type, alpaka::concepts::Vector T_Extents, alpaka::concepts::Vector T_Pitches>
    struct GetDim<onHost::PitchedPtr<T_Type, T_Extents, T_Pitches>>
    {
        static constexpr uint32_t value = getDim_v<T_Extents>;
    };
#if 0
    template<typename T_Type, alpaka::concepts::Vector T_Extents, alpaka::concepts::Vector T_Pitches>
    struct GetExtentType<onHost::PitchedPtr<T_Type, T_Extents, T_Pitches>>
    {
        using type = T_Extents;
    };

    template<typename T_Type, alpaka::concepts::Vector T_Extents, alpaka::concepts::Vector T_Pitches>
    struct GetSizeType<onHost::PitchedPtr<T_Type, T_Extents, T_Pitches>>
    {
        using type = trait::GetSizeType_t<trait::GetExtentType_t<T_Extents>>;
    };
#endif
} // namespace alpaka::trait
