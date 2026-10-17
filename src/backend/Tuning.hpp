#pragma once

#include "Common.hpp"

// enable search parameter tuning
// will expose all parameters defined with DEFINE_PARAM to UCI
//#define ENABLE_TUNING

// enable neural network last-layer tuning
//#define ENABLE_NET_L3_TUNING

#ifdef ENABLE_TUNING

#include "containers/DynArray.hpp"
#include "containers/FixedString.hpp"

struct TunableParameter
{
    using Type = int32_t;

    TunableParameter(const char* name, Type& v, Type minValue, Type maxValue)
        : m_name(name)
        , m_value(v)
        , m_min(minValue)
        , m_max(maxValue)
    { }

    FixedString<64> m_name;
    Type& m_value;
    Type m_min = 0;
    Type m_max = 0;
};

extern DynArray<TunableParameter> g_TunableParameters;

void PrintParametersForTuning();

template<typename Type>
struct TunableParameterWrapper
{
    explicit TunableParameterWrapper(const char* name, const Type value, const Type minValue, const Type maxValue)
        : m_value(value)
    {
        g_TunableParameters.EmplaceBack(name, m_value, minValue, maxValue);
    }

    INLINE operator Type () const { return m_value; }

    Type m_value = 0;
};

#define DEFINE_PARAM(Name, Value, MinValue, MaxValue) \
    static TunableParameterWrapper<int32_t> Name(#Name, Value, MinValue, MaxValue)

#else

#define DEFINE_PARAM(Name, Value, MinValue, MaxValue) \
    static constexpr int32_t Name = Value

#endif // ENABLE_TUNING