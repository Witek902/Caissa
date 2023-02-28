#pragma once

#include "Common.hpp"

// enable search parameter tuning
// will expose all parameters defined with DEFINE_PARAM to UCI
//#define ENABLE_TUNING


#ifdef ENABLE_TUNING

#include <vector>

struct TunableParameter
{
    using Type = int32_t;

    TunableParameter(const char* name, Type* v)
        : m_name(name)
        , m_valuePtr(v)
    { }

    const char* m_name = nullptr;
    Type* m_valuePtr = nullptr;
};

extern std::vector<TunableParameter> g_TunableParameters;

void RegisterParameter(const char* name, int32_t* value);

template<typename Type>
struct TunableParameterWrapper
{
    explicit TunableParameterWrapper(const char* name, const Type v)
        : m_value(v)
    {
        RegisterParameter(name, &m_value);
    }

    INLINE operator Type () const { return m_value; }

    Type m_value = 0;
};

#define DEFINE_PARAM(Name, Value) \
    static TunableParameterWrapper<int32_t> Name(#Name, Value)

#else

#define DEFINE_PARAM(Name, Value) \
    static constexpr int32_t Name = Value

#endif // ENABLE_TUNING