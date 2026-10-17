#include "Tuning.hpp"

#include <stdio.h>

#ifdef ENABLE_TUNING

DynArray<TunableParameter> g_TunableParameters;

void PrintParametersForTuning()
{
    for (const TunableParameter& param : g_TunableParameters)
    {
        printf("%s, int, %g, %g, %g, %g, %g\n",
            param.m_name.CStr(),
            double(param.m_value),
            double(param.m_min),
            double(param.m_max),
            Max(0.5, double(param.m_max - param.m_min) / 20.0),
            0.002);
    }
}

#endif // ENABLE_TUNING