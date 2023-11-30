#include "Tuning.hpp"

#ifdef ENABLE_TUNING

std::vector<TunableParameter> g_TunableParameters;

void PrintParametersForTuning()
{
    for (const TunableParameter& param : g_TunableParameters)
    {
        std::cout
            << param.m_name
            << ", " << "int"
            << ", " << double(param.m_value)
            << ", " << double(param.m_min)
            << ", " << double(param.m_max)
            << ", " << std::max(0.5, double(param.m_max - param.m_min) / 20.0)
            << ", " << 0.002
            << "\n";
    }
}

#endif // ENABLE_TUNING