#include "Tuning.hpp"

#ifdef ENABLE_TUNING

std::vector<TunableParameter> g_TunableParameters;

void RegisterParameter(const char* name, int32_t* value)
{
    g_TunableParameters.emplace_back(name, value);
}

#endif // ENABLE_TUNING