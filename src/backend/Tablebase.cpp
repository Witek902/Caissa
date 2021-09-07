#include "Tablebase.hpp"

void LoadTablebase(const char* path)
{
#ifdef USE_TABLE_BASES
    if (tb_init(path))
    {
        std::cout << "Tablebase loaded successfully. Size = " << TB_LARGEST << std::endl;
    }
    else
    {
        std::cout << "Failed to load tablebase" << std::endl;
    }
#else
    (void*)path;
#endif // USE_TABLE_BASES
}

void UnloadTablebase()
{
#ifdef USE_TABLE_BASES
    tb_free();
#endif // USE_TABLE_BASES
}