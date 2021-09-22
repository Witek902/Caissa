#include "Tablebase.hpp"

#ifdef USE_TABLE_BASES

#include "tablebase/tbprobe.h"

#include <iostream>

void LoadTablebase(const char* path)
{
    if (tb_init(path))
    {
        std::cout << "Tablebase loaded successfully. Size = " << TB_LARGEST << std::endl;
    }
    else
    {
        std::cout << "Failed to load tablebase" << std::endl;
    }
}

void UnloadTablebase()
{
    tb_free();
}

#else // !USE_TABLE_BASES

void LoadTablebase(const char*) { }
void UnloadTablebase() { }

#endif // USE_TABLE_BASES
