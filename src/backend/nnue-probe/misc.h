#ifndef MISC_H
#define MISC_H

#include <inttypes.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <unistd.h>
#  include <sys/mman.h>
#endif

/*
Force inline
*/
#if defined (__GNUC__)
#   define INLINE  __inline __attribute__((always_inline))
#elif defined (_WIN32)
#   define INLINE  __forceinline
#else
#   define INLINE  __inline
#endif

/*
Intrinsic bsf
*/
#   if defined(__GNUC__)
#       define bsf(b) __builtin_ctzll(b)
#       define bsr(b) (63 - __builtin_clzll(b))
#   elif defined(_WIN32)
#       include <intrin.h>
        INLINE int bsf(uint64_t b) {
            unsigned long x;
            _BitScanForward64(&x, b);
            return (int) x;
        }
        INLINE int bsr(uint64_t b) {
            unsigned long x;
            _BitScanReverse64(&x, b);
            return (int) x;
        }
#   endif

#ifdef _WIN32

typedef HANDLE FD;
#define FD_ERR INVALID_HANDLE_VALUE
typedef HANDLE map_t;

#else /* Unix */

typedef int FD;
#define FD_ERR -1
typedef size_t map_t;

#endif

FD open_file(const char *name);
void close_file(FD fd);
size_t file_size(FD fd);
const void *map_file(FD fd, map_t *map);
void unmap_file(const void *data, map_t map);

INLINE uint32_t readu_le_u32(const void *p)
{
  const uint8_t *q = (const uint8_t*) p;
  return q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
}

INLINE uint16_t readu_le_u16(const void *p)
{
  const uint8_t *q = (const uint8_t*) p;
  return q[0] | (q[1] << 8);
}

#define clamp(a, b, c) ((a) < (b) ? (b) : (a) > (c) ? (c) : (a))

#endif
