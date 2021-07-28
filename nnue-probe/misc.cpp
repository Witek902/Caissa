#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

#include "misc.h"

FD open_file(const char *name)
{
#ifndef _WIN32
  return open(name, O_RDONLY);
#else
  return CreateFileA(name, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
      FILE_FLAG_RANDOM_ACCESS, NULL);
#endif
}

void close_file(FD fd)
{
#ifndef _WIN32
  close(fd);
#else
  CloseHandle(fd);
#endif
}

size_t file_size(FD fd)
{
#ifndef _WIN32
  struct stat statbuf;
  fstat(fd, &statbuf);
  return statbuf.st_size;
#else
  DWORD sizeLow, sizeHigh;
  sizeLow = GetFileSize(fd, &sizeHigh);
  return ((uint64_t)sizeHigh << 32) | sizeLow;
#endif
}

const void *map_file(FD fd, map_t *map)
{
#ifndef _WIN32

  *map = file_size(fd);
  void *data = mmap(NULL, *map, PROT_READ, MAP_SHARED, fd, 0);
#ifdef MADV_RANDOM
  madvise(data, *map, MADV_RANDOM);
#endif
  return data == MAP_FAILED ? NULL : data;

#else

  DWORD sizeLow, sizeHigh;
  sizeLow = GetFileSize(fd, &sizeHigh);
  *map = CreateFileMapping(fd, NULL, PAGE_READONLY, sizeHigh, sizeLow, NULL);
  if (*map == NULL)
    return NULL;
  return MapViewOfFile(*map, FILE_MAP_READ, 0, 0, 0);

#endif
}

void unmap_file(const void *data, map_t map)
{
  if (!data) return;

#ifndef _WIN32

  munmap((void *)data, map);

#else

  UnmapViewOfFile(data);
  CloseHandle(map);

#endif
}
