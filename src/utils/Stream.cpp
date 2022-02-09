#include "Stream.hpp"

MemoryInputStream::MemoryInputStream(const std::vector<uint8_t>& buffer)
    : mBuffer(buffer)
    , mPosition(0)
{
}

uint64_t MemoryInputStream::GetSize()
{
    return mBuffer.size();
}

bool MemoryInputStream::Read(void* data, size_t size)
{
    if (size > 0)
    {
        ASSERT(data);

        if (mPosition + size > mBuffer.size())
        {
            return false;
        }

        memcpy(data, mBuffer.data() + mPosition, size);

        mPosition += size;
    }

    return true;
}

//////////////////////////////////////////////////////////////////////////

MemoryOutputStream::MemoryOutputStream(std::vector<uint8_t>& buffer)
    : mBuffer(buffer)
{
}

uint64_t MemoryOutputStream::GetSize()
{
    return mBuffer.size();
}

bool MemoryOutputStream::Write(const void* data, size_t size)
{
    if (size > 0)
    {
        ASSERT(data);

        const size_t prevSize = mBuffer.size();

        mBuffer.resize(mBuffer.size() + size);
        memcpy(mBuffer.data() + prevSize, data, size);
    }
    return true;
}

//////////////////////////////////////////////////////////////////////////

FileInputStream::FileInputStream(const char* filePath)
{
    mFile = fopen(filePath, "rb");
    if (!mFile)
    {
        perror(filePath);
        return;
    }
}

bool FileInputStream::IsOpen() const
{
    return mFile != nullptr;
}

uint64_t FileInputStream::GetSize()
{
    // TODO
    const long originalPos = ftell(mFile);

    fseek(mFile, 0, SEEK_END);
    const long size = ftell(mFile);
    fseek(mFile, originalPos, SEEK_SET);

    return size;
}

bool FileInputStream::Read(void* data, size_t size)
{
    return fread(data, size, 1, mFile) == 1;
}

//////////////////////////////////////////////////////////////////////////

FileOutputStream::FileOutputStream(const char* filePath)
{
    mFile = fopen(filePath, "wb");
    if (!mFile)
    {
        perror(filePath);
        return;
    }
}

bool FileOutputStream::IsOpen() const
{
    return mFile != nullptr;
}

bool FileOutputStream::Seek(uint64_t pos)
{
    if (mFile != nullptr)
    {
        fseek(mFile, (long)pos, SEEK_SET);
    }

    return false;
}

uint64_t FileOutputStream::GetSize()
{
    const long originalPos = ftell(mFile);

    fseek(mFile, 0, SEEK_END);
    const long size = ftell(mFile);
    fseek(mFile, originalPos, SEEK_SET);

    return size;
}

bool FileOutputStream::Write(const void* data, size_t size)
{
    return fwrite(data, size, 1, mFile) == 1;
}