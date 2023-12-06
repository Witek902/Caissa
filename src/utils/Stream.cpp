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

bool MemoryInputStream::IsEndOfFile() const
{
    return mPosition >= mBuffer.size();
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
    mPath = filePath;
    mFile = fopen(filePath, "rb");
    if (!mFile)
    {
        perror(filePath);
        return;
    }
    mSize = GetSize();
}

FileInputStream::~FileInputStream()
{
    if (mFile)
    {
        fclose(mFile);
        mFile = nullptr;
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
#if defined(_MSC_VER)
    const uint64_t size = _ftelli64(mFile);
#else
    const uint64_t size = ftello64(mFile);
#endif
    fseek(mFile, originalPos, SEEK_SET);

    return size;
}

uint64_t FileInputStream::GetPosition() const
{
#if defined(_MSC_VER)
    return _ftelli64(mFile);
#else
    return ftello64(mFile);
#endif
}

bool FileInputStream::SetPosition(uint64_t offset)
{
#if defined(_MSC_VER)
    return 0 != _fseeki64(mFile, offset, SEEK_SET);
#else
    return 0 != fseeko64(mFile, offset, SEEK_SET);
#endif
}

bool FileInputStream::IsEndOfFile() const
{
    return GetPosition() >= mSize;
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

FileOutputStream::~FileOutputStream()
{
    if (mFile)
    {
        fclose(mFile);
        mFile = nullptr;
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

void FileOutputStream::Flush()
{
    if (mFile != nullptr)
    {
        fflush(mFile);
    }
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

bool FileOutputStream::IsOK() const
{
    return mFile;
}
