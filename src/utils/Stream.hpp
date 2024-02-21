#pragma once

#include "../backend/Common.hpp"

#include <vector>
#include <string>
#include <stdio.h>

class InputStream
{
public:
    virtual ~InputStream() = default;
    virtual uint64_t GetSize() = 0;
    virtual uint64_t GetPosition() const = 0;
    virtual bool IsEndOfFile() const = 0;
    virtual bool Read(void* data, size_t size) = 0;
    virtual const char* GetFileName() const { return ""; }
};

class OutputStream
{
public:
    ~OutputStream() = default;
    virtual uint64_t GetSize() = 0;
    virtual bool Write(const void* data, size_t size) = 0;
    virtual bool IsOK() const { return true; }
};

//////////////////////////////////////////////////////////////////////////

class MemoryInputStream : public InputStream
{
public:
    MemoryInputStream(const std::vector<uint8_t>& buffer);
    virtual uint64_t GetSize() override;
    virtual uint64_t GetPosition() const override { return mPosition; }
    virtual bool IsEndOfFile() const override;
    virtual bool Read(void* data, size_t size) override;
private:
    const std::vector<uint8_t>& mBuffer;
    size_t mPosition;
};

class MemoryOutputStream : public OutputStream
{
public:
    MemoryOutputStream(std::vector<uint8_t>& buffer);
    virtual uint64_t GetSize() override;
    virtual bool Write(const void* data, size_t size) override;
private:
    std::vector<uint8_t>& mBuffer;
};

//////////////////////////////////////////////////////////////////////////

class FileInputStream : public InputStream
{
public:
    FileInputStream(const char* filePath);
    virtual ~FileInputStream();
    bool IsOpen() const;
    virtual uint64_t GetPosition() const override;
    bool SetPosition(uint64_t offset);
    virtual uint64_t GetSize() override;
    virtual bool IsEndOfFile() const override;
    virtual bool Read(void* data, size_t size) override;
    virtual const char* GetFileName() const override { return mPath.c_str(); }
private:
    FILE* mFile;
    uint64_t mSize = 0;
    std::string mPath;
};

class FileOutputStream : public OutputStream
{
public:
    FileOutputStream(const char* filePath);
    virtual ~FileOutputStream();
    bool IsOpen() const;
    bool Seek(uint64_t pos);
    void Flush();
    virtual uint64_t GetSize() override;
    virtual bool Write(const void* data, size_t size) override;
    virtual bool IsOK() const override;
private:
    FILE* mFile;
};
