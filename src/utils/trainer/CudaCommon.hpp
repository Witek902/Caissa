#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

#include <vector>
#include <memory>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
            __debugbreak(); \
            exit(1); \
        } \
    } while(0)

namespace nn {
namespace cuda {

// CUDA memory management utilities
template<typename T>
class CudaBuffer
{
public:
    CudaBuffer() : m_data(nullptr), m_size(0) {}
    CudaBuffer(size_t size) : m_data(nullptr), m_size(0) { Allocate(size); }
    ~CudaBuffer() { Free(); }

    void Allocate(size_t size)
    {
        Free();
        m_size = size;
        CUDA_CHECK(cudaMalloc(&m_data, size * sizeof(T)));
        CUDA_CHECK(cudaMemset(m_data, 0, size * sizeof(T)));
    }

    void Free()
    {
        if (m_data)
        {
            CUDA_CHECK(cudaFree(m_data));
            m_data = nullptr;
            m_size = 0;
        }
    }

    void ClearAsync(cudaStream_t stream) const
    {
        CUDA_CHECK(cudaMemsetAsync(m_data, 0, m_size * sizeof(T), stream));
    }

    void CopyFromHost(const T* hostData, size_t size, cudaStream_t stream)
    {
        if (size > m_size) Allocate(size);
        CUDA_CHECK(cudaMemcpyAsync(m_data, hostData, size * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    void CopyFromHost(const T* hostData, size_t size)
    {
        if (size > m_size) Allocate(size);
        CUDA_CHECK(cudaMemcpy(m_data, hostData, size * sizeof(T), cudaMemcpyHostToDevice));
    }

    void CopyToHost(T* hostData, size_t size) const
    {
        if (size > m_size)
        {
            std::cerr << "CudaBuffer::CopyToHost size " << size << " exceeds buffer size " << m_size << std::endl;
            __debugbreak();
            exit(1);
        }
        CUDA_CHECK(cudaMemcpy(hostData, m_data, size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void CopyFromDevice(const T* deviceData, size_t size)
    {
        if (size > m_size) Allocate(size);
        CUDA_CHECK(cudaMemcpy(m_data, deviceData, size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    T* Get() { return m_data; }
    const T* Get() const { return m_data; }
    size_t Size() const { return m_size; }

private:
    T* m_data;
    size_t m_size;
};

// Pinned memory for faster transfers
template<typename T>
class PinnedBuffer
{
public:
    PinnedBuffer() : m_data(nullptr), m_size(0) {}
    PinnedBuffer(size_t size) : m_data(nullptr), m_size(0) { Allocate(size); }
    ~PinnedBuffer() { Free(); }

    void Allocate(size_t size)
    {
        Free();
        m_size = size;
        CUDA_CHECK(cudaMallocHost(&m_data, size * sizeof(T)));
    }

    void Free()
    {
        if (m_data)
        {
            CUDA_CHECK(cudaFreeHost(m_data));
            m_data = nullptr;
            m_size = 0;
        }
    }

    T* Get() { return m_data; }
    const T* Get() const { return m_data; }
    size_t Size() const { return m_size; }

private:
    T* m_data;
    size_t m_size;
};

// CUDA stream wrapper
class CudaStream
{
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&m_stream)); }
    ~CudaStream() { CUDA_CHECK(cudaStreamDestroy(m_stream)); }

    cudaStream_t Get() const { return m_stream; }

    void Synchronize() const { CUDA_CHECK(cudaStreamSynchronize(m_stream)); }

private:
    cudaStream_t m_stream;
};

} // namespace cuda
} // namespace nn
