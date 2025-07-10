#pragma once

#include <cuda_runtime.h>

__global__ void u8ToHalf(const uint8_t* src, half* dst, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = static_cast<half>(src[idx]);
  }
}

__global__ void halfToU8(const half* src, uint8_t* dst, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = static_cast<uint8_t>(src[idx]);
  }
}

__global__ void u8ToFloat(const uint8_t* src, float* dst, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = static_cast<float>(src[idx]);
  }
}

__global__ void floatToU8(const float* src, uint8_t* dst, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = static_cast<uint8_t>(src[idx]);
  }
}