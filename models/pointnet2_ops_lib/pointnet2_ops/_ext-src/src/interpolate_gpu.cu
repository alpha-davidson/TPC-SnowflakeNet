#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: unknown(b, n, 3) known(b, m, 3)
// output: dist2(b, n, 3), idx(b, n, 3)
__global__ void three_nn_kernel(int b, int n, int m, int point_dim,
                                const float *__restrict__ unknown,
                                const float *__restrict__ known,
                                float *__restrict__ dist2,
                                int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  unknown += batch_index * n * point_dim;
  known += batch_index * m * point_dim;
  dist2 += batch_index * n * point_dim;
  idx += batch_index * n * point_dim;

  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int j = index; j < n; j += stride) {
    float ux = unknown[j * point_dim + 0];
    float uy = unknown[j * point_dim + 1];
    float uz = unknown[j * point_dim + 2];
    float uq;
    if (point_dim == 4) {
      uq = unknown[j * point_dim + 3];
    }
    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
      float x, y, z, q;
      x = known[k * 4 + 0];
      y = known[k * 4 + 1];
      z = known[k * 4 + 2];
      if (point_dim == 4) {
        q = known[k * 4 + 3];
      }
      float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
      if (point_dim == 4) {
        d += (uq - q) * (uq - q);
      }
      if (d < best1) {
        best3 = best2;
        besti3 = besti2;
        best2 = best1;
        besti2 = besti1;
        best1 = d;
        besti1 = k;
      } else if (d < best2) {
        best3 = best2;
        besti3 = besti2;
        best2 = d;
        besti2 = k;
      } else if (d < best3) {
        best3 = d;
        besti3 = k;
      }
    }
    dist2[j * 3 + 0] = best1;
    dist2[j * 3 + 1] = best2;
    dist2[j * 3 + 2] = best3;

    idx[j * 3 + 0] = besti1;
    idx[j * 3 + 1] = besti2;
    idx[j * 3 + 2] = besti3;
  }
}

void three_nn_kernel_wrapper(int b, int n, int m, int point_dim, const float *unknown,
                             const float *known, float *dist2, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  three_nn_kernel<<<b, opt_n_threads(n), 0, stream>>>(b, n, m, point_dim, unknown, known,
                                                      dist2, idx);

  CUDA_CHECK_ERRORS();
}

// input: points(b, c, m), idx(b, n, 3), weight(b, n, 3)
// output: out(b, c, n)
__global__ void three_interpolate_kernel(int b, int c, int m, int n, int point_dim,
                                         const float *__restrict__ points,
                                         const int *__restrict__ idx,
                                         const float *__restrict__ weight,
                                         float *__restrict__ out) {
  int batch_index = blockIdx.x;
  points += batch_index * m * c;

  idx += batch_index * n * point_dim;
  weight += batch_index * n * point_dim;

  out += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * n; i += stride) {
    const int l = i / n;
    const int j = i % n;
    float w1 = weight[j * point_dim + 0];
    float w2 = weight[j * point_dim + 1];
    float w3 = weight[j * point_dim + 2];
    float w4;
    if (point_dim == 4) {
      w4 = weight[j * point_dim + 3];
    }

    int i1 = idx[j * point_dim + 0];
    int i2 = idx[j * point_dim + 1];
    int i3 = idx[j * point_dim + 2];
    int i4;
    if (point_dim == 4) {
      i4 = idx[j * point_dim + 3];
    }

    out[i] = points[l * m + i1] * w1 + points[l * m + i2] * w2 +
             points[l * m + i3] * w3;
    if (point_dim == 4) {
      out[i] += points[l * m + i4] * w4;
    }
  }
}

void three_interpolate_kernel_wrapper(int b, int c, int m, int n, int point_dim,
                                      const float *points, const int *idx,
                                      const float *weight, float *out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  three_interpolate_kernel<<<b, opt_block_config(n, c), 0, stream>>>(
      b, c, m, n, point_dim, points, idx, weight, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, n), idx(b, n, 3), weight(b, n, 3)
// output: grad_points(b, c, m)

__global__ void three_interpolate_grad_kernel(
    int b, int c, int n, int m, int point_dim, const float *__restrict__ grad_out,
    const int *__restrict__ idx, const float *__restrict__ weight,
    float *__restrict__ grad_points) {
  int batch_index = blockIdx.x;
  grad_out += batch_index * n * c;
  idx += batch_index * n * point_dim;
  weight += batch_index * n * point_dim;
  grad_points += batch_index * m * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * n; i += stride) {
    const int l = i / n;
    const int j = i % n;
    float w1 = weight[j * point_dim + 0];
    float w2 = weight[j * point_dim + 1];
    float w3 = weight[j * point_dim + 2];
    float w4;
    if (point_dim == 4) {
      w4 = weight[j * point_dim + 3];
    }
    int i1 = idx[j * point_dim + 0];
    int i2 = idx[j * point_dim + 1];
    int i3 = idx[j * point_dim + 2];
    int i4;
    if (point_dim == 4) {
      i4 = idx[j * point_dim + 3];
    }
    atomicAdd(grad_points + l * m + i1, grad_out[i] * w1);
    atomicAdd(grad_points + l * m + i2, grad_out[i] * w2);
    atomicAdd(grad_points + l * m + i3, grad_out[i] * w3);
    if (point_dim == 4) {
      atomicAdd(grad_points + l * m + i4, grad_out[i] * w4);
    }
  }
}

void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m, int point_dim,
                                           const float *grad_out,
                                           const int *idx, const float *weight,
                                           float *grad_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  three_interpolate_grad_kernel<<<b, opt_block_config(n, c), 0, stream>>>(
      b, c, n, m, point_dim, grad_out, idx, weight, grad_points);

  CUDA_CHECK_ERRORS();
}
