#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_xyz(b, m, 4) xyz(b, n, 4)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 4;
  new_xyz += batch_index * m * 4;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 4 + 0];
    float new_y = new_xyz[j * 4 + 1];
    float new_z = new_xyz[j * 4 + 2];
    float new_q = new_xyz[j * 4 + 3];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 4 + 0];
      float y = xyz[k * 4 + 1];
      float z = xyz[k * 4 + 2];
      float q = xyz[k * 4 + 3];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z) + (new_q - q) * (new_q - q);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, new_xyz, xyz, idx);

  CUDA_CHECK_ERRORS();
}
