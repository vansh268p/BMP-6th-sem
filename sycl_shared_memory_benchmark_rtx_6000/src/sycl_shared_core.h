#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SyclSharedTimings {
    double kernel_ms;
    double total_ms;
} SyclSharedTimings;

int sycl_shared_has_gpu(void);
int sycl_shared_init(size_t n);
void sycl_shared_finalize(void);
size_t sycl_shared_size(void);
float* sycl_shared_a(void);
float* sycl_shared_b(void);
float* sycl_shared_c(void);
int sycl_shared_matmul(float* a, float* b, float* c, size_t n, SyclSharedTimings* t);

#ifdef __cplusplus
}
#endif
