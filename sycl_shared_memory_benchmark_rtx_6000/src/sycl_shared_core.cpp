#include "sycl_shared_core.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sycl/sycl.hpp>

namespace {

constexpr int TILE = 16;

std::unique_ptr<sycl::queue> g_queue;
float* g_a = nullptr;
float* g_b = nullptr;
float* g_c = nullptr;
size_t g_n = 0;

double elapsed_ms(std::chrono::high_resolution_clock::time_point a,
                  std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

size_t round_up(size_t x, size_t m) {
    return ((x + m - 1) / m) * m;
}

void ensure_queue() {
    if (!g_queue) {
        g_queue = std::make_unique<sycl::queue>(sycl::gpu_selector_v);
    }
}

} // namespace

extern "C" int sycl_shared_has_gpu(void) {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        return q.get_device().is_gpu() ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

extern "C" int sycl_shared_init(size_t n) {
    try {
        sycl_shared_finalize();
        ensure_queue();
        g_n = n;
        const size_t elems = n * n;
        g_a = sycl::malloc_shared<float>(elems, *g_queue);
        g_b = sycl::malloc_shared<float>(elems, *g_queue);
        g_c = sycl::malloc_shared<float>(elems, *g_queue);
        if (!g_a || !g_b || !g_c) return 2;

        float* a = g_a;
        float* b = g_b;
        float* c = g_c;
        g_queue->parallel_for(sycl::range<1>(elems), [=](sycl::id<1> idx) {
            const uint32_t i = static_cast<uint32_t>(idx[0]);
            a[i] = static_cast<float>((i * 17u + 13u) & 0xFFu) / 255.0f;
            b[i] = static_cast<float>((i * 29u + 7u) & 0xFFu) / 255.0f;
            c[i] = 0.0f;
        }).wait_and_throw();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sycl_shared_init failed: " << e.what() << "\n";
        return 1;
    } catch (...) {
        return 1;
    }
}

extern "C" void sycl_shared_finalize(void) {
    if (g_queue) {
        if (g_a) sycl::free(g_a, *g_queue);
        if (g_b) sycl::free(g_b, *g_queue);
        if (g_c) sycl::free(g_c, *g_queue);
    }
    g_a = nullptr;
    g_b = nullptr;
    g_c = nullptr;
    g_n = 0;
}

extern "C" size_t sycl_shared_size(void) {
    return g_n;
}

extern "C" float* sycl_shared_a(void) {
    return g_a;
}

extern "C" float* sycl_shared_b(void) {
    return g_b;
}

extern "C" float* sycl_shared_c(void) {
    return g_c;
}

extern "C" int sycl_shared_matmul(float* a, float* b, float* c, size_t n, SyclSharedTimings* t) {
    if (!a || !b || !c || !t || n == 0) return 2;
    try {
        ensure_queue();
        const size_t padded = round_up(n, TILE);
        auto t0 = std::chrono::high_resolution_clock::now();
        g_queue->submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 2> asub(sycl::range<2>(TILE, TILE), h);
            sycl::local_accessor<float, 2> bsub(sycl::range<2>(TILE, TILE), h);
            h.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(padded, padded),
                                  sycl::range<2>(TILE, TILE)),
                [=](sycl::nd_item<2> item) {
                    const size_t row = item.get_global_id(0);
                    const size_t col = item.get_global_id(1);
                    const size_t lr = item.get_local_id(0);
                    const size_t lc = item.get_local_id(1);
                    float acc = 0.0f;

                    for (size_t tile = 0; tile < n; tile += TILE) {
                        const size_t a_col = tile + lc;
                        const size_t b_row = tile + lr;
                        asub[lr][lc] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
                        bsub[lr][lc] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
                        item.barrier(sycl::access::fence_space::local_space);
                        #pragma unroll
                        for (int k = 0; k < TILE; ++k) {
                            acc += asub[lr][k] * bsub[k][lc];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (row < n && col < n) {
                        c[row * n + col] = acc;
                    }
                });
        });
        auto k0 = std::chrono::high_resolution_clock::now();
        g_queue->wait_and_throw();
        auto t1 = std::chrono::high_resolution_clock::now();
        t->kernel_ms = elapsed_ms(k0, t1);
        t->total_ms = elapsed_ms(t0, t1);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sycl_shared_matmul failed: " << e.what() << "\n";
        return 1;
    } catch (...) {
        return 1;
    }
}
