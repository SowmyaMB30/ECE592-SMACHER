// Exercise 2 — Time any AVX2 operation (vector multiply)
// Build: gcc -O0 -march=native -mavx2 -std=gnu11 ex2_avx2_timing.c -o ex2_avx2_timing
// Run:   ./ex2_avx2_timing
// Output: avx2_ex2.csv (column: cycles)

#define _GNU_SOURCE
#include <cpuid.h> 
#include <immintrin.h>
#include <x86intrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdalign.h>

static inline uint64_t rdtscp_now(void) {
    unsigned int aux;
    return __rdtscp(&aux);
}

static int have_avx2(void) {
    int a,b,c,d;
    __cpuid_count(7, 0, a,b,c,d);
    return (b & (1<<5)) != 0; // AVX2 feature bit
}

volatile float sink_f;

int main(void) {
    const int TRIALS = 1000000;
    if (!have_avx2()) {
        fprintf(stderr, "[Ex2] AVX2 not supported on this CPU; exiting.\n");
        return 0;
    }

    FILE* f = fopen("avx2_ex2.csv", "w");
    if (!f) { perror("fopen"); return 1; }
    fprintf(f, "cycles\n");

    alignas(32) float A[8] = {2,4,6,8,10,12,14,16};
    alignas(32) float B[8] = {1,3,5,7, 9,11,13,15};

    __m256 a = _mm256_load_ps(A);
    __m256 b = _mm256_load_ps(B);
    volatile __m256 c;

    for (int i = 0; i < TRIALS; ++i) {
        _mm_mfence();                   // complete prior mem ops
        uint64_t t0 = rdtscp_now();     // start
        c = _mm256_mul_ps(a, b);        // AVX2 op to time
        uint64_t t1 = rdtscp_now();     // end
        float out[8]; _mm256_store_ps(out, (__m256)c); sink_f = out[0]; // prevent DCE
        fprintf(f, "%llu\n", (unsigned long long)(t1 - t0));
    }

    fclose(f);
    fprintf(stderr, "[Ex2] Wrote avx2_ex2.csv\n");
    return 0;
}

