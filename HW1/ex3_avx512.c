// Exercise 3 — Time an AVX-512 operation (vector multiply)
// Build: gcc -O0 -march=native -mavx512f -std=gnu11 ex3_avx512.c -o ex3_avx512
// Run:   ./ex3_avx512   # writes avx512_ex3.csv

#define _GNU_SOURCE
#include <immintrin.h>
#include <x86intrin.h>
#include <cpuid.h>       
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdalign.h>

static inline uint64_t rdtscp_now(void) {
    unsigned int aux;
    return __rdtscp(&aux);
}

static int have_avx512f(void) {
    unsigned int a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
    return (b & bit_AVX512F) != 0;  // from <cpuid.h>
}

volatile float sink_f;

int main(void) {
    const int TRIALS = 1000000;

    if (!have_avx512f()) {
        fprintf(stderr, "[Ex3] AVX-512F not supported; exiting.\n");
        return 0;
    }

    FILE* f = fopen("avx512_ex3.csv", "w");
    if (!f) { perror("fopen"); return 1; }
    fprintf(f, "cycles\n");

    alignas(64) float A[16] = {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32};
    alignas(64) float B[16] = {1,3,5,7, 9,11,13,15,17,19,21,23,25,27,29,31};

    __m512 a = _mm512_load_ps(A);
    __m512 b = _mm512_load_ps(B);
    volatile __m512 c;

    for (int i = 0; i < TRIALS; ++i) {
        _mm_mfence();
        uint64_t t0 = rdtscp_now();
        c = _mm512_mul_ps(a, b);          // the AVX-512 work
        uint64_t t1 = rdtscp_now();
        float out[16]; _mm512_store_ps(out, (__m512)c); sink_f = out[0]; // prevent DCE
        fprintf(f, "%llu\n", (unsigned long long)(t1 - t0));
    }

    fclose(f);
    fprintf(stderr, "[Ex3] Wrote avx512_ex3.csv\n");
    return 0;
}

