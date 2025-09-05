// ex1_llc.c
// Exercise 1 — Time LLC hit vs LLC miss using RDTSCP + MFENCE + CLFLUSH
// Build: gcc -O0 -march=native -std=gnu11 ex1_llc.c -o ex1_llc
// Run:   ./ex1_llc
// Output: llc_hit_ex1.csv, llc_miss_ex1.csv

#define _GNU_SOURCE
#include <immintrin.h>
#include <x86intrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdalign.h>
#include <time.h>

static inline uint64_t rdtscp_now(void) {
    unsigned aux;
    return __rdtscp(&aux);
}

#define LINE_SIZE        64                  // cache line bytes on x86
#define WORKING_SET_MB   16                  // footprint to keep data resident in LLC
#define L2_EVICT_MB      4                   // footprint to evict from L1/L2
#define TRIALS           1000000             

// Allocate page-aligned memory
static void* alloc_pages(size_t bytes) {
    void* p = NULL;
    // 4096-byte page alignment
    if (posix_memalign(&p, 4096, bytes) != 0) return NULL;
    // Touch to commit pages
    memset(p, 0, bytes);
    return p;
}

// Sweep a buffer by cache-line stride to bring it to (or keep it in) cache
static inline void sweep_lines_volatile(volatile unsigned char* buf, size_t bytes) {
    for (size_t i = 0; i < bytes; i += LINE_SIZE) {
        // Volatile read prevents the compiler from optimizing away
        (void)buf[i];
    }
}

// Thrash L1/L2 using a modest buffer (> L2) so the target line is not in upper levels
static inline void evict_L1_L2(volatile unsigned char* buf, size_t bytes) {
    sweep_lines_volatile(buf, bytes);
}

int main(void) {
   
    const size_t working_set_bytes = (size_t)WORKING_SET_MB * 1024 * 1024;
    const size_t l2_evict_bytes    = (size_t)L2_EVICT_MB   * 1024 * 1024;
    unsigned char* working_set = alloc_pages(working_set_bytes);
    unsigned char* l2_evict    = alloc_pages(l2_evict_bytes);

    if (!working_set || !l2_evict) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    volatile unsigned char* target = working_set + (128 * LINE_SIZE);
    // Files for results
    FILE* fhit  = fopen("llc_hit_ex1.csv",  "w");
    FILE* fmiss = fopen("llc_miss_ex1.csv", "w");
    if (!fhit || !fmiss) {
        perror("fopen");
        return 1;
    }
    fprintf(fhit,  "cycles\n");
    fprintf(fmiss, "cycles\n");
    sweep_lines_volatile(working_set, working_set_bytes);
    sweep_lines_volatile(l2_evict,    l2_evict_bytes);
   //LLC Hit
    for (int i = 0; i < TRIALS; ++i) {
        sweep_lines_volatile(working_set, working_set_bytes);
        evict_L1_L2(l2_evict, l2_evict_bytes);
        _mm_mfence();
        uint64_t t0 = rdtscp_now();
        unsigned char v = *target;  // timed load (expected LLC hit)
        (void)v;
        uint64_t t1 = rdtscp_now();

        fprintf(fhit, "%llu\n", (unsigned long long)(t1 - t0));
    }

    //LLC Miss
    for (int i = 0; i < TRIALS; ++i) {
        _mm_clflush((const void*)target); // evict from entire hierarchy
        _mm_mfence();

        uint64_t t0 = rdtscp_now();
        unsigned char v = *target;  // timed load (LLC miss -> DRAM)
        (void)v;
        uint64_t t1 = rdtscp_now();

        fprintf(fmiss, "%llu\n", (unsigned long long)(t1 - t0));
    }

    fclose(fhit);
    fclose(fmiss);

    free((void*)working_set);
    free((void*)l2_evict);

    fprintf(stderr, "[Ex1] Wrote llc_hit_ex1.csv and llc_miss_ex1.csv\n");
    return 0;
}

