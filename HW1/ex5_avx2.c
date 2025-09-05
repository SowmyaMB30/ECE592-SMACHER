// Build: gcc -O0 -mavx2 -std=gnu11 ex5_avx.c -o ex5_avx2
// Run:   ./ex5_avx2
// Output: avx_rest_ex5.csv  (cycles_first,cycles_second)

#include <immintrin.h>
#include <x86intrin.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

static inline uint64_t rdtscp_now(void){ unsigned aux; return __rdtscp(&aux); }
static volatile float sink_f = 0.0f;

#define TRIALS 1000000
#define REST_US 100000   // 100 ms

int main(void){
    alignas(32) float A[8] = {2,4,6,8,10,12,14,16};
    alignas(32) float B[8] = {1,3,5,7,9,11,13,15};
    alignas(32) float OUT[8];

    __m256 a = _mm256_load_ps(A);
    __m256 b = _mm256_load_ps(B);
    __m256 c;

    FILE* f = fopen("avx_rest_ex5.csv","w");
    if(!f){ perror("csv"); return 1; }
    fprintf(f,"cycles_first,cycles_second\n");

    for(long i=0;i<TRIALS;i++){
        _mm_mfence();
        uint64_t t0 = rdtscp_now();
        c = _mm256_mul_ps(a,b);                // first timing (no delay)
        uint64_t t1 = rdtscp_now();
        _mm256_store_ps(OUT, c); sink_f += OUT[0];  // prevent DCE

        usleep(REST_US);                        // 100 ms rest

        _mm_mfence();
        uint64_t t2 = rdtscp_now();
        c = _mm256_mul_ps(a,b);                // second timing (after rest)
        uint64_t t3 = rdtscp_now();
        _mm256_store_ps(OUT, c); sink_f += OUT[1];

        fprintf(f,"%llu,%llu\n",
                (unsigned long long)(t1 - t0),
                (unsigned long long)(t3 - t2));
    }
    fclose(f);
    return 0;
}

