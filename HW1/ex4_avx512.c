// Build: gcc -O0 -mavx512f -std=gnu11 ex4_avx512.c -o ex4_avx512
// Run:   ./ex4_avx512
// Output: avx512_rest_ex4.csv  (cycles_first,cycles_second)

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
    alignas(64) float A[16] = {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32};
    alignas(64) float B[16] = {1,3,5,7, 9,11,13,15,17,19,21,23,25,27,29,31};
    alignas(64) float OUT[16];

    __m512 a = _mm512_load_ps(A);
    __m512 b = _mm512_load_ps(B);
    __m512 c;

    FILE* f = fopen("avx512_rest_ex4.csv","w");
    if(!f){ perror("csv"); return 1; }
    fprintf(f,"cycles_first,cycles_second\n");

    for(long i=0;i<TRIALS;i++){
        _mm_mfence();
        uint64_t t0 = rdtscp_now();
        c = _mm512_mul_ps(a,b);                // first timing (no delay)
        uint64_t t1 = rdtscp_now();
        _mm512_store_ps(OUT, c); sink_f += OUT[0];

        usleep(REST_US);                        // 100 ms rest

        _mm_mfence();
        uint64_t t2 = rdtscp_now();
        c = _mm512_mul_ps(a,b);                // second timing (after rest)
        uint64_t t3 = rdtscp_now();
        _mm512_store_ps(OUT, c); sink_f += OUT[1];

        fprintf(f,"%llu,%llu\n",
                (unsigned long long)(t1 - t0),
                (unsigned long long)(t3 - t2));
    }
    fclose(f);
    return 0;
}

