#pragma once

#define K_C 512

#define M_R 8
#define N_R 16

#define M_S (M_R * 8)
#define N_S (N_R * 8)

#define M_C (M_S * 1)
#define N_C (N_S * 1)

#define NUM_ELEMENTS_PER_VEC_A 8
#define NUM_ELEMENTS_PER_VEC_B 8
#define NUM_ELEMENTS_PER_VEC_C 8

#define F0(src,dst) dst = src
#define F1(src,dst) dst = src
#define F2(src,dst) dst = src
#define F3(src,dst) dst = src
#define F5(src,dst) dst = src
#define OP1(a,b,r)  (r) = (a) * (b)
#define OP2(a,b,r)  (r) = (a) + (b)

typedef float blis_data_t;
typedef blis_data_t blis_data_a_t;
typedef blis_data_t blis_data_b_t;
typedef blis_data_t blis_data_c_t;
typedef unsigned blis_size_t;

template<typename T, blis_size_t N>
struct __attribute__((aligned(N * sizeof(T)))) BlisVec {
    T data[N];
};

extern "C" {

void blis(int m, int n, int k,
          const BlisVec<blis_data_a_t, NUM_ELEMENTS_PER_VEC_A> *A,
          const BlisVec<blis_data_b_t, NUM_ELEMENTS_PER_VEC_B> *B,
          BlisVec<blis_data_c_t, NUM_ELEMENTS_PER_VEC_C> *C);

}
