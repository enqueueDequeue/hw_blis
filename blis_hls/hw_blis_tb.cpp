#include <cstdlib>
#include <iostream>

#include "hw_blis.h"

#define D_TYPE_FLOAT 0
#define D_TYPE_INT 1
#define D_TYPE D_TYPE_FLOAT

void print_arr(blis_data_t *data, int d, int w) {

    for (int j = 0; j < d; j++) {
        for (int i = 0; i < w; i++) {
            std::cout << data[j * w + i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

int main() {
    const int m = 128;
    const int k = 232;
    const int n = 152;

    // const int m = 32;
    // const int k = 32;
    // const int n = 32;

    // const int m = 16;
    // const int k = 24;
    // const int n = 8;

    const int val_max = 10;

    BlisVec<blis_data_a_t, NUM_ELEMENTS_PER_VEC_A> A[(m * k + NUM_ELEMENTS_PER_VEC_A - 1) / NUM_ELEMENTS_PER_VEC_A] __attribute__((aligned(64)));
    BlisVec<blis_data_b_t, NUM_ELEMENTS_PER_VEC_B> B[(k * n + NUM_ELEMENTS_PER_VEC_B - 1) / NUM_ELEMENTS_PER_VEC_B] __attribute__((aligned(64)));
    BlisVec<blis_data_c_t, NUM_ELEMENTS_PER_VEC_C> C[(m * n + NUM_ELEMENTS_PER_VEC_C - 1) / NUM_ELEMENTS_PER_VEC_C] __attribute__((aligned(64)));
    BlisVec<blis_data_c_t, NUM_ELEMENTS_PER_VEC_C> C_Ref[(m * n + NUM_ELEMENTS_PER_VEC_C - 1) / NUM_ELEMENTS_PER_VEC_C] __attribute__((aligned(64)));

    #define GET(src, idx) (src).data[(idx)]
    #define SET(dst, idx, value) (dst).data[(idx)] = (value)

    #define BLOCK_IDX(x, nv) ((x) / (nv))
    #define BLOCK_OFF(x, nv) ((x) - (BLOCK_IDX(x, nv) * (nv)))

    #define GET_H(src, abs_idx, nv) GET(src[BLOCK_IDX(abs_idx, nv)], BLOCK_OFF(abs_idx, nv))
    #define SET_H(dst, abs_idx, nv, value) SET(dst[BLOCK_IDX(abs_idx, nv)], BLOCK_OFF(abs_idx, nv), (value))

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int idx = i * k + j;
            blis_data_a_t value = static_cast<blis_data_a_t>(std::rand()) / static_cast<blis_data_a_t>(RAND_MAX / val_max);
            SET_H(A, idx, NUM_ELEMENTS_PER_VEC_A, value);
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            blis_data_b_t value = static_cast<blis_data_b_t>(std::rand()) / static_cast<blis_data_b_t>(RAND_MAX / val_max);
            SET_H(B, idx, NUM_ELEMENTS_PER_VEC_B, value);
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;

            // blis_data_c_t value = static_cast<blis_data_c_t>(std::rand()) / static_cast<blis_data_c_t>(RAND_MAX / val_max);
            blis_data_c_t value = 0;

            SET_H(C_Ref, idx, NUM_ELEMENTS_PER_VEC_C, value);
            SET_H(C, idx, NUM_ELEMENTS_PER_VEC_C, value);
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            blis_data_c_t sum = 0;

            for (int z = 0; z < k; z++) {
                int a_idx = i * k + z;
                int b_idx = z * n + j;
                sum += GET_H(A, a_idx, NUM_ELEMENTS_PER_VEC_A) * GET_H(B, b_idx, NUM_ELEMENTS_PER_VEC_B);
            }

            int c_idx = i * n + j;
            SET_H(C_Ref, c_idx, NUM_ELEMENTS_PER_VEC_C, sum);
        }
    }

    std::cout << "A:" << std::endl;
    print_arr((blis_data_t *) A, m, k);

    std::cout << "B:" << std::endl;
    print_arr((blis_data_t *) B, k, n);

    std::cout << "C_Ref:" << std::endl;
    print_arr((blis_data_t *) C_Ref, m, n);

    blis(m, n, k,
        (const BlisVec<blis_data_a_t, NUM_ELEMENTS_PER_VEC_A> *) A,
        (const BlisVec<blis_data_b_t, NUM_ELEMENTS_PER_VEC_B> *) B,
        (BlisVec<blis_data_c_t, NUM_ELEMENTS_PER_VEC_C> *) C);

    std::cout << "C:" << std::endl;
    print_arr((blis_data_t *) C, m, n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            blis_data_t x = GET_H(C_Ref, idx, NUM_ELEMENTS_PER_VEC_C);
            blis_data_t y = GET_H(C, idx, NUM_ELEMENTS_PER_VEC_C);

            #if D_TYPE_FLOAT == D_TYPE
            blis_data_t delta = (x - y) / (x + y);
            blis_data_t abs_delta = (delta >= 0) ? delta : -1 * delta;

            if (abs_delta > 0.001) {
                std::cout << "Error: " << x << " != " << y << " @ (" << i << ", " << j << ")" << std::endl;
                return 1;
            }
            #elif D_TYPE_INT == D_TYPE
            if (x != y) {
                std::cout << "Error: " << x << " != " << y << " @ (" << i << ", " << j << ")" << std::endl;
                return 2;
            }
            #endif
        }
    }
}
