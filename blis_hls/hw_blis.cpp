#include <cassert>
#include <iostream>

#include "hw_blis.h"


// implied in the case of an outer product
// using this for modularity
constexpr int KR = 1;

template<typename T>
static void print_arr(const char *prefix, T *arr, int m, int n) {
  std::cout << std::endl;
  std::cout << prefix << std::endl;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << arr[i * n + j] << ",";
    }
    std::cout << std::endl;
  }
}

template<typename T, blis_size_t N>
static void copy_arr(const T *src, T *dst) {
  #pragma HLS INLINE

  loop_copy:
  for (int i = 0; i < N; i++) {
    #pragma HLS UNROLL
    dst[i] = src[i];
  }
}

template <typename T, blis_size_t vec_len, blis_size_t mc, blis_size_t nc, blis_size_t mr, blis_size_t nr>
static void vec_load(blis_size_t y, blis_size_t x,
                     blis_size_t src_m, blis_size_t src_n,
                     blis_size_t src_mn, blis_size_t y_src_n,
                     const BlisVec<T, vec_len> *src,
                     T dst[mc * nc]) {

  #pragma HLS INLINE

  assert(x % vec_len == 0);
  assert(nc % vec_len == 0);
  assert(nr % vec_len == 0 || vec_len % nr == 0);

  blis_size_t io_src_n = 0;

  loop_load_io:
  for (blis_size_t io = 0; io < mc; io += mr) {
    #pragma HLS UNROLL off

    if (nr > vec_len) {
      blis_size_t ii_src_n = 0;

      loop_load_r_ii:
      for (blis_size_t ii = 0; ii < mr; ii++) {
        #pragma HLS UNROLL off
        blis_size_t src_base_idx = 0;

        src_base_idx += y_src_n;
        src_base_idx += io_src_n;
        src_base_idx += ii_src_n;

        loop_load_r_jo:
        for (blis_size_t jo = 0; jo < nc; jo += nr) {
          #pragma HLS UNROLL off

          loop_load_ji:
          for (int ji = 0; ji < nr; ji += vec_len) {
            #pragma HLS UNROLL off
            #pragma HLS PIPELINE

            blis_size_t src_idx_x = x + jo + ji;
            blis_size_t src_idx = (src_base_idx + src_idx_x) / vec_len;

            // actual load from memory
            BlisVec<T, vec_len> read;

            if (src_idx * vec_len < src_mn
                && src_idx_x + vec_len <= src_n) {

              read = src[src_idx];
            } else {
              read = { 0 };
            }

            copy_arr<T, vec_len>(read.data, &dst[io * nc + (jo + ji) * mr + ii]);
          }
        }

        ii_src_n += src_n;
      }
    } else {
      loop_load_v_jo:
      for (blis_size_t jo = 0; jo < nc; jo += vec_len) {
        #pragma HLS UNROLL off

        T rob[mr * vec_len];
        #pragma HLS ARRAY_PARTITION variable = rob type = cyclic factor = vec_len
        #pragma HLS BIND_STORAGE variable = rob type = ram_1p impl = lutram

        blis_size_t ii_src_n = 0;

        loop_load_v_ii:
        for (blis_size_t ii = 0; ii < mr; ii++) {
          #pragma HLS UNROLL off
          #pragma HLS PIPELINE

          blis_size_t src_base_idx = 0;

          src_base_idx += y_src_n;
          src_base_idx += io_src_n;
          src_base_idx += ii_src_n;

          blis_size_t src_idx_x = x + jo;
          blis_size_t src_idx = (src_base_idx + src_idx_x) / vec_len;

          // actual load from memory
          BlisVec<T, vec_len> read;

          if (src_idx * vec_len < src_mn
              && src_idx_x + vec_len <= src_n) {

            read = src[src_idx];
          } else {
            read = { 0 };
          }

          copy_arr<T, vec_len>(read.data, &rob[ii * vec_len]);

          ii_src_n += src_n;
        }

        loop_scatter_ji:
        for (int ji = 0; ji < vec_len; ji += nr) {
          #pragma HLS UNROLL off

          loop_scatter_ii:
          for (blis_size_t ii = 0; ii < mr; ii++) {
            #pragma HLS UNROLL

            copy_arr<T, nr>(&rob[ii * vec_len + ji], &dst[io * nc + (jo + ji) * mr + ii]);
          }
        }
      }
    }

    io_src_n += (src_n * mr);
  }
}

template <typename T, blis_size_t vec_len, blis_size_t mf, blis_size_t nf, blis_size_t mr, blis_size_t nr>
static void vec_load_c(blis_size_t y, blis_size_t x,
                       blis_size_t src_m, blis_size_t src_n,
                       blis_size_t src_mn, blis_size_t y_src_n,
                       const BlisVec<T, vec_len> *src,
                       T dst[mf * nf][mr * nr]) {

  #pragma HLS INLINE

  assert(x % vec_len == 0);
  assert(nr % vec_len == 0);

  blis_size_t io_src_n = 0;

  loop_load_c_io:
  for (blis_size_t io = 0; io < mf; io++) {
    #pragma HLS UNROLL off

    loop_load_c_jo:
    for (blis_size_t jo = 0; jo < nf; jo++) {
      #pragma HLS UNROLL off

      T rob[mr * nr];

      blis_size_t ii_src_n = 0;

      loop_load_c_ii:
      for (blis_size_t ii = 0; ii < mr; ii++) {
        #pragma HLS UNROLL off

        blis_size_t src_base_idx = 0;

        src_base_idx += y_src_n;
        src_base_idx += io_src_n;
        src_base_idx += ii_src_n;

        loop_load_c_ji:
        for (blis_size_t ji = 0; ji < nr; ji += vec_len) {
          #pragma HLS UNROLL off
          #pragma HLS PIPELINE

          blis_size_t src_idx_x = x + (jo * nr) + ji;
          blis_size_t src_idx = (src_base_idx + src_idx_x) / vec_len;

          // actual load from memory
          BlisVec<T, vec_len> read;

          if (src_idx * vec_len < src_mn
              && src_idx_x + vec_len <= src_n) {

            read = src[src_idx];
          } else {
            read = { 0 };
          }

          copy_arr<T, vec_len>(read.data, &rob[ii * nr + ji]);
        }

        ii_src_n += src_n;
      }

      copy_arr<T, mr * nr>(rob, dst[io * nf + jo]);
    }

    io_src_n += (src_n * mr);
  }
}

template <typename T, blis_size_t vec_len, blis_size_t mf, blis_size_t nf, blis_size_t mr, blis_size_t nr>
static void vec_store_c(blis_size_t y, blis_size_t x,
                        blis_size_t dst_m, blis_size_t dst_n,
                        blis_size_t dst_mn, blis_size_t y_dst_n,
                        T src[mf * nf][mr * nr],
                        BlisVec<T, vec_len> *dst) {

  #pragma HLS INLINE

  assert(x % vec_len == 0);
  assert(nr % vec_len == 0);

  blis_size_t io_dst_n = 0;

  loop_store_c_io:
  for (blis_size_t io = 0; io < mf; io++) {
    #pragma HLS UNROLL off

    loop_store_c_jo:
    for (blis_size_t jo = 0; jo < nf; jo++) {
      #pragma HLS UNROLL off

      T rob[mr * nr];

      copy_arr<T, mr * nr>(src[io * nf + jo], rob);

      blis_size_t ii_dst_n = 0;

      loop_store_i:
      for (blis_size_t ii = 0; ii < mr; ii++) {
        #pragma HLS UNROLL off

        blis_size_t dst_base_idx = 0;

        dst_base_idx += y_dst_n;
        dst_base_idx += io_dst_n;
        dst_base_idx += ii_dst_n;

        loop_store_j:
        for (blis_size_t ji = 0; ji < nr; ji += vec_len) {
          #pragma HLS UNROLL off
          #pragma HLS PIPELINE

          blis_size_t dst_idx_x = x + (jo * nr) + ji;
          blis_size_t dst_idx = (dst_base_idx + dst_idx_x) / vec_len;

          BlisVec<T, vec_len> write;

          copy_arr<T, vec_len>(&rob[(ii * nr) + ji], write.data);

          if (dst_idx * vec_len < dst_mn
              && dst_idx_x + vec_len <= dst_n) {

            dst[dst_idx] = write;
          }
        }

        ii_dst_n += dst_n;
      }
    }

    io_dst_n += (dst_n * mr);
  }
}

template<typename T, blis_size_t nc, blis_size_t mr>
static void act_load_a(blis_size_t y, blis_size_t x, const T *src, T dst[mr]) {
  #pragma HLS INLINE

  int xx = x * mr;
  int yy = y * nc;
  int zz = (xx + yy);

  copy_arr<T, mr>(&src[zz], dst);
}

template<typename T, blis_size_t nc, blis_size_t nr>
static void act_load_b(blis_size_t y, blis_size_t x, const T *src, T dst[nr]) {
  #pragma HLS INLINE

  int xx = x;
  int yy = y * nc;
  int zz = (xx + yy);

  copy_arr<T, nr>(&src[zz], dst);
}

template<blis_size_t mr, blis_size_t nr>
static void multiply(const blis_data_a_t a[mr], const blis_data_b_t b[nr], blis_data_c_t c[mr * nr]) {
  #pragma HLS ALLOCATION operation instances = fmul limit = 43
  #pragma HLS INLINE

  for (blis_size_t i = 0; i < mr; i++) {
    #pragma HLS UNROLL

    for (blis_size_t j = 0; j < nr; j++) {
      #pragma HLS UNROLL

      blis_size_t dst_idx = i * nr + j;
      OP1(a[i], b[j], c[dst_idx]);
    }
  }
}

template<blis_size_t mr, blis_size_t nr>
static void accumulate(const blis_data_c_t c_in[mr * nr], blis_data_c_t cr[mr * nr], blis_data_c_t c_out[mr * nr]) {
  #pragma HLS ALLOCATION operation instances = fadd limit = 43
  #pragma HLS INLINE

  for (blis_size_t i = 0; i < mr; i++) {
    #pragma HLS UNROLL

    for (blis_size_t j = 0; j < nr; j++) {
      #pragma HLS UNROLL

      blis_size_t idx = i * nr + j;
      OP2(c_in[idx], cr[idx], c_out[idx]);
    }
  }
}

template<blis_size_t mr, blis_size_t nr,
         blis_size_t ms, blis_size_t ns,
         blis_size_t mc, blis_size_t nc, blis_size_t kc>
static void micro_kernel(const blis_data_a_t *a,
                         const blis_data_b_t *b,
                         blis_size_t y, blis_size_t x,
                         blis_size_t i, blis_size_t j,
                         blis_size_t m, blis_size_t n,
                         blis_size_t mn, blis_size_t y_n,
                         BlisVec<blis_data_c_t, NUM_ELEMENTS_PER_VEC_C> *c) {

  constexpr blis_size_t mf = ms / mr;
  constexpr blis_size_t nf = ns / nr;

  blis_data_c_t c_temp[mf * nf][mr * nr];
  #pragma HLS ARRAY_RESHAPE variable = c_temp type = block factor = (mr * nr) dim = 2
  // #pragma HLS ARRAY_PARTITION variable = c_temp type = complete dim = 0
  // #pragma HLS ARRAY_PARTITION variable = c_temp type = cyclic factor = (mr * nr / 2) dim = 2
  #pragma HLS BIND_STORAGE variable = c_temp type = ram_1p impl = lutram

  // maybe make MR * NR the width of the channel and mf * nf will be a single port in that case

  vec_load_c<blis_data_c_t, NUM_ELEMENTS_PER_VEC_C, mf, nf, mr, nr>(y, x, m, n, mn, y_n, c, c_temp);

  // matrix multiply logic
  loop_micro_mat_mul:
  for (blis_size_t p = 0; p <  kc; p++) {
    #pragma HLS LOOP_FLATTEN off

    loop_micro_mat_mul_ii:
    for (blis_size_t ii = 0; ii < mf; ii++) {

      loop_micro_mat_mul_ji:
      for (blis_size_t ji = 0; ji < nf; ji++) {
        #pragma HLS PIPELINE

        blis_data_c_t c_ref[mr * nr];

        blis_data_c_t ar[mr];
        blis_data_c_t br[nr];
        blis_data_c_t cr[mr * nr];

        blis_size_t idx = (ii * nf) + ji;

        copy_arr<blis_data_c_t, mr * nr>(c_temp[idx], c_ref);

        act_load_a<blis_data_a_t, kc, mr>(i + (ii * mr), p, a, ar);
        act_load_b<blis_data_b_t, nc, nr>(p, j + (ji * nr), b, br);
        multiply<mr, nr>(ar, br, cr);
        accumulate<mr, nr>(c_ref, cr, c_ref);

        copy_arr<blis_data_c_t, mr * nr>(c_ref, c_temp[idx]);
      }
    }
  }

  vec_store_c<blis_data_c_t, NUM_ELEMENTS_PER_VEC_C, mf, nf, mr, nr>(y, x, m, n, mn, y_n, c_temp, c);
}

template<blis_size_t mr, blis_size_t nr,
         blis_size_t ms, blis_size_t ns,
         blis_size_t mc, blis_size_t nc, blis_size_t kc>
static void macro_kernel(const BlisVec<blis_data_a_t, NUM_ELEMENTS_PER_VEC_A> *a,
                         const BlisVec<blis_data_b_t, NUM_ELEMENTS_PER_VEC_B> *b,
                         BlisVec<blis_data_c_t, NUM_ELEMENTS_PER_VEC_C> *c,
                         blis_size_t m, blis_size_t n, blis_size_t k,
                         blis_size_t mn, blis_size_t mk, blis_size_t kn) {

  assert(mc % mr == 0);
  assert(nc % nr == 0);

  blis_data_a_t bram_block_a[mc * kc];
  blis_data_b_t bram_block_b[kc * nc];

  #pragma HLS ARRAY_PARTITION variable = bram_block_a type = cyclic factor = (mr / 2)
  #pragma HLS ARRAY_PARTITION variable = bram_block_b type = cyclic factor = (nr / 2)
  #pragma HLS BIND_STORAGE variable = bram_block_a type = ram_t2p impl = bram
  #pragma HLS BIND_STORAGE variable = bram_block_b type = ram_t2p impl = bram

  blis_size_t po_n = 0;

  loop_macro_mm_po:
  for (blis_size_t po = 0; po < k; po += kc) {
    #pragma HLS LOOP_FLATTEN off

    #pragma HLS LOOP_TRIPCOUNT min = (128 / kc) max = (1024 / kc)

    blis_size_t io_k = 0;
    blis_size_t io_n = 0;

    loop_macro_mm_io:
    for (blis_size_t io = 0; io < m; io += mc) {
      #pragma HLS LOOP_TRIPCOUNT min = (128 / mc) max = (1024 / mc)

      vec_load<blis_data_a_t, NUM_ELEMENTS_PER_VEC_A, mc, kc, mr, KR>(io, po, m, k, mk, io_k, a, bram_block_a);

      loop_macro_mm_jo:
      for (blis_size_t jo = 0; jo < n; jo += nc) {
        #pragma HLS LOOP_TRIPCOUNT min = (128 / nc) max = (1024 / nc)

        vec_load<blis_data_b_t, NUM_ELEMENTS_PER_VEC_B, kc, nc, KR, nr>(po, jo, k, n, kn, po_n, b, bram_block_b);

        blis_size_t ii_n = 0;

        assert(mc % (ms) == 0);
        assert(nc % (ns) == 0);

        loop_macro_mm_ii:
        for (blis_size_t ii = 0; ii < mc; ii += ms) {

          loop_macro_mm_ji:
          for (blis_size_t ji = 0; ji < nc; ji += ns) {

            micro_kernel<mr, nr, ms, ns, mc, nc, kc>(bram_block_a,
                                                     bram_block_b,
                                                     io + ii, jo + ji,
                                                     ii, ji,
                                                     m, n, mn, io_n + ii_n,
                                                     c);
          }

          ii_n += (n * ms);
        }
      }

      io_k += (k * mc);
      io_n += (n * mc);
    }

    po_n += (n * kc);
  }
}

void blis(int m, int n, int k,
          const BlisVec<blis_data_a_t, NUM_ELEMENTS_PER_VEC_A> *A,
          const BlisVec<blis_data_b_t, NUM_ELEMENTS_PER_VEC_B> *B,
          BlisVec<blis_data_c_t, NUM_ELEMENTS_PER_VEC_C> *C) {

  #pragma HLS INTERFACE s_axilite port = m
  #pragma HLS INTERFACE s_axilite port = k
  #pragma HLS INTERFACE s_axilite port = n
  #pragma HLS INTERFACE s_axilite port = return

  #pragma HLS INTERFACE m_axi port = A offset = slave \
                          bundle = BUNDLE_A depth = 8192 num_read_outstanding = 16 \
                          num_write_outstanding = 1 max_read_burst_length = 32 \
                          max_write_burst_length = 1 max_widen_bitwidth = 256

  #pragma HLS INTERFACE m_axi port = B offset = slave \
                          bundle = BUNDLE_B depth = 8192 num_read_outstanding = 16 \
                          num_write_outstanding = 1 max_read_burst_length = 32 \
                          max_write_burst_length = 1 max_widen_bitwidth = 256

  #pragma HLS INTERFACE m_axi port = C offset = slave \
                          bundle = BUNDLE_C depth = 8192 num_read_outstanding = 8 \
                          num_write_outstanding = 8 max_read_burst_length = 32 \
                          max_write_burst_length = 32 max_widen_bitwidth = 256

  #pragma HLS ALLOCATION operation instances = mul limit = 1

  assert(m % NUM_ELEMENTS_PER_VEC_A == 0);
  assert(n % NUM_ELEMENTS_PER_VEC_B == 0);
  assert(k % NUM_ELEMENTS_PER_VEC_C == 0);

  blis_size_t kn = k * n;
  blis_size_t mk = m * k;
  blis_size_t mn = m * n;

  macro_kernel<M_R, N_R, M_S, N_S, M_C, N_C, K_C>(A, B, C, m, n, k, mn, mk, kn);
}
