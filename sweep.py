import os
import math
import tempfile


MR_MIN = 2
MR_MAX = 16

NR_MIN = 2
NR_MAX = 16

MS_F_MIN = 1
MS_F_MAX = 16

NS_F_MIN = 1
NS_F_MAX = 16

KC_MIN = 8
KC_MAX = 1024

MC_F_MIN = 1
MC_F_MAX = 256

NC_F_MIN = 1
NC_F_MAX = 256

SETUP_CMD = 'source ~/xilinx/Vitis/2024.1/settings64.sh'
CSIM_CMD = 'vitis-run --mode hls --csim --config ./blis_hls/hls_config.cfg --work_dir blis_hls'
CSYN_CMD = 'v++ -c --mode hls --config ./blis_hls/hls_config.cfg --work_dir blis_hls'
PACK_CMD = 'vitis-run --mode hls --package --config ./blis_hls/hls_config.cfg --work_dir blis_hls'

HEADER_FILE = 'hw_blis.h'
SOURCE_FILES = [ 'hw_blis.cpp', 'hw_blis_tb.cpp' ]

BLIS_HEADER_TEMPLATE = '''
#pragma once

#define K_C {kc}

#define M_R {mr}
#define N_R {nr}

#define M_S {ms}
#define N_S {ns}

#define M_C {mc}
#define N_C {nc}

#define NUM_ELEMENTS_PER_VEC_A 8
#define NUM_ELEMENTS_PER_VEC_B 8
#define NUM_ELEMENTS_PER_VEC_C 8

#define F0(src,dst) dst = src
#define F1(src,dst) dst = src
#define F2(src,dst) dst = src
#define F3(src,dst) dst = src
#define F5(src,dst) dst = src
#define OP1(a,b,r)  (r) = (a) {op_1} (b)
#define OP2(a,b,r)  (r) = (a) {op_2} (b)

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
'''


class Utilization:
    def __init__(self):
        self.bram = -1
        self.dsp = -1
        self.ff = -1
        self.lut = -1
        self.time = -1
        self.synthesizable = False

    def __init__(self, bram: float, dsp: float, ff: float, lut: float, time: float):
        self.bram = bram
        self.dsp = dsp
        self.ff = ff
        self.lut = lut
        self.time = time
        self.synthesizable = self.bram <= 100 and self.dsp <= 100 and self.ff <= 100 and self.lut <= 100

    def synthesizable(self) -> bool:
        return self.synthesizable


def is_pow2(a: int) -> bool:
    return a & (a - 1) == 0


def pow2_range(a: int, b: int, b_inclusive: bool = True) -> Iterator[int]:
    assert a < b

    b = b + 1 if b_inclusive and is_pow2(b) else b

    al = math.ceil(math.log2(a))
    bl = math.ceil(math.log2(b))

    for p in range(al, bl):
        yield 1 << p


def generate_header(kc: int, mc_f: int, nc_f: int, ms_f: int, ns_f: int, mr: int, nr: int) -> str:
    return BLIS_HEADER_TEMPLATE.format(kc=kc, mr=mr, nr=nr, ms=(mr * ms_f), ns=(nr * ns_f), mc=(mr * ms_f * mc_f), nc=(nr * ns_f * nc_f))


def test(kc: int, mc_f: int, nc_f: int, ms_f: int, ns_f: int, mr: int, nr: int) -> Utilization:
    with tempfile.TemporaryDirectory(prefix=f'blis_{kc}_{mc_f}_{nc_f}_{ms_f}_{ns_f}_{mr}_{nr}') as work_dir:
        # Copy the files to the working directory
        sources = ' '.join([ f'./{f}' for f in SOURCE_FILES ])

        os.system(f'cp {sources} {work_dir}')

        # Override the header file with the new values
        with open(f'{work_dir}/{HEADER_FILE}', 'w') as header_file:
            header_file.write(generate_header(kc, mc_f, nc_f, ms_f, ns_f, mr, nr))

        # Execute the commands
        ret = os.system(f'cd {work_dir} && {SETUP_CMD} && {CSIM_CMD} && {CSYN_CMD}')

        if ret != 0:
            return Utilization()

        # Open the synthesis reports
        with open(f'{work_dir}/hls/syn/report/blis_csynth.xml', r) as reports_file:
            pass

        return Utilization(100, 100, 100, 100, 0.24)


def run():
    for kc in pow2_range(KC_MIN, KC_MAX):
        for mc_f in pow2_range(MC_F_MIN, MC_F_MAX):
            for nc_f in pow2_range(NC_F_MIN, NC_F_MAX):
                for ms_f in pow2_range(MS_F_MIN, MS_F_MAX):
                    for ns_f in pow2_range(NS_F_MIN, NS_F_MAX):
                        for mr in pow2_range(MR_MIN, MR_MAX):
                            for nr in pow2_range(NR_MIN, NR_MAX):
                                test(kc, mc_f, nc_f, ms_f, ns_f, mr, nr)
                                return


if __name__ == '__main__':
    run()
