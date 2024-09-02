import os
import math
import tempfile

from typing import Iterator
from xml.etree import ElementTree
from multiprocessing import Pool


MR_MIN = 2
MR_MAX = 32

NR_MIN = 2
NR_MAX = 32

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
CSIM_CMD = 'vitis-run --mode hls --csim --config ./hls_config.cfg --work_dir blis_hls > /dev/null'
CSYN_CMD = 'v++ -c --mode hls --config ./hls_config.cfg --work_dir blis_hls > /dev/null'
PACK_CMD = 'vitis-run --mode hls --package --config ./hls_config.cfg --work_dir blis_hls'

HLS_DIR = 'blis_hls'
HEADER_FILE = 'hw_blis.h'
SOURCE_FILES = [ 'hls_config.cfg', 'hw_blis.cpp', 'hw_blis_tb.cpp' ]

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
#define OP1(a,b,r)  {op_1}
#define OP2(a,b,r)  {op_2}

typedef float blis_data_t;
typedef blis_data_t blis_data_a_t;
typedef blis_data_t blis_data_b_t;
typedef blis_data_t blis_data_c_t;
typedef unsigned blis_size_t;

template<typename T, blis_size_t N>
struct __attribute__((aligned(N * sizeof(T)))) BlisVec {{
    T data[N];
}};

extern "C" {{

void blis(int m, int n, int k,
          const BlisVec<blis_data_a_t, NUM_ELEMENTS_PER_VEC_A> *A,
          const BlisVec<blis_data_b_t, NUM_ELEMENTS_PER_VEC_B> *B,
          BlisVec<blis_data_c_t, NUM_ELEMENTS_PER_VEC_C> *C);

}}
'''


def res_chk(util: float) -> bool:
    return util >= 0 and util <= 100


class Config:
    def __init__(self, kc: int, mc_f: int, nc_f: int, ms_f: int, ns_f: int, mr: int, nr: int):
        self.kc = kc
        self.mc_f = mc_f
        self.nc_f = nc_f
        self.ms_f = ms_f
        self.ns_f = ns_f
        self.mr = mr
        self.nr = nr


class Utilization:
    def __init__(self, bram: float = -1, dsp: float = -1, ff: float = -1, lut: float = -1, latency: float = -1, latency_unit: str = '?'):
        self.bram = bram
        self.dsp = dsp
        self.ff = ff
        self.lut = lut
        self.latency = latency
        self.latency_unit = latency_unit
        self.synthesizable = res_chk(self.bram) and res_chk(self.dsp) and res_chk(self.ff) and res_chk(self.lut)

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


def generate_header(config: Config) -> str:
    kc = config.kc
    mc_f = config.mc_f
    nc_f = config.nc_f
    ms_f = config.ms_f
    ns_f = config.ns_f
    mr = config.mr
    nr = config.nr

    return BLIS_HEADER_TEMPLATE.format(kc=kc, mr=mr, nr=nr, ms=(mr * ms_f), ns=(nr * ns_f), mc=(mr * ms_f * mc_f), nc=(nr * ns_f * nc_f), op_1='(r) = (a) * (b)', op_2='(r) = (a) + (b)')


def synthesize(config: Config) -> Utilization:
    config_desc = f'blis_{config.kc}_{config.mc_f}_{config.nc_f}_{config.ms_f}_{config.ns_f}_{config.mr}_{config.nr}'

    with tempfile.TemporaryDirectory(prefix=f'{config_desc}_') as work_dir:
        print(f'config: {config_desc}: begin')

        # Generate the header file
        with open(f'{work_dir}/{HEADER_FILE}', 'w') as header_file:
            header_file.write(generate_header(config))

        # Copy the files to the working directory
        sources = ' '.join([ f'./{HLS_DIR}/{f}' for f in SOURCE_FILES ])

        ret = os.system(f'cp {sources} {work_dir}')

        if ret != 0:
            print(f'config: {config_desc}: cannot copy source files')
            return Utilization()

        # Execute the commands
        ret = os.system(f'/bin/bash -c "cd {work_dir} && {SETUP_CMD} && {CSIM_CMD} && {CSYN_CMD}"')

        if ret != 0:
            print(f'config: {config_desc}: failed to synthesize')
            return Utilization()

        # Open the synthesis reports
        report_tree = ElementTree.parse(f'{work_dir}/blis_hls/hls/syn/report/blis_csynth.xml')

        report_root = report_tree.getroot()

        resource_estimates = report_root.find('AreaEstimates')

        utilized_resources = resource_estimates.find('Resources')
        available_resources = resource_estimates.find('AvailableResources')

        utilized_bram = float(utilized_resources.find('BRAM_18K').text)
        utilized_dsp = float(utilized_resources.find('DSP').text)
        utilized_ff = float(utilized_resources.find('FF').text)
        utilized_lut = float(utilized_resources.find('LUT').text)

        available_bram = float(available_resources.find('BRAM_18K').text)
        available_dsp = float(available_resources.find('DSP').text)
        available_ff = float(available_resources.find('FF').text)
        available_lut = float(available_resources.find('LUT').text)

        performance_estimates = report_root.find('PerformanceEstimates')

        latency = float(performance_estimates.find('SummaryOfOverallLatency').find('Worst-caseLatency').text)
        latency_unit = performance_estimates.find('SummaryOfOverallLatency').find('unit').text

        bram = utilized_bram / available_bram
        dsp = utilized_dsp / available_dsp
        ff = utilized_ff / available_ff
        lut = utilized_lut / available_lut

        print(f'config: {config_desc}: bram: {bram}, dsp: {dsp}, ff: {ff}, lut: {lut}, latency: {latency} {latency_unit}')

        return Utilization(bram, dsp, ff, lut, latency)


def should_process(config: Config) -> bool:
    # inner kernel size filtering
    inner_kernel_size = config.mr * config.nr

    kernel_matches = inner_kernel_size >= 128

    # c (lutram) filtering
    ms = config.ms_f * config.mr
    ns = config.ns_f * config.nr

    # lut is directly proportional to ms * ns
    c_lut = ms * ns

    c_lut_limit = 32 * 1024

    c_lut_matches = c_lut <= c_lut_limit

    # bram filtering
    mc = config.mc_f * config.ms_f * config.mr
    nc = config.nc_f * config.ns_f * config.nr

    n_cache_elements = (mc * config.kc) + (config.kc * nc)

    # 1 bram element can fit 512 words
    estimated_bram_utilization = n_cache_elements / 512

    # interfaces use upto 60 bram elements
    estimated_bram_utilization += 60

    # FPGA has 280 bram units
    bram_units_in_fpga = 280

    bram_lower_limit = bram_units_in_fpga * 0.8
    bram_upper_limit = bram_units_in_fpga * 1.1

    bram_matches = estimated_bram_utilization >= bram_lower_limit and estimated_bram_utilization <= bram_upper_limit

    matches = kernel_matches and c_lut_matches and bram_matches

    return matches


def run():
    count = 0

    configs = []

    for kc in pow2_range(KC_MIN, KC_MAX):
        for mc_f in pow2_range(MC_F_MIN, MC_F_MAX):
            for nc_f in pow2_range(NC_F_MIN, NC_F_MAX):
                for ms_f in pow2_range(MS_F_MIN, MS_F_MAX):
                    for ns_f in pow2_range(NS_F_MIN, NS_F_MAX):
                        for mr in pow2_range(MR_MIN, MR_MAX):
                            for nr in pow2_range(NR_MIN, NR_MAX):
                                config = Config(kc, mc_f, nc_f, ms_f, ns_f, mr, nr)

                                if should_process(config):
                                    configs.append(config)
                                    count += 1

    print('Testing')

    for idx, config in enumerate(configs):
        print(f'{idx}: kc={config.kc} mc_f={config.mc_f} nc_f={config.nc_f} ms_f={config.ms_f} ns_f={config.ns_f} mr={config.mr} nr={config.nr}')

    print(f'Testing {count} configurations')

    with Pool(processes=16) as process_pool:
        utilizations = process_pool.map(synthesize, configs)

    print('n,kc,mc_f,nc_f,ms_f,ns_f,mr,nr,bram,dsp,ff,lut,latency,latency_units')

    for idx, config, utilization in enumerate(zip(configs, utilizations)):
        print(f'{idx},{config.kc},{config.mc_f},{config.nc_f},{config.ms_f},{config.ns_f},{config.mr},{config.nr},{utilization.bram},{utilization.dsp},{utilization.ff},{utilization.lut},{utilization.latency},{utilization.latency_unit}')


if __name__ == '__main__':
    run()
