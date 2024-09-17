import os
import sys
import math
import time
import tempfile
import random

from typing import IO
from typing import Iterator
from typing import Tuple
from typing import TypeVar
from xml.etree import ElementTree
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from fabric import Connection

from paramiko.pkey import PKey


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
CSIM_CMD = 'vitis-run --mode hls --csim --config ./hls_config.cfg --work_dir blis_hls'
CSYN_CMD = 'v++ -c --mode hls --config ./hls_config.cfg --work_dir blis_hls'
PACK_CMD = 'vitis-run --mode hls --package --config ./hls_config.cfg --work_dir blis_hls'

HLS_DIR = 'blis_hls'
HEADER_FILE = 'hw_blis.h'
SOURCE_FILES = [ HEADER_FILE, 'hls_config.cfg', 'hw_blis.cpp', 'hw_blis_tb.cpp' ]

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


def describe(config: Config) -> str:
    return f'blis_{config.kc}_{config.mc_f}_{config.nc_f}_{config.ms_f}_{config.ns_f}_{config.mr}_{config.nr}'


def passphrase() -> str:
    return os.getenv('OU_CLOUD_PASSPHRASE')


def process_configs(args: Tuple[Iterator[Config], str, str, int, int, str]) -> list[Tuple[Config, Utilization]]:

    (configs, key_file_path, worker, wid, cid, logs_path) = args

    configs = list(configs)

    logs_dir = os.path.dirname(logs_path)
    logs_file_basename = os.path.basename(logs_path)

    logs_file_name, logs_file_ext = os.path.splitext(logs_file_basename)

    results = []

    pkey = PKey.from_path(key_file_path, passphrase().encode('utf-8'))

    with open(f'{logs_dir}/{logs_file_name}_w{wid}_c{cid}{logs_file_ext}', 'w') as result_log_file:
        # print(f'rank: {cid}@{wid} testing {len(configs)} configs, logging to: {log_file.name}')

        with Connection(f'arya@{worker}', connect_kwargs={'pkey': pkey}) as con:
            for cfg in configs:
                config_desc = describe(cfg)

                with open(f'{logs_dir}/{logs_file_name}_{config_desc}_w{wid}_c{cid}{logs_file_ext}', 'w') as log_file:
                    util = work(cfg, con, log_file)
                    result_log_file.write(f'{cfg.kc},{cfg.mc_f},{cfg.nc_f},{cfg.ms_f},{cfg.ns_f},{cfg.mr},{cfg.nr},{util.bram},{util.dsp},{util.ff},{util.lut},{util.latency},{util.latency_unit}\n')
                    results.append((cfg, util))

    return results


def work(config: Config, con: Connection, log_file: IO) -> Utilization:
    config_desc = describe(config)

    with tempfile.TemporaryDirectory(prefix=f'{config_desc}_') as work_dir:
        print(f'config: {config_desc}: begin')

        # NOTE: Currently generating the header file and copying all
        #       the source files. This can always be changed to generating
        #       all the files and config files for generation.

        # Copy the files to the working directory
        sources = ' '.join([ f'./{HLS_DIR}/{f}' for f in SOURCE_FILES ])

        ret = os.system(f'cp {sources} {work_dir}')

        if ret != 0:
            print(f'config: {config_desc}: cannot copy source files locally')
            return Utilization()

        # Generate the header file and overwrite the original one
        with open(f'{work_dir}/{HEADER_FILE}', 'w') as header_file:
            header_file.write(generate_header(config))

        try:
            res = con.run('ip addr', out_stream=log_file)

            if res.exited != 0:
                raise Exception(f'error describing the machine, return code: {res.exited}')
        except Exception as e:
            print(f'error occurred while describing the machine, proceeding ahead, {e}')

        # create a temp directory in remote
        r_tmp_dir = None

        try:
            res = con.run('mktemp -d', out_stream=log_file)

            if res.exited != 0:
                raise Exception(f'error occurred while creating temp directory, return code: {res.exited}')

            r_tmp_dir = res.stdout.lstrip().rstrip()
        except Exception as e:
            print(f'error occurred {e}')
            return Utilization()

        assert(r_tmp_dir is not None)

        # push the files to the remote
        for source in SOURCE_FILES:
            try:
                res = con.put(f'{work_dir}/{source}', f'{r_tmp_dir}/')
            except Exception as e:
                print(f'error occured while copying files to remote: {e}')
                return Utilization()

        # Execute the commands
        # timeout = 20 mins
        start = time.time()

        try:
            res = con.run(f'timeout --signal=SIGKILL 20m docker run --rm -v {r_tmp_dir}:/opt/data xilinx /bin/bash -c "{SETUP_CMD} && cd /opt/data && {CSIM_CMD} && {CSYN_CMD}"', out_stream=log_file)

            if res.exited != 0:
                raise Exception(f'''
                                Error occured while synthesizing the config:
                                return code: {res.exited}
                                ---------------------------------------------
                                stdout:
                                {res.stdout}
                                ---------------------------------------------
                                stderr:
                                {res.stderr}''')
        except Exception as e:
            end = time.time()

            print(f'config: {config_desc}: failed, elapsed: {end - start}s, generated by: {con}')
            return Utilization()

        # Copy the results file back to the local
        try:
            res = con.get(f'{r_tmp_dir}/blis_hls/hls/syn/report/blis_csynth.xml', f'{work_dir}/')
        except Exception as e:
            print(f'Error occurred while getting the synth results: {e}')
            return Utilization()

        # delete the temp directory
        try:
            res = con.run(f'docker run --rm -v {r_tmp_dir}:/opt/data xilinx /bin/bash -c "rm -rf /opt/data/*" && rm -rf {r_tmp_dir}')

            if res.exited != 0:
                raise Exception(f'Error while deleting the temporary directory, ret code: {res.exited}')
        except Exception as e:
            print(f'Failed to delete the remote tmp directory: {e}')

        # Open the synthesis reports
        report_tree = ElementTree.parse(f'{work_dir}/blis_csynth.xml')

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

        print(f'config: {config_desc}: bram: {bram}, dsp: {dsp}, ff: {ff}, lut: {lut}, latency: {latency} {latency_unit}, generated by: {con}')

        return Utilization(bram, dsp, ff, lut, latency, latency_unit)


T = TypeVar('T')
def chunkify(data: list[T], nchunks: int) -> Iterator[list[T]]:
    dlen = len(data)

    clen = dlen // nchunks
    head = dlen % nchunks

    start = 0

    for r in range(nchunks):
        end = (start + clen) + (1 if r < head else 0)

        yield data[start : end]

        start = end


def synthesize(args: Tuple[list[Config], str, str, int, int]) -> list[Tuple[Config, Utilization]]:

    (configs, key_file_path, slave, slave_id, logs_path) = args

    print(f'rank: {slave_id} testing {len(configs)} configurations')

    nremote_processes = 16

    # Process 16 items at a time
    # Each connection will run 16 configurations at a time
    chunks = chunkify(configs, nremote_processes)

    # NOTE: Currently opening (nremote_processes) connections
    #       one per process essentially. But, it could be made to
    #       be one per all of the processes. The reasoning behind this is:
    #       I'm not sure about the MT-Safety of the underlying fabric arch.
    #       I am not sure if invoking two run commands simultaneously
    #       could cause some issues and potential race conditions.

    results = []

    with ThreadPoolExecutor(max_workers=nremote_processes) as pool:
        for chunk_results in pool.map(process_configs, [ (chunk, key_file_path, slave, slave_id, idx, logs_path) for (idx, chunk) in enumerate(chunks)]):
            results.extend(chunk_results)

    return results


def should_process(config: Config) -> bool:
    # inner kernel size filtering
    inner_kernel_size = config.mr * config.nr

    kernel_matches = inner_kernel_size >= 128 and inner_kernel_size <= 512

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


def gen_configs() -> list[Config]:
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

    return configs


def run(key_file_path:str, logs_path: str, results_path: str):
    workers = [ f'10.0.2.{w}' for w in range(3, 19) ]

    print(f'using workers: {workers}')

    configs = gen_configs()

    # shuffle to make sure that not all
    # heavy hitting configs are on one side
    random.shuffle(configs)

    print('Testing')

    for idx, config in enumerate(configs):
        print(f'{idx}: kc={config.kc} mc_f={config.mc_f} nc_f={config.nc_f} ms_f={config.ms_f} ns_f={config.ns_f} mr={config.mr} nr={config.nr}')

    print(f'Testing {len(configs)} configurations')

    nworkers = len(workers)

    chunks = chunkify(configs, nworkers)

    results = []

    with Pool(processes=nworkers) as process_pool:
        for chunk_results in process_pool.imap(synthesize, [ (chunk, key_file_path, worker, idx, logs_path) for (idx, (worker, chunk)) in enumerate(zip(workers, chunks)) ]):
            results.extend(chunk_results)

    with open(results_path, 'w') as results_file:
        print(f'results to: {results_file.name}')

        results_file.write('kc,mc_f,nc_f,ms_f,ns_f,mr,nr,bram,dsp,ff,lut,latency,latency_units\n')

        for config, utilization in results:
            results_file.write(f'{config.kc},{config.mc_f},{config.nc_f},{config.ms_f},{config.ns_f},{config.mr},{config.nr},{utilization.bram},{utilization.dsp},{utilization.ff},{utilization.lut},{utilization.latency},{utilization.latency_unit}\n')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Incorrect usage')
        print(f'Correct usage: {sys.argv[0]} <key file path> <log file path> <results path>')
        exit(1)

    if passphrase() is None:
        print('Using no passphrase. Are you sure?')
        exit(2)

    key_file_path = sys.argv[1]
    log_file_path = sys.argv[2]
    results_path = sys.argv[3]

    run(key_file_path, log_file_path, results_path)
