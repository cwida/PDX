# ADSampling
add_executable(BenchmarkNaryIVFADSampling bench_adsampling/nary_ivf_adsampling.cpp)
add_executable(BenchmarkNaryIVFADSamplingSIMD bench_adsampling/nary_ivf_adsampling_simd.cpp)
add_executable(BenchmarkPDXADSampling bench_adsampling/pdx_ivf_adsampling.cpp)

# Phases experiment
add_executable(BenchmarkPhasesNaryADSampling bench_adsampling/nary_ivf_adsampling_simd_phases.cpp)
add_executable(BenchmarkPhasesPDXADSampling bench_adsampling/pdx_ivf_adsampling_phases.cpp)
add_executable(BenchmarkPhasesPDXBSA bench_bsa/pdx_ivf_bsa_phases.cpp)
add_executable(BenchmarkPhasesNaryBSA bench_bsa/nary_ivf_bsa_simd_phases.cpp)
add_executable(BenchmarkPhasesPDXBOND bench_bond/pdx_bond_ivf_phases.cpp)

# ADSampling threshold adaptiveness experiment
add_executable(BenchmarkAdaptivenessInc bench_adsampling/pdx_ivf_adsampling_adaptive_inc.cpp)
add_executable(BenchmarkAdaptiveness32 bench_adsampling/pdx_ivf_adsampling_adaptive_32.cpp)

# BSA
add_executable(BenchmarkNaryIVFBSASIMD bench_bsa/nary_ivf_bsa_simd.cpp)
add_executable(BenchmarkPDXBSA bench_bsa/pdx_ivf_bsa.cpp)

# BOND
add_executable(BenchmarkPDXIVFBOND ./bench_bond/pdx_bond_ivf.cpp)
add_executable(BenchmarkPDXBOND ./bench_bond/pdx_bond.cpp)

# Linear
add_executable(BenchmarkPDXLinearScan ./bench_linear/pdx_linear_scan.cpp)
add_executable(BenchmarkNaryIVFLinearScan bench_linear/nary_ivf_linear_scan.cpp)

# Kernels experiment
add_executable(KernelPDXL1 ./bench_kernels/pdx_l1.cpp)
add_executable(KernelPDXL2 ./bench_kernels/pdx_l2.cpp)
add_executable(KernelPDXIP ./bench_kernels/pdx_ip.cpp)
add_executable(KernelNaryL1 ./bench_kernels/nary_l1.cpp)
add_executable(KernelNaryL2 ./bench_kernels/nary_l2.cpp)
add_executable(KernelNaryIP ./bench_kernels/nary_ip.cpp)
add_executable(PureScanGATHER bench_kernels/nary_gather.cpp)
add_executable(BenchmarkCorrectness ./bench_kernels/pdx_correctness.cpp)

# Block Study
add_executable(KernelPDXL2_16 ./bench_kernels/pdx_l2_16.cpp)
add_executable(KernelPDXL2_32 ./bench_kernels/pdx_l2_32.cpp)
add_executable(KernelPDXL2_64 ./bench_kernels/pdx_l2_64.cpp)
add_executable(KernelPDXL2_128 ./bench_kernels/pdx_l2_128.cpp)
add_executable(KernelPDXL2_256 ./bench_kernels/pdx_l2_256.cpp)
add_executable(KernelPDXL2_512 ./bench_kernels/pdx_l2_512.cpp)

# DSM
add_executable(BenchmarkDSMADSampling ./bench_adsampling/pdx_ivf_adsampling_dsm.cpp)
add_executable(BenchmarkDSMBSA ./bench_bsa/pdx_ivf_bsa_dsm.cpp)
add_executable(BenchmarkDSMIVFBOND ./bench_bond/pdx_bond_ivf_dsm.cpp)
add_executable(BenchmarkDSMLinearScan ./bench_linear/dsm_linear_scan.cpp)
add_executable(BenchmarkDSMBOND ./bench_bond/pdx_bond_dsm.cpp)


# Not used for benchmarks in paper
# add_executable(BenchmarkHorLinearScan hor_linear_scan.cpp)
# add_executable(BenchmarkHorADSampling hor_adsampling.cpp)
# add_executable(BenchmarkPDXBONDTuning ./bench_selectivity/pdx_bond_parameters_tuning.cpp)
# add_executable(BenchmarkPDXADSamplingTuning ./bench_selectivity/pdx_adsampling_parameters_tuning.cpp)
