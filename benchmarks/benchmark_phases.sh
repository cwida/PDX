#!/bin/bash

set -e

datasets=("sift-128-euclidean" "msong-420" "instructorxl-arxiv-768" "gist-960-euclidean" "openai-1536-angular")

make BenchmarkPhasesNaryADSampling
make BenchmarkPhasesPDXADSampling
make BenchmarkPhasesNaryBSA
make BenchmarkPhasesPDXBSA
make BenchmarkPhasesPDXBOND

for dataset in "${datasets[@]}" 
do

  echo "Nary ADSampling SIMD Phases | $dataset"
  ./benchmarks/BenchmarkPhasesNaryADSampling ${dataset}

  echo "PDX ADSampling Phases | $dataset"
  ./benchmarks/BenchmarkPhasesPDXADSampling ${dataset}

  echo "Nary BSA SIMD Phases | $dataset"
  ./benchmarks/BenchmarkPhasesNaryBSA ${dataset}

  echo "PDX BSA Phases | $dataset"
  ./benchmarks/BenchmarkPhasesPDXBSA ${dataset}

  echo "PDX BOND Phases | $dataset"
  ./benchmarks/BenchmarkPhasesPDXBOND ${dataset} 0 5

done
