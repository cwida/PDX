#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <python-command>"
  exit 1
fi

set -e

python_command="$1"
datasets=("nytimes-16-angular" "glove-50-angular" "deep-image-96-angular" "sift-128-euclidean" "glove-200-angular" "msong-420" "contriever-768" "instructorxl-arxiv-768" "gist-960-euclidean" "openai-1536-angular")

make BenchmarkNaryIVFADSampling
make BenchmarkNaryIVFADSamplingSIMD
make BenchmarkPDXADSampling
make BenchmarkNaryIVFBSASIMD
make BenchmarkPDXBSA
make BenchmarkPDXIVFBOND
make BenchmarkNaryIVFLinearScan

for dataset in "${datasets[@]}" 
do

#  echo "IVF MILVUS | $dataset"
#  ${python_command} ./benchmarks/python_scripts/ivf_milvus.py ${dataset}

  echo "IVF FAISS | $dataset"
  ${python_command} ./benchmarks/python_scripts/ivf_faiss.py ${dataset}

  echo "Nary IVF LinearScan Scalar | $dataset"
  ./benchmarks/BenchmarkNaryIVFLinearScan ${dataset}

  echo "Nary IVF ADSampling Scalar | $dataset"
  ./benchmarks/BenchmarkNaryIVFADSampling ${dataset}

  echo "Nary IVF ADSampling SIMD | $dataset"
  ./benchmarks/BenchmarkNaryIVFADSamplingSIMD ${dataset}
  echo "PDX IVF ADSampling | $dataset"
  ./benchmarks/BenchmarkPDXADSampling ${dataset}

  echo "Nary IVF BSA SIMD | $dataset"
  ./benchmarks/BenchmarkNaryIVFBSASIMD ${dataset}
  echo "PDX IVF BSA | $dataset"
  ./benchmarks/BenchmarkPDXBSA ${dataset}

  echo "PDX IVF BOND [Dimension Zones] | $dataset"
  ./benchmarks/BenchmarkPDXIVFBOND ${dataset} 0 5 # Dimension zones

done
