#!/bin/bash

set -e

datasets=("sift-128-euclidean" "msong-420" "gist-960-euclidean" "deep-image-96-angular" "instructorxl-arxiv-768" "openai-1536-angular" "glove-200-angular" "contriever-768")

make BenchmarkAdaptivenessInc
make BenchmarkAdaptiveness32

for dataset in "${datasets[@]}" 
do
  echo "Benchmark ADS adaptiveness Inc | $dataset"
  ./benchmarks/BenchmarkAdaptivenessInc ${dataset} 512
done

for dataset in "${datasets[@]}"
do
  echo "Benchmark ADS adaptiveness 32 | $dataset"
  ./benchmarks/BenchmarkAdaptiveness32 ${dataset} 512
done
