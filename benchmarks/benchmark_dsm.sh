#!/bin/bash

set -e

datasets=("nytimes-16-angular" "glove-50-angular" "deep-image-96-angular" "sift-128-euclidean" "glove-200-angular" "msong-420" "har-561" "contriever-768" "instructorxl-arxiv-768" "fashion-mnist-784-euclidean" "gist-960-euclidean" "openai-1536-angular")

make BenchmarkDSMBOND
make BenchmarkDSMLinearScan

for dataset in "${datasets[@]}" 
do

  echo "DSM Linear Scan | $dataset"
  ./benchmarks/BenchmarkDSMLinearScan ${dataset}

  echo "DSM BOND | $dataset"
  ./benchmarks/BenchmarkDSMBOND ${dataset} 1 # Distance to means

done
