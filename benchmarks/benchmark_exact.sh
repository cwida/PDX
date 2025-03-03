#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <python-command>"
  exit 1
fi

set -e

python_command="$1"
datasets=("nytimes-16-angular" "glove-50-angular" "deep-image-96-angular" "sift-128-euclidean" "glove-200-angular" "msong-420" "har-561" "contriever-768" "instructorxl-arxiv-768" "fashion-mnist-784-euclidean" "gist-960-euclidean" "openai-1536-angular")

make BenchmarkPDXLinearScan
make BenchmarkPDXBOND

for dataset in "${datasets[@]}"
do
    echo "Exact SKLearn | $dataset"
    $python_command ./benchmarks/python_scripts/exact_sklearn.py ${dataset}
    echo "Exact FAISS | $dataset"
    $python_command ./benchmarks/python_scripts/exact_faiss.py ${dataset}
    echo "Exact USearch | $dataset"
    $python_command ./benchmarks/python_scripts/exact_usearch.py ${dataset}

    echo "PDX LinearScan | $dataset"
    ./benchmarks/BenchmarkPDXLinearScan ${dataset}
    echo "PDX BOND [Distance to Means] | $dataset"
    ./benchmarks/BenchmarkPDXBOND ${dataset} 1
    echo "PDX BOND [Decreasing] | $dataset"
    ./benchmarks/BenchmarkPDXBOND ${dataset} 2
    echo "PDX BOND [Distance to means+] | $dataset"
    ./benchmarks/BenchmarkPDXBOND ${dataset} 3
done
