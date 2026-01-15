#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <python-command>"
  exit 1
fi

set -e

python_command="$1"

# Update package lists
echo "Updating package lists..."
sudo apt-get update

if command -v docker 2>&1 >/dev/null
then
    echo "docker is installed"
else
    sudo apt-get install curl
    curl -fsSL https://get.docker.com | sudo sh
    docker --version
fi

if command -v docker-compose 2>&1 >/dev/null
then
    echo "docker-compose is installed"
else
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    docker-compose --version
fi

sudo systemctl start docker

docker-compose -f ./benchmarks/milvus/docker-compose.yml up -d

${python_command} ./benchmarks/python_scripts/exact_milvus.py
${python_command} ./benchmarks/python_scripts/ivf_milvus_build.py
${python_command} ./benchmarks/python_scripts/ivf_milvus.py

docker-compose -f ./benchmarks/milvus/docker-compose.yml stop

docker-compose -f ./benchmarks/milvus/docker-compose.yml down