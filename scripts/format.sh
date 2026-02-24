#!/bin/bash

# Format C++ files using clang-format
# This script formats all .cpp, .h, and .hpp files in the project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Formatting C++ files in SuperKMeans project..."
echo "Project root: $PROJECT_ROOT"

REQUIRED_VERSION="18.1.8"

# Check if clang-format is available
if ! command -v clang-format &> /dev/null; then
    echo "Error: clang-format not found. Please install it first."
    exit 1
fi

CURRENT_VERSION=$(clang-format --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
if [ "$CURRENT_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "Error: clang-format version $REQUIRED_VERSION required, but found $CURRENT_VERSION"
    echo "Install the correct version: pip install clang-format==$REQUIRED_VERSION"
    exit 1
fi

DIRECTORIES=(
    "include"
    "python"
    "tests"
    "benchmarks"
    "examples"
)

EXTENSIONS=("cpp" "h" "hpp")

total_files=0

# Format files in each directory
for dir in "${DIRECTORIES[@]}"; do
    dir_path="$PROJECT_ROOT/$dir"

    if [ ! -d "$dir_path" ]; then
        echo "Warning: Directory $dir does not exist, skipping..."
        continue
    fi

    echo "Processing directory: $dir"

    for ext in "${EXTENSIONS[@]}"; do
        while IFS= read -r -d '' file; do
            echo "  Formatting: ${file#$PROJECT_ROOT/}"
            clang-format -i "$file"
            total_files=$((total_files + 1))
        done < <(find "$dir_path" -type f -name "*.$ext" -print0)
    done
done

echo ""
echo "Done! Formatted $total_files file(s)."
