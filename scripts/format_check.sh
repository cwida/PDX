#!/bin/bash

# Check C++ file formatting using clang-format
# This script checks if all .cpp, .h, and .hpp files are properly formatted
# Exits with error code 1 if any files need formatting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Checking C++ formatting in SuperKMeans project..."
echo "Project root: $PROJECT_ROOT"

REQUIRED_VERSION="18.1.8"

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
unformatted_files=()

for dir in "${DIRECTORIES[@]}"; do
    dir_path="$PROJECT_ROOT/$dir"

    if [ ! -d "$dir_path" ]; then
        echo "Warning: Directory $dir does not exist, skipping..."
        continue
    fi

    echo "Checking directory: $dir"

    for ext in "${EXTENSIONS[@]}"; do
        while IFS= read -r -d '' file; do
            total_files=$((total_files + 1))

            # Run clang-format and compare with original
            if ! diff -q <(clang-format "$file") "$file" > /dev/null 2>&1; then
                echo "  ✗ ${file#$PROJECT_ROOT/}"
                unformatted_files+=("$file")
            fi
        done < <(find "$dir_path" -type f -name "*.$ext" -print0)
    done
done

echo ""
echo "Checked $total_files file(s)."

# Report results
if [ ${#unformatted_files[@]} -eq 0 ]; then
    echo "✓ All files are properly formatted!"
    exit 0
else
    echo "✗ Found ${#unformatted_files[@]} file(s) that need formatting:"
    for file in "${unformatted_files[@]}"; do
        echo "  - ${file#$PROJECT_ROOT/}"
    done
    echo ""
    echo "Run './scripts/format.sh' to fix formatting issues."
    exit 1
fi
