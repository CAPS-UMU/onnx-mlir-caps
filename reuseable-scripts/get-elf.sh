#!/usr/bin/env bash

set -e

# ---- Configurable paths ----
CLANG=/workdir/llvm-project/build-v20/bin/clang++
RUNTIME_LIB=/workdir/oml-vect-docker/runtime/libcruntime.a
INCLUDE_DIR=/workdir/oml-vect-docker/include

# ---- Required inputs ----
input_asm_file="$1"
main_cpp_file="$2"
march="$3"

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_asm_file.s> <main_cpp_file.cpp> <march>"
    exit 1
fi

# ---- Derive output file ----
# Same directory, remove .s, add .elf
out_file="${input_asm_file%.s}.elf"

# ---- Build ----
"$CLANG" \
    --std=c++11 \
    -static \
    -O3 \
    -ffast-math \
    -stdlib=libstdc++ \
    -L/usr/lib64 \
    -L/lib64 \
    -I/usr/include/c++/14 \
    -I/usr/include \
    -I/usr/include/c++/14/x86_64-redhat-linux/ \
    -I"$INCLUDE_DIR" \
    "$input_asm_file" \
    "$main_cpp_file" \
    "$RUNTIME_LIB" \
    -target x86_64-unknown-linux-gnu \
    -march="$march" \
    -o "$out_file"

echo "Build successful: $out_file"
