#!/bin/bash

set -e

# ---- Check input ----
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <model_name>"
  echo "Example: $0 bert_google"
  exit 1
fi

file="$1"
inputs="$2"
USER_DIMS="$3"

# ---- Convert * to , ----
DIMS="${USER_DIMS//\*/,}"

# ---- Directories ----
mkdir -p irs-x86
mkdir -p elf-x86

onnx_file="${file}.onnx"
irs_dir="irs-x86"
out_dir="elf-x86"
# ---- Check ONNX file ----
if [ ! -f "$onnx_file" ]; then
  echo "File $onnx_file does not exist. Exiting."
  exit 1
fi

echo "Processing file: $file"

# ---- ONNX-MLIR ----
/workdir/oml-vect-docker/t1-cgo-build/Release/bin/onnx-mlir \
  -O3 --vlen=4 --uf1=8 --uf2=4 --uf3=2 \
  --EmitLLVMIR "$onnx_file" \
  -o "${irs_dir}/${file}"

echo "ONNX-MLIR completed for $file"

# ---- MLIR → LLVM IR ----
/workdir/llvm-project/build-v20/bin/mlir-translate \
  --mlir-to-llvmir "${irs_dir}/${file}.onnx.mlir" \
  > "${irs_dir}/${file}.ll"

echo "MLIR to LLVM IR translation completed for $file"

# ---- LLVM opt ----
/workdir/llvm-project/build-v20/bin/opt \
  -O3 -S "${irs_dir}/${file}.ll" \
  -o "${irs_dir}/${file}.opt.ll"

echo "Optimization completed for $file"

# ---- LLVM IR → ASM ----
/workdir/llvm-project/build-v20/bin/llc \
  -O3 --filetype=asm \
  -o "${irs_dir}/${file}.s" \
  "${irs_dir}/${file}.opt.ll"

echo "LLVM IR to Assembly conversion completed for $file"

# ---- ASM → ELF ----
/workdir/llvm-project/build-v20/bin/clang++ \
  --std=c++11 -static -O3 -ffast-math -stdlib=libstdc++ \
  -L/usr/lib64 -L/lib64 \
  -I/usr/include/c++/14 -I/usr/include \
  -I/usr/include/c++/14/x86_64-redhat-linux/ \
  -I/workdir/oml-vect-docker/include/ \
  "${irs_dir}/${file}.s" \
  /workdir/inputs/run.cpp \
  /workdir/oml-vect-docker/runtime/libcruntime.a \
  -target x86_64-unknown-linux-gnu \
  -march=alderlake \
  -o "${out_dir}/${file}"

echo "Assembly to Executable compilation completed for $file"
echo "Finished processing $file" output saved to "${out_dir}/${file}"
echo "--------------------------------------------"


${out_dir}/${file} $inputs $DIMS -times=1
