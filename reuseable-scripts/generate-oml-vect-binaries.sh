#!/bin/bash


files=("auto_Opset18" "bert_google"  "bgesmallenv" "deberta" "distilbert" "phonemizerbig" "nsnet2" "roberta")
mkdir -p /workdir/outputs/oml-vect
mkdir -p /workdir/outputs/oml-vect/irs-x86
mkdir -p /workdir/outputs/oml-vect/elf-x86

for file in "${files[@]}"; do
  onnx_file="/workdir/inputs/${file}_t.onnx"
  irs_dir="/workdir/outputs/oml-vect/irs-x86"
  out_dir="/workdir/outputs/oml-vect/elf-x86"

  if [ -f "$onnx_file" ]; then
    echo "Processing file: $file"


    /workdir/oml-vect-docker/t1-cgo-build/Release/bin/onnx-mlir -O3 --vlen=4 --uf1=8 --uf2=4 --uf3=2  --EmitLLVMIR "$onnx_file" -o "${irs_dir}/${file}"
    echo "ONNX-MLIR completed for $file"


    /workdir/llvm-project/build-v20/bin/mlir-translate --mlir-to-llvmir "${irs_dir}/${file}.onnx.mlir" > "${irs_dir}/${file}.ll"
    echo "MLIR to LLVM IR translation completed for $file"


    /workdir/llvm-project/build-v20/bin/opt -O3 -S "${irs_dir}/${file}.ll" -o "${irs_dir}/${file}.opt.ll"
    echo "Optimization completed for $file"


    /workdir/llvm-project/build-v20/bin/llc -O3 --filetype=asm -o "${irs_dir}/${file}.s" "${irs_dir}/${file}.opt.ll"
    echo "LLVM IR to Assembly conversion completed for $file"


    /workdir/llvm-project/build-v20/bin/clang++ \
      --std=c++11 -static -O3 -ffast-math -stdlib=libstdc++ \
      -L/usr/lib64 -L/lib64 \
      -I /usr/include/c++/14 -I /usr/include \
      -I /usr/include/c++/14/x86_64-redhat-linux/ \
      -I /workdir/oml-vect-docker/include/ \
      "${irs_dir}/${file}.s" /workdir/inputs/run.cpp \
      /workdir/oml-vect-docker/runtime/libcruntime.a \
      -target x86_64-unknown-linux-gnu -march=alderlake \
      -o "${out_dir}/${file}"

    echo "Assembly to Executable compilation completed for $file"
  else
    echo "File $onnx_file does not exist. Skipping."
  fi

  echo "Finished processing $file"
  echo "--------------------------------------------"
done