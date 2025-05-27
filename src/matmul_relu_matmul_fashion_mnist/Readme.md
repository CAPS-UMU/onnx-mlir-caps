# ONNX-MLIR FashionMNIST Example

This README explains how to:

1. Export a small MLP model (PyTorch ➔ ONNX), both vanilla and ONNX-Runtime-optimized  
2. Export a standalone fused Gemm + ReLU ONNX node model and embed it in C  
3. Compile and benchmark the optimized model with ONNX-MLIR and a custom C runtime  
4. Run the optimized ONNX model directly with ONNX Runtime

---

## Prerequisites

```bash
# Python ≥3.x
python3 --version

# Install required Python packages (latest as of cutoff):
pip install \
  onnx \
  numpy \
  torch \
  torchvision \
  onnxruntime \
```

You also need:
- **onnx-mlir** (v latest as of cutoff) in your `PATH`
- A C++ compiler (`clang++` / `g++`) and linker
- onnxruntime binaries from https://github.com/microsoft/onnxruntime/releases (onnxruntime-linux-x64-X.xx.x.tgz)
---

# 1. EXPORT MLP → ONNX

File: `src/matmul_relu_matmul_fashion_mnist/onnx_any_ep_only_export.py`

This script loads the pickled MLP weights (`fasionmnist_mlp_params.pkl`), builds a tiny two-layer network in PyTorch, and:

1. Exports a **vanilla ONNX** model (`mnist_model_cpu_initial.onnx`, opset 11).  
2. Creates an **optimized ONNX** with ONNX Runtime’s graph optimizations (`mnist_model_cpu_optimized.onnx`).

Usage:

```bash
cd src/matmul_relu_matmul_fashion_mnist
python3 onnx_any_ep_only_export.py
```

Outputs:

- `mnist_model_cpu_initial.onnx`  
- `mnist_model_cpu_optimized.onnx`

---

Note: You can use `src/matmul_relu_matmul_fashion_mnist/onnx_any_ep_export_bench.py` to also get a benchmark with the PyTorch model to evaluate accuracy with the given weights.

# 2. EXPORT STANDALONE GEMM+ReLU ONNX

File: `src/matmul_relu_matmul_fashion_mnist/export_gemm_relu.py`

This utility emits a minimal ONNX model performing

```text
Y = Gemm(A, B, C, transA=0, transB=1)  
Z = Relu(Y)
```

with HARDCODED shapes:

- A: [1, 784]  
- B: [128, 784]  
- C: [128]  
- Z: [1, 128]

It also runs

```bash
xxd -i gemm_relu.onnx > gemm_relu_model.inc
```

to generate a C include file for embedding in your C++ runtime.

Usage:

```bash
cd src/matmul_relu_matmul_fashion_mnist
python3 export_gemm_relu.py
```

Outputs:

- `gemm_relu.onnx`  
- `gemm_relu_model.inc`

---

# 3. COMPILE & BENCHMARK WITH ONNX-MLIR

File: `FusedGemmRuntime_omtensor_ort.cpp`

`FusedGemmRuntime_omtensor_ort.cpp` includes `gemm_relu_model.inc` and performs the ORT call to run the minimal ONNX model's nodes. To be able to perform ORT calls you need to link against ORT binaries.

File: `src/matmul_relu_matmul_fashion_mnist/onnx-mlir_test_any_ep.py`

This end-to-end script:

1. **Compiles** `mnist_model_cpu_optimized.onnx` with `onnx-mlir --EmitObj`  
2. **Compiles** and links `FusedGemmRuntime_omtensor_ort.cpp` into a shared library.
3. **Links** everything into `model.so`  
4. **Runs** `RunONNXModel.py` over a small FashionMNIST test set  
5. **Reports** latency, throughput, and accuracy

In order to correctly compile everything, it performs a linking of the executable ONNX model and the C implementation in `FusedGemmRuntime_omtensor_ort.cpp` against ORT binaries, by default `/workdir/onnxruntime-linux-x64-gpu-1.21.1`.

Usage:

```bash
# From project root
cd src/matmul_relu_matmul_fashion_mnist
python3 onnx-mlir_test_any_ep.py
```

On success you’ll see:

```
✅ Model compiled successfully. Library 'model.so' is in: compiled_model_onnx_mlir
✅ Completed 1 warmup iterations.
✅ Completed 10 benchmarking iterations.
======= ONNX-MLIR BENCHMARK RESULTS =======
...
```

Accuracy should be arround 80% depending on NUM_ITERATIONS.

---

# 4. RUN OPTIMIZED ONNX WITH ORT 

You can also skip ONNX-MLIR and invoke the optimized ONNX directly via ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np

# Load optimized ONNX model
sess = ort.InferenceSession(
    "src/matmul_relu_matmul_fashion_mnist/mnist_model_cpu_optimized.onnx",
    providers=["CPUExecutionProvider"]
)

# Prepare dummy input
x = np.random.rand(1, 784).astype(np.float32)
input_name = sess.get_inputs()[0].name

# Run inference
y = sess.run(None, { input_name: x })  # y[0].shape == (1, 10)
print("Output shape:", y[0].shape)
```

---

## Summary

- **`onnx_any_ep_only_export.py`** → PyTorch ➔ ONNX (vanilla & optimized)  
- **`export_gemm_relu.py`** → stand-alone Gemm+ReLU ONNX + C-include  
- **`onnx-mlir_test_any_ep.py`** → compile via ONNX-MLIR + custom C runtime + benchmark  
- **Direct ORT call** → run `.onnx` with ONNX Runtime