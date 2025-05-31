# ONNX-MLIR FashionMNIST Example (Parametrized Pipeline)

This README explains how to run the end-to-end FashionMNIST pipeline using a single JSON configuration and driver script:

1. Export a small MLP model (PyTorch ➔ ONNX), both vanilla and ONNX-Runtime-optimized  
2. Run the optimized ONNX model directly with a RISC-V library call.

All steps are parameterized via **`config_riscv.json`** and orchestrated by **`build_and_bench_riscv.py`**.

---

## Prerequisites

```bash
# Python ≥3.x
python3 --version

# Install required Python packages (latest as of cutoff):
pip install \
  onnx         # Version: 1.26.4  
  numpy        # Version: 1.24.3  
  torch        # Version: 2.2.2  
  torchvision  # Version: 0.17.2  
```

You also need:  
- **onnx-mlir** (v latest as of cutoff) in your `PATH`  
- A C++ compiler and linker (by default `clang++` is used in all programs) 

---

## Configuration

All parameters live in **`config_riscv.json`**:

- Model dimensions, batch size, paths, opsets, iteration counts  
- Compile and runtime paths (onnx-mlir, include dirs)  

Edit `config_riscv.json` to change network width, dataset paths or benchmark settings.

---

## Usage

From the example directory:

```bash
cd src/matmul_relu_matmul_fashion_mnist
python3 build_and_bench_riscv.py --config config_riscv.json
```

The driver will:

  1. Export MLP ⟶ ONNX (vanilla & optimized)   
  2. Compile & benchmark with ONNX-MLIR + custom C++ runtime for RISC-V

On success you’ll see per-step ✅ messages and a final summary:

```
✅ Model compiled successfully. Library 'model.so' is in: compiled_model_onnx_mlir  
✅ Completed 10 warmup iterations.  
✅ Completed 10 benchmarking iterations.  
======= ONNX-MLIR BENCHMARK RESULTS =======  
…
```

Accuracy should be around 80% (depending on `num_iterations` in `config_riscv.json`).

---

## Detailed Steps

### 1. EXPORT MLP → ONNX

File: `onnx_any_ep_only_export.py`  

- Reads weights and parameters from `config_riscv.json`  
- Builds PyTorch MLP, exports `mnist_model_cpu_initial.onnx` (opset 11)  
- Optimizes with ONNX Runtime ⇒ `mnist_model_cpu_optimized.onnx`
- It could be modified along with `config_riscv.json` to export other PyTorch models.

Usage under driver: no direct call needed (step run by `build_and_bench_riscv.py`).

### 2. COMPILE & BENCHMARK WITH ONNX-MLIR

File: `onnx-mlir_test_any_ep_riscv.py`  

- Compiles `mnist_model_cpu_optimized.onnx` via `onnx-mlir --EmitObj`  
- Compiles the `matmu_rvvlib.cpp` with the RISC-V binary and links it with `FusedGemmRuntime_omtensor_riscv.cpp` which
holds the lib call invoked by onnx-mlir.
- Links everything into `model.so`  
- Runs it using `RunONNXModel.py` over a FashionMNIST test set  
- Reports latency, throughput, and accuracy  

### 3. RUN OPTIMIZED ONNX WITH ORT (Optional)

Skip MLIR and run directly:

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
  "mnist_model_cpu_optimized.onnx",
  providers=["CPUExecutionProvider"]
)
x = np.random.rand(1, 784).astype(np.float32)
y = sess.run(None, { sess.get_inputs()[0].name: x })
print("Output shape:", y[0].shape)
```

---

## Summary

Use **`config_riscv.json`** to drive all dimensions, filenames and paths.  
Launch the full pipeline with:  

```bash
python3 build_and_bench_riscv.py --config config_riscv.json
```  