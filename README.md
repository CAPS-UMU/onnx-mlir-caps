<!--- SPDX-License-Identifier: Apache-2.0 -->
<p align="center"><img width="50%" src="docs/logo/onnx-mlir-1280x640.png" /></p>

# [ONNX-MLIR](https://github.com/CAPS-UMU/onnx-mlir-caps/tree/main?tab=readme-ov-file#onnx-mlir)

# [CGO 2026: Artifact Abstract](https://github.com/CAPS-UMU/onnx-mlir-caps/tree/main?tab=readme-ov-file#artifact-abstract-cgo-2026)

# Artifact Abstract CGO 2026
This is the supporting artifact for the paper titled "Enabling Automatic Compiler-Driven Vectorization of Transformers" as published in CGO 2026. It includes the source code for the proposed tool (oml-vect), along with scripts to compile and reproduce all experimental results presented in the paper. We provide a Docker image containing all necessary dependencies preinstalled, enabling straightforward setup and execution of the experiments. The setup supports running the experiments on an x86 host, and it also allows cross-compilation for RISC-V ISA, and it automatically transfers the statically linked RISC-V binaries to the host to be then copied to the RISC-V target via scp. The reported results were obtained on Intel Xeon an x86\_64 Intel(R) Xeon(R) E5-2630 v4 CPU @ 2.20GHz and a Xilinx U55C FPGA emulating an Atrevido 423 RISC-V 64-bit core with a 512-bit vector unit.

# Artifact check-list (meta-information)



1.  **Algorithm:** Data-layout optimization and reduction identification to enable auto-vectorization in MLIR
2.  **Program:** ONNX graph representations of transformer and neural network models
3. **Compilation:** Publicly available and included in this artifact: LLVM v20.0, MLIR v20.0, and ORT v1.23.2
4. **Transformations:** onnx-mlir toolchain to obtain vectorized affine dialect code for each kernel and LLVM to statically compile binaries for x86 and RISC-V, included in this artifact
- **Binary:** Linux executables for x86 and RISC-V included in this artifact and scripts to generate these binaries automatically.
- **Data set:** Included in this artifact, testing datasets used for all models are detailed in the corresponding references
- **Run-time environment:**
  - x86 - The Docker container (see x86 Experiment workflow section)
  - RISC-V - The Docker container to cross-compile binaries and RISC-V CPU with Linux OS to run the binaries (see RISC-V Experiment workflow section)
- **Hardware:** Intel Xeon and RISC-V CPU with vector extension
- **Execution:** We recommend running the experiments in an isolated environment, as the results may vary if other processes are active.
- **Metrics:** Execution time and number of reductions identified
- **Output:** CSV files containing the normalized execution times, performance improvements, and console logs reporting the performance improvements.
- **Experiments:** Docker image included in this artifact contains scripts to regenerate the results. Results may vary with respect to hardware parameters such as vector width, L1, L2 cache size, as the unroll-and jam-factor is hardware dependent.
- **How much disk space required (approximately)?:** 80GB
- **How much time is needed to complete experiments (approximately)?:** Dependent on the CPU and its operating frequency. Completing the full benchmark suite requires 15 - 20 mins on an Intel 13th Gen Intel(R) Core(TM) i7-13700K, 5.4 GHz with 64 GiB RAM
- **Publicly available?:** Yes, see GitHub and AE Zenodo



## Description

### How delivered

This artifact is available at Docker Hub

### Hardware dependencies

A machine with vector units is required. We evaluated on an Intel Xeon and Atrevido (RISC-V Core). OML-vect auto-vectorizes the code and improves performance. Performance improvement may vary with the hardware, therefore, we recommend similar platforms for reproducing the results.

### Software dependencies

Working Docker installation. This has been tested on a Linux x86 host machine with Docker v28.1.1. For RISC-V, we recommend Bianbu Star 2.1.7 or LINUX OS.

To test the state-of-the-art tool ORT on the RISC-V platform, we used proprietary software Semidynamics ONNX-Runtime. The open source alternatives, such as the Python package (pip install OnnxRuntime), are not supported on RISC-V. Hence, while we used ORT as a point of comparison to our proposal, it cannot be installed on RISC-V without manual porting. This underlines the need for an automatic neural networks compiler for RISC-V, such as oml-vect.

## Installation
Download the docker image repository using the command:

```bash
docker pull shreyasubhash/omlvect:r1
```

Verify availability of the image:

```bash
docker images -a | grep "omlvect"
```


## Experiment workflow

### x86

To compile, run, and verify the results discussed in this paper, on host machine (x86 Intel), **navigate to the directory where the output should be stored** and run the command:

```bash
docker run -ti --volume ${PWD}:/workdir/shared \
  shreyasubhash/omlvect:r1 \
  bash -c "/workdir/scripts/infer.sh"
```

This command will compile the benchmarks, execute them (for x86), process the execution time and number of reductions identified for each kernel, and generate CSV files. The resulting CSV files can be accessed on the host machine.

### RISC-V

To cross-compile and generate binaries for RISC-V, run the command:

```bash
docker run -ti --volume \
    ${PWD}:/workdir/shared  \
    shreyasubhash/omlvect:r1 \
    bash -c "/workdir/scripts/compile-riscv.sh"
```

This command will cross-compile all the benchmarks, and the statically linked ELFs will be available on the host machine `${PWD}/elf-rvv` along with `infer-rvv.sh` script to run the files on the RISC-V CPU and generate the corresponding CSV file.

To obtain speedup measurements for the RISC-V platform, first copy the `runtime.csv` file generated on the RISC-V CPU into the `${PWD}` directory on the host system. Then execute the following command to process the results:

```bash
docker run -ti --volume \
 ${PWD}:/workdir/shared \
 shreyasubhash/omlvect:r1 \
 bash -c "/workdir/scripts/get-rvv-speed-up.sh \
 /workdir/shared/<your-file>.csv"
```

| Filename | Information |
|----------|-------------|
| Runtime.csv | Raw runtime |
| normalised_runtime.csv | Normalised runtime compared to baseline (`onnx-mlir-no-cust-opts`) |
| performance.csv | Percentage improvement in performance over the baseline |
| reduction_counts.csv | Total number of reduction identified by MLIR and OML-vect |

Details of all generated CSV files are shown in the table above for both x86 and RISC-V experiments.



## Evaluation and expected results

We expect OML-vect will autovectorize the code and reduce the execution time, but the performance improvements may vary across different hardware platforms, since the vector width and unroll-and-jam factor are hardware-dependent.

## Reusability
The following instructions summarize how to reproduce the OML-vect workflow. See the README file in the [GitHub repository](https://github.com/onnx/onnx-mlir) and the Docker image `shreyasubhash/omlvect:reusable`

### OML-vect toolchain workflow

Here we describe the full OML-vect workflow—from modifying the toolchain to generating and executing the final ELF.

#### Code changes for OML-vect

The following subsection details the complete OML-vect workflow, including the required toolchain updates, compilation steps, and execution of the resulting ELF.

- Incorporate the revised MatMul algorithm in the file `src/Conversion/ONNXToKrnl/Math/MatMul.cpp`
- Implement the reduction pass and invokes the MLIR vectorization pipeline in file `src/Conversion/KrnlToAffine/ConvertKrnlToAffine.cpp`

Additional Scripts and other file modifications:

- Preprocessing steps for data layout transformation are described in a separate section.
- A CMake file enables cross-compilation when ONNX-MLIR runtime libraries are not available for the target hardware.

Follow the build instructions in the [onnx-mlir repository](https://github.com/onnx/onnx-mlir) with modifications to files as mentioned above to compile **oml-vect**

#### Data Layout Modification

Run:

```bash
data-layout.sh <input onnx graph>
```

This command applies the data layout transformation to generate transposed MatMul's operand B.

#### Get vectorized MLIR code

Run:

```bash
<oml-vect-build-path>/bin/onnx-mlir -O3 \
--vlen=<vector length> \
  --uf1=<unroll factor k> \
  --uf2=<unroll factor m> \
  --uf3=<unroll factor n> \
  --EmitLLVMIR <input.onnx>
```

This command integrates the updated MatMul algorithm, runs the reduction-mapping pass, and then invokes MLIR super-affine vectorization using the resulting reduction map to generate vectorized LLVM Dialect.

#### Get LLVM IR

Run:

```bash
/workdir/llvm-project/build-v20/bin/mlir-translate \
  --mlir-to-llvmir <input.mlir> > <output.ll>
```

This command converts vectorized MLIR to LLVM IR.

#### Optimize LLVM IR

Run:

```bash
/workdir/llvm-project/build-v20/bin/opt -O3 -S \
    <input.ll> -o <output.opt.ll>
```

This command applies LLVM IR O3 level optimizations

#### Get asm code

Run:

```bash
/workdir/llvm-project/build-v20/bin/llc -O3 \
  -mcpu=<MCPU> --filetype=asm <input.opt.ll> \
  -o <output.s>
```

This command generates architecture-specific assembly.

#### Compile ONNX-MLIR runtime libs

Set the property `CMAKE_C_COMPILER` in `oml-vect/runtime` to desired compiler for target Hardware and run:

```bash
cd oml-vect/runtime && cmake .
```

This command cross-compiles the runtime support libraries for the given target hardware required by ONNX-MLIR and OML-vect generated kernels.

#### Get ELF for target hardware

To generate ELF by linking assembly, runtime libraries, and main.cpp and other dependencies into the final ELF executable run:

```bash
./get-elf.sh <main.cpp> \
    -march=<march> <asm_file.s>
```

#### Run the ELF

Run:

```bash
./<output_elf> <input_tensor_as_numpy_array.c> <dim1*dim2*dim3*...>
```

### Portability Across Hardware
Porting OML-vect to any new hardware requires no modifications to our methodology. One only needs to re-run Steps 6–9 with the target architecture’s -mcpu/-march values. Because OML-vect relies exclusively on LLVM, all LLVM-supported hardware targets are inherently supported.

### Running New Benchmarks

To add new benchmarks, it is sufficient to:

- Re-execute the data-layout preprocessing to construct a new ONNX graph with the transposed B input
- Run the provided script `run-infer.sh` using the new input tensors to run the inference using the command:

```bash
run-infer.sh <ELF name> \
<input_tensor_as_numpy_array.c> \
<dim1*dim2*dim3*...>
```

## Notes

Depending on the host machine configuration, Docker commands might require elevated privileges (`sudo`). Shell scripts inside `${PWD}/shared/elf-rvv` folder may need execute permissions, run `(sudo) chmod +x filename`



# ONNX-MLIR

This project (https://onnx.ai/onnx-mlir/) provides compiler technology to transform a valid Open Neural Network Exchange (ONNX) graph into code that implements the graph with minimum runtime support.
It implements the [ONNX standard](https://github.com/onnx/onnx#readme) and is based on the underlying [LLVM/MLIR](https://mlir.llvm.org) compiler technology.

| System        | Build Status | Model Zoo Status |
|---------------|--------------|------------------|
| s390x-Linux   | [![Build Status](https://www.onnxmlir.xyz/jenkins/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&subject=Jenkins%20CI)](https://www.onnxmlir.xyz/jenkins/job/ONNX-MLIR-Pipeline-Docker-Build/) | [![Model Zoo Status](https://www.onnxmlir.xyz/jenkins/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&config=modelzoo)](https://www.onnxmlir.xyz/jenkins/job/ONNX-MLIR-Pipeline-Docker-Build/Model_20Zoo_20Report/) |
| ppc64le-Linux | [![Build Status](https://www.onnxmlir.xyz/jenkinp/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&subject=Jenkins%20CI)](https://www.onnxmlir.xyz/jenkinp/job/ONNX-MLIR-Pipeline-Docker-Build/) | [![Model Zoo Status](https://www.onnxmlir.xyz/jenkinp/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&config=modelzoo)](https://www.onnxmlir.xyz/jenkinp/job/ONNX-MLIR-Pipeline-Docker-Build/Model_20Zoo_20Report/) |
| amd64-Linux   | [![Build Status](https://www.onnxmlir.xyz/jenkinx/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&subject=Jenkins%20CI)](https://www.onnxmlir.xyz/jenkinx/job/ONNX-MLIR-Pipeline-Docker-Build/) | [![Model Zoo Status](https://www.onnxmlir.xyz/jenkinx/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&config=modelzoo)](https://www.onnxmlir.xyz/jenkinx/job/ONNX-MLIR-Pipeline-Docker-Build/Model_20Zoo_20Report/) |
| amd64-Windows | [![Build Status](https://dev.azure.com/onnx-pipelines/onnx/_apis/build/status/MLIR-Windows-CI?branchName=main)](https://dev.azure.com/onnx-pipelines/onnx/_build/latest?definitionId=9&branchName=main) | |
| amd64-macOS   | [![Build Status](https://github.com/onnx/onnx-mlir/actions/workflows/macos-amd64-build.yml/badge.svg)](https://github.com/onnx/onnx-mlir/actions/workflows/macos-amd64-build.yml) |
| | [![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5549/badge)](https://bestpractices.coreinfrastructure.org/projects/5549) |

This project contributes:
* an ONNX Dialect that can be integrated in other projects,
* a compiler interfaces that lower ONNX graphs into MLIR files/LLVM bytecodes/C & Java libraries,
* an `onnx-mlir` driver to perform these lowering,
* and a python/C/C++/Java runtime environment.

Current levels of support for the code generation of ONNX operations are listed here for
[a generic CPU](docs/SupportedONNXOps-cpu.md) and
[IBM's Telum integrated AI accelerator](docs/SupportedONNXOps-NNPA.md).

## Interacting with the community.

For ongoing discussions, we use an [`#onnx-mlir-discussion`](https://lfaifoundation.slack.com/archives/C01J4NAL4A2) slack channel established under the Linux Foundation AI and Data Workspace.
Join this workspace using this [link](https://join.slack.com/t/lfaifoundation/shared_invite/zt-o65errpw-gMTbwNr7FnNbVXNVFkmyNA).

We use GitHub Issues for request for comments, questions, or bug reports.
Security-related issues are reported using the channels listed in the [SECURITY](SECURITY.md) page.

We hold informal weekly meetings on Tuesdays where we discuss  current issues and progress. Meeting agenda, notes, and links (to participate) are found [here](https://github.com/onnx/onnx-mlir/wiki/Informal-meeting-agenda-and-notes). Please email alexe@us.ibm.com to request a 15-30 min time slot to discuss a specific topic of interest.

## Setting up ONNX-MLIR using Prebuilt Containers

The preferred approach to using and developing ONNX-MLIR is to use Docker Images and Containers, as getting the proper code dependences may be tricky on some systems. Our instructions on using ONNX-MLIR with Dockers are [here](docs/Docker.md).

If you intend to develop code, you should look at our [workflow](docs/Workflow.md) document which help you setup your Docker environment in a way that let you contribute code easily.

## Setting up ONNX-MLIR directly

ONNX-MLIR runs natively on Linux, OSX, and Windows.
Detailed instructions are provided below.

### Prerequisites

<!-- Keep list below in sync with docs/Prerequisite.md. -->
```
python >= 3.8
gcc >= 6.4
protobuf >= 4.21.12
cmake >= 3.13.4
make >= 4.2.1 or ninja >= 1.10.2
java >= 1.11 (optional)
```

All the `PyPi` package dependencies and their appropriate versions are captured in [requirements.txt](requirements.txt).

Look [here](docs/Prerequisite.md) for help to set up the prerequisite software.

At any point in time, ONNX-MLIR depends on a specific commit of the LLVM project that has been shown to work with the project.
Periodically the maintainers need to move to a more recent LLVM level.
Among other things, this requires to update the LLVM commit string in [clone-mlir.sh](utils/clone-mlir.sh).
When updating ONNX-MLIR, it is good practice to check that the commit string of the MLIR/LLVM is the same as the one listed in that file. See instructions [here](docs/BuildONNX.md) when third-party ONNX also need to be updated.

### Build

Directions to install MLIR and ONNX-MLIR are dependent on your OS.
* [Linux or OSX](docs/BuildOnLinuxOSX.md).
* [Windows](docs/BuildOnWindows.md).

After installation, an `onnx-mlir` executable should appear in the `build/Debug/bin` or `build/Release/bin` directory.

If you have difficulties building, rebuilding, or testing `onnx-mlir`, check this [page](docs/TestingHighLevel.md) for helpful hints.


## Using ONNX-MLIR

The usage of `onnx-mlir` is as such:

```
OVERVIEW: ONNX-MLIR modular optimizer driver

USAGE: onnx-mlir [options] <input file>

OPTIONS:

Generic Options:

  --help        - Display available options (--help-hidden for more)
  --help-list   - Display list of available options (--help-list-hidden for more)
  --version     - Display the version of this program

ONNX-MLIR Options:
These are frontend options.

  Choose target to emit:
      --EmitONNXBasic - Ingest ONNX and emit the basic ONNX operations without inferred shapes.
      --EmitONNXIR    - Ingest ONNX and emit corresponding ONNX dialect.
      --EmitMLIR      - Lower the input to MLIR built-in transformation dialect.
      --EmitLLVMIR    - Lower the input to LLVM IR (LLVM MLIR dialect).
      --EmitObj       - Compile the input to an object file.
      --EmitLib       - Compile and link the input into a shared library (default).
      --EmitJNI       - Compile the input to a jar file.

  Optimization levels:
      --O0           - Optimization level 0 (default).
      --O1           - Optimization level 1.
      --O2           - Optimization level 2.
      --O3           - Optimization level 3.
```

The full list of options is given by the `-help` option.
The `-` and the `--` prefix for flags can be used interchangeably.
Note that just as most compilers, the default optimization level is `-O0`.
We recommend using `-O3` for most applications.

Options are also read from the `ONNX_MLIR_FLAGS` environment variable. For example, `ONNX_MLIR_FLAGS="-O3"` will ensure `-O3` for all compilations.

### Simple Example

For example, use the following command to lower an ONNX model (e.g., add.onnx) to ONNX dialect:
```shell
./onnx-mlir --EmitONNXIR add.onnx
```
The output should look like:
```mlir
module {
  func.func @main_graph(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
    return %0 : tensor<10x10x10xf32>
  }
}
```

An example based on the add operation is found [here](docs/doc_example), which build an ONNX model using a python script, and then provide a main program to load the model's value, compute, and print the models output.

### Writing a driver to perform inferences: end to end example

An end to end example is provided [here](docs/mnist_example/README.md), which train, compile, and execute a simple MNIST example using our
C/C++, Python, or Java interface.

## Documentation

Documentation is provided in the `docs` sub-directory; the [DocumentList](docs/DocumentList.md) page provides an organized list of documents. Information is also provided on our public facing
[onnx.ai/onnx-mlir](https://onnx.ai/onnx-mlir/) pages.

## Contributing

We are welcoming contributions from the community.
Please consult the [CONTRIBUTING](CONTRIBUTING.md) page for help on how to proceed.

ONNX-MLIR requires committers to sign their code using the [Developer Certificate of Origin (DCO)](https://developercertificate.org).
Practically, each `git commit` needs to be signed, see [here](docs/Workflow.md#step-7-commit--push) for specific instructions.

## Code of Conduct

The ONNX-MLIR code of conduct is described at https://onnx.ai/codeofconduct.html.

## Adopters
<!-- Please open a PR to add your company/product here. -->

* IBM [zDLC compiler](https://github.com/IBM/zDLC) uses onnx-mlir technology to transform ONNX models into executable binary for [IBM Telum](https://www.ibm.com/z/telum) servers.

## Projects related/using onnx-mlir

* The [onnx-mlir-serving](https://github.com/IBM/onnx-mlir-serving) project implements a GRPC server written with C++ to serve onnx-mlir compiled models. Benefiting from C++ implementation, ONNX Serving has very low latency overhead and high throughput.
