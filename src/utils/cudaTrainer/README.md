# CUDA Neural Network Trainer

GPU trainer for Caissa's multilayer NNUE: `(32×768 → 1536) × 2 → pairwise CReLU → 16 → 32 → 1`, with 8 output
variants selected by piece count. Built into the `utils` executable when CMake finds the CUDA Toolkit
(`USE_CUDA` define, `CUDA_ARCHITECTURES native`).

## Usage

```bash
bin/utils trainCudaNetwork [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--name <dir>` | `eval` | Output directory and file prefix (`.log`, `.pnn`, `.ckpt`) |
| `--net <file.pnn>` | (from scratch) | Start from an existing packed net |
| `--resume <file.ckpt>` | | Continue an interrupted run (weights, Adam moments, position count) |
| `--restartSchedule` | off | With `--resume`: keep weights and optimizer state, restart the LR schedule |
| `--LR`, `--endLR` | `1e-4`, `2.5e-6` | Cosine learning-rate schedule start and end |
| `--trainingLength <B>` | `150` | Length of the LR/lambda schedule in billions of positions |
| `--startLambda`, `--endLambda` | `0`, `0` | Target blend: 0 = game result, 1 = evaluation |
| `--iterations <n>` | unlimited | Stop after `n` iterations |
| `--seed <n>` | `12345` | Weight initialization seed |
| `--bucketLeak <p>` | `0` | Probability of training a position on a neighbouring output bucket |
| `--freezeFeatureTransformer` | off | Train the output subnets only |
| `--reviveDeadNeurons` | off | Re-seed output subnet neurons that can no longer learn |

Training data is read from `data/trainingData` (`TrainingDataLoader`). One iteration is 2M positions in
32K batches, followed by validation on 256K positions; a packed net and a full checkpoint are written every
10B positions.

## Training

- **Quantization-aware training**: weights and biases are fake-quantized on read with the same scales as the
  packed net (`nn::*QuantizationScale` in `backend/PackedNeuralNetwork.hpp`); gradients flow to the float
  master weights.
- **Input factorizer** (`USE_FACTORIZER`): a shared 768-feature block added to every king bucket during
  training and folded into the bucket weights when the net is packed.
- **AdamW** with decoupled weight decay on weights only: 0.0025 on the feature transformer, none on the
  output subnet. Feature-transformer weights are clipped to ±6, factorizer weights to ±0.99.
- **Overlap**: the main stream runs the passes; an aux stream clears the FT gradient buffer during the forward
  pass, and a copy stream uploads the next batch during the Adam updates (synchronized with events).

## Source Layout

| File | Contents |
|---|---|
| `CudaCommon.hpp` | `CUDA_CHECK`, `CudaBuffer`, `PinnedBuffer`, `CudaStream` (no STL) |
| `CudaKernels.hpp` | Interface between host code and kernels: `CudaBatchData`, `DeviceLayer`, `AdamUpdateParams`, factorizer constants, kernel launcher declarations, device helpers shared by the kernels (no STL) |
| `CudaForward.cu` | Forward kernels: sparse input + pairwise activation, L1, L2/L3, sigmoid |
| `CudaBackward.cu` | Backward kernels: sigmoid derivative, dense weight gradients, backprop to hidden layers, feature transformer and factorizer gradients |
| `CudaOptimizer.cu` | AdamW update kernel |
| `CudaNetwork.hpp/.cpp` | `CudaNeuralNetwork`: owns the weights, streams and events; runs `Forward`/`Backward` by calling the launchers; checkpoint save/load |
| `CudaWeightsStorage.hpp/.cpp` | Device weights and Adam moments of one layer: init, host copies, Adam step bookkeeping |

The `.cu` files include only `CudaKernels.hpp`, so nvcc does not parse the STL; all host-side code that needs
it lives in the `.cpp` files. `CudaNetworkTrainer.cpp` (one directory up) drives training: data loading,
LR/lambda schedules, validation against the packed net, and output files.
