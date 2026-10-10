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
| `--name <dir>` | `eval` | Output directory and file prefix, see [Output Files](#output-files) |
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
| `--decisiveSkip` | off | Skip positions whose decisive score agrees with the game result, with probability `clamp((\|score\| − 400) / 800, 0, 0.75)` |
| `--pieceCountTarget` | off | Replace the fixed piece-count skip curve with a target distribution per pair of piece counts |
| `--hmcSkipFrom20` | off | Skip by half-move counter only above 20 (`clamp((hmc − 20) / 80, 0, 1)`) instead of `sqrt(hmc / 100)` |

Training data is read from `data/trainingData` (`TrainingDataLoader`). One iteration is 2M positions in
32K batches, followed by validation on 256K positions; a packed net and a full checkpoint are written every
10B positions.

## Output Files

Everything goes to `<name>/`, file names prefixed with `<name>`. Tables are tab-separated with one header row;
FENs contain spaces but no tabs. A run continued with `--resume` appends to the same tables (no second
header). `positions` is the training-position count of the schedule and keeps counting across resumes, so it is
the key for joining files and runs; `iteration` and `elapsed_s` restart from 0 in every process.

| File | Contents |
|---|---|
| `-settings.txt` | `key=value` run settings: full command line, start time, build, device, all options including the loader flags, data path/files/bytes, and with `--pieceCountTarget` the keep probabilities (one per pair of piece counts, from 0–1 pieces). A resumed run writes `-settings-resume-<N>B.txt` instead |
| `-progress.tsv` | One row per iteration from the third on (columns below) |
| `-weights.tsv` | Every 1B positions: `positions, layer (FT/L1/L2/L3), w_min, w_max, w_avg, w_std, b_min, b_max, b_avg, b_std` |
| `-test-positions.tsv` | Every 1B positions: `positions, index, eval, fen` — packed-net eval of the fixed test positions in internal units; `index` is stable (one FEN is listed twice) |
| `-<N>B.pnn`, `-<N>B.ckpt` | Packed net and full training state every 10B positions |
| `.pnn` | Latest packed net, rewritten every 50 iterations |

`-progress.tsv` columns:

| Column | Meaning |
|---|---|
| `iteration`, `positions`, `elapsed_s` | Iteration in this process, positions after it, seconds since training started in this process |
| `learning_rate`, `lambda` | Schedule values used by the iteration |
| `train_rmse` | RMSE of this iteration's training batches against their targets (game result at lambda 0) |
| `val_rmse` | Packed net on the 256K validation positions; it runs concurrently with training, so it scores the weights after the previous iteration. Validation targets use lambda 1, decayed towards the game result by move count |
| `val_float_subset_rmse`, `val_pnn_subset_rmse` | Float and packed net on the first 4K validation positions; the gap is the quantization error |
| `train_ms`, `gpu_ms`, `loader_ms`, `validation_ms` | Wall time of the training task (GPU work plus host copies), GPU time alone, training-set generation and validation. The three tasks run concurrently, so the largest one limits the speed |

The console prints a three-line report every 10 seconds (progress and ETA; learning rate, lambda and
validation; positions per second of each task and which one limits the speed) and, at every checkpoint, the
weight statistics and test position evals.

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
