# Shakespeare speedrun notes

Goal: get the character-level Shakespeare model to produce convincing text in as little training time as
possible, using the AutoGrad framework, without cheating (train only on the 90% training split, report loss on the
held-out 10%, no pretrained weights, no external data, no other ML libraries).

Everything here is reproducible with the `Speedrun` project (`Speedrun.exe <mode> --key value ...`):

| mode        | what it does |
|-------------|--------------|
| `gradcheck` | compares every parameter's backprop gradient with a central finite difference (build with `CL=-DAUTOGRAD_DOUBLE` for a meaningful check) |
| `causal`    | changes the last token of a sequence and checks no earlier prediction changes |
| `gemmtest`  | checks the matrix multiply against a reference and measures GFLOP/s |
| `profile`   | runs a few training steps and prints time per kernel |
| `train`     | time-budgeted training with periodic validation, CSV log and a final sample |

## Bugs found (and why they matter)

### 1. LayerNorm normalised over time, letting the model see the future (Transformer\LayerNorm.h)
`x->Mean(1,true)` / `x->Var(1,...)` on a `(B,T,C)` tensor averages over **dimension 1, which is time**. In
PyTorch, `nn.LayerNorm(n_embd)` normalises over the *last* dimension: each token's own feature vector. Normalising
over time is a different (and odd) operation, and in a decoder it is also a leak: the mean and variance at
position *t* include positions *t+1...T*, so every prediction gets information from the characters it is supposed
to predict. `Speedrun causal` showed earlier logits changing by up to 0.78 when only the last character changed.
A leak like this makes the training/validation loss look better than the model really is, and generation (which
has no future to peek at) is worse than the loss suggests.

Fix: normalise over `Shape().size()-1`. (Note `Mean(-1,...)` can't be used: `KoMean` treats -1 as "mean of all
elements".)

### 2. Adam's time step advanced once per parameter, not once per step (AutoGrad\ADAM.h)
`++_t` was inside the `for(auto& p:_parameters)` loop. Adam's bias correction divides the moment estimates by
`1-beta^t`, which matters for the first few hundred steps. With ~40 parameter tensors, `t` raced ahead ~40x
faster, so the early-training correction was mostly lost. It's also slightly different per tensor (the first tensor
saw t=1, the second t=2, ...). It's not catastrophic (Adam still works), but it isn't Adam as published.

### 3. Shared gradient buffers could be modified in place (AutoGrad\Tensor.h)
`KoAdd::Backward` returns the *same* NDArray to both inputs when no broadcasting happened (ReverseBroadcast returns
its input unchanged), and `Tensor::Backward` accumulates further gradients with `_gradient->_data += ...`. If the
tensor that received the shared buffer later received a second contribution, the `+=` would also change the other
input's gradient. The depth-first backward order happens to make this safe for the tutorial model (the gradient
check passes), but `x + parameter` with equal shapes would have produced wrong parameter gradients. Fix: parameters
take a private copy of their first gradient; intermediate tensors sum out of place.

### 4. Parameters' child list grew forever (AutoGrad\Tensor.h / ADAM.h)
Every forward pass registers the new tensor as a "child" of each parameter. Backward decrements the counts, but
nothing removed entries, and inference passes (evaluation, generation) never backpropagate at all, so the map grew
with every forward pass for the whole run. Optimisers now call `ClearChildren()` after updating a parameter.

### 5. The thread pool could never shut down (AutoGrad\NDThreadPool.h)
`~NDThreadPool` woke the workers but never set `_stop = true`, so each worker re-checked its wait predicate
(`!_workQueue.empty()||_stop`), went back to sleep, and `join()` waited forever: the program could not exit. `_stop`
is now set (and atomic) before waking.

### 6. Minor: operator precedence in `NDThreadPool::_ForEach`
`(end-begin)+subRange-1/subRange` evaluates `1/subRange` first (integer 0 for subRange>1), so up to 32 tasks are
created even when the range is smaller. The extra tasks are empty, so the result is correct, just wasteful.

### Not a bug: the autograd itself is right
In double precision, every parameter gradient of the tutorial model matches finite differences to ~1e-9 relative
error. The float32 check looks alarming (errors of 10-40%) only because a central difference with eps=1e-2 on a
loss of ~5.7 is below float32 resolution for small gradients. Use doubles for gradient checks.

### Pre-existing: AutoGrad's own test executable crashes
`AutoGrad.exe` segfaults on both `master` and this branch (identical output up to the crash). The cause is a stale
test, not the engine: the first `Test_Tensor` case (`Test_Add`, scalar+scalar) does `z = x->Add(y); z->Backward();
print(z->Gradient())`, but `Tensor::Backward` deliberately releases an intermediate tensor's gradient once it has
been passed on to its creators (`_gradient.reset()`, a memory saving), so `z->Gradient()` is null. The tests
predate that optimisation and expect intermediate gradients to survive. Left as is (it's your call whether to keep
the memory saving or the tests' expectation).

## Why it was slow

Profiling the tutorial model (128 wide, 4 layers, batch 32x128): ~1.8 s per step for ~0.02 TFLOP of work.

* Every matmul went to `GpuMatMul`, which `cudaMalloc`s, creates a stream, copies to the GPU, runs a naive kernel,
  copies back and frees, for every single multiply, including one per batch item per head in attention.
* Element-wise ops are single threaded and each allocates a new array; LayerNorm alone is ~10 graph nodes.
* The batched `Dot` (attention, and the weight-gradient of `x@W`) runs one small matmul per batch element.
* The thread pool takes ~1 ms to dispatch work (condition variables + per-worker queues), which dominates small ops.

Fixes, all in AutoGrad:
* `Gemm.h`: packed AVX2/FMA GEMM (6x16 micro-kernel) with transpose flags; ~1.1 TFLOP/s on large matrices.
* `ParallelFor.h`: spin-then-park thread pool; a job completes when its work is done, not when every worker has
  checked in (otherwise one descheduled worker stalls everything).
* `KoFused.h`: single-node kernels with analytic backward for Linear, LayerNorm, GELU/ReLU/ReLU², residual add,
  causal multi-head attention (optionally with RoPE), softmax cross-entropy and a whole-sequence LSTM.
  Each one is gradient-checked.

Same-size model: ~1.8 s -> ~0.1 s per step.
