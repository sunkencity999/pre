# Plan: Cross-Layer Async Pread Pipeline

## Status
- Async pread mechanism implemented and working (async_pread_start/wait in infer.m)
- Within-layer overlap tested: no improvement (only 0.1ms overlap window)
- Need: CROSS-LAYER overlap for ~2ms of pread hiding

## Current Per-Layer Sequence (4.5ms total)
```
[deferred_wait] → [CMD1 submit+wait] → [CPU attn] → [CMD2 submit+wait] → [routing] → [SYNC pread] → [CMD3 submit]
     0.87ms           0.5ms              0.27ms          0.45ms           0.003ms      2.43ms          0.03ms
```

## Target Sequence
```
Layer N:  ... → [routing] → [START async pread into BUF_A] → [CMD3 submit (using BUF_B from prev)]
Layer N+1: [deferred_wait] → [CMD1] → [CPU attn] → [CMD2] → [routing] → [WAIT async pread BUF_A] → [CMD3 submit (using BUF_A)]
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              ~2.1ms of compute overlapping with N's pread
```

Pread for layer N runs during layer N+1's compute. By the time N+1 needs expert data, N's pread has had 2.1ms of head start.

## Implementation Steps

### 1. Double-Buffer Expert Data
Already have: `buf_multi_expert_data[MAX_K]` (set A) and `buf_multi_expert_data_B[MAX_K]` (set B).

Add a flip flag:
```c
static int g_expert_buf_flip = 0;  // 0 = use set A for current, 1 = use set B
```

Each layer alternates which buffer set it writes pread data INTO vs which set CMD3 reads FROM.

### 2. Restructure fused_layer_forward

**At layer start (after deferred_wait):**
- If async pread is in flight from previous layer: DON'T wait yet
- Continue with CMD1, CPU attn, CMD2, routing

**After routing:**
- NOW wait for the previous layer's async pread (it's had ~2ms to complete)
- Start THIS layer's async pread into the OTHER buffer set
- Encode CMD3 using the COMPLETED buffer set (from previous layer's pread)

Wait — this doesn't work because CMD3 needs THIS layer's expert data, not the previous layer's. Let me rethink.

### Correct Design

The pread for layer N needs to complete before layer N's CMD3 encodes. But we want pread for N to overlap with layer N's CMD1+attn+CMD2 (which don't use expert data).

**Revised flow:**
```
Layer N start:
  1. Wait for N-1's deferred CMD3 (or GPU combine)
  2. Submit CMD1 (attention projections)
  3. [BACKGROUND: start async pread for layer N's experts]
     - But we don't know N's experts yet! Routing hasn't happened.
```

This is the fundamental problem: we can't start pread until after routing, but routing is the LAST thing before pread in the pipeline.

### The Real Solution: Decouple Routing from Expert Loading

Split CMD2 into two parts:
- CMD2a: o_proj + residual + norm (produces h_post for routing)
- CMD2b: routing gate_proj (produces gate_scores)

Then:
```
Layer N:
  CMD1 → CPU attn → CMD2a+CMD2b → wait → routing topK → [START async pread] → CMD3 (deferred)
Layer N+1:
  [async pread from N still running]
  deferred_wait → CMD1 → CPU attn → CMD2a+CMD2b → wait → routing topK
  [NOW wait for N's async pread — it's had the entire N+1 compute time]
  → CMD3 using N+1's expert data that we NOW start loading synchronously
```

Hmm, this still doesn't help because we need N+1's experts, not N's.

### Actually Correct Solution: Pipeline Expert Data One Layer Ahead

The insight: at the end of layer N, we have N's expert data loaded. We submit CMD3 (deferred) which uses that data. CMD3 runs on GPU while we start layer N+1.

If we started loading N+1's experts AT THE SAME TIME as submitting N's CMD3:
```
Layer N end:  [submit CMD3_N using BUF_A] + [start async pread for N+1 into BUF_B]
Layer N+1:    [deferred_wait N] → [CMD1] → [attn] → [CMD2] → [routing]
              [async pread N+1 completes during this time]
              → [check: do loaded experts match routing? if yes, use BUF_B; if no, sync pread]
```

But we DON'T KNOW layer N+1's experts at the end of layer N. We'd need to PREDICT them.

Previous prediction attempts failed (53% accuracy, overhead > benefit).

### ALTERNATIVE: Overlap pread with CMD3 GPU execution

Currently CMD3 is deferred — GPU runs it while we start the next layer. But we DON'T start loading the next layer's experts during CMD3. What if we did?

After CMD3 submit for layer N:
```
[submit CMD3_N] → [start next layer's CMD1+attn+CMD2] → [routing N+1] → [pread N+1]
                  ↑ CMD3_N runs on GPU here, overlapping with N+1's compute
```

The pread for N+1 currently starts AFTER routing for N+1, which is after CMD2 for N+1, which is after deferred_wait for CMD3_N. So the pread can't start until CMD3_N is done.

But with GPU combine+norm in CMD3, we eliminated the deferred_wait. CMD1 for N+1 submits immediately after CMD3_N. The GPU executes CMD3_N → CMD1_N+1 back-to-back. The CPU is free during this time to do... nothing useful, because it's waiting for CMD1_N+1 to complete.

### THE REAL REAL SOLUTION: Start pread during CMD1 wait

CMD1_wait takes 0.87ms (includes CMD3_prev + CMD1 GPU time). During that 0.87ms, the CPU is IDLE waiting for GPU. What if the CPU started the pread during that wait?

But we don't have the routing results yet — routing happens after CMD2.

UNLESS we use the PREVIOUS TOKEN's routing for the same layer as a prediction. This is temporal locality — 20-35% overlap between tokens at the same layer.

We already have the prediction infrastructure (`g_prefetch_experts`). The issue was that F_RDADVISE predictions wasted SSD bandwidth. But what about LOADING into actual buffers?

```
Layer N, token T:
  [CMD1 submit] → [while waiting: pread PREDICTED experts into BUF_B based on token T-1]
  → [CMD1 wait returns] → [CPU attn] → [CMD2] → [routing]
  → [check predictions: how many of K=4 match?]
  → [pread only the MISSES into BUF_A (typically 2-3 instead of 4)]
  → [CMD3 using mix of BUF_A (misses) and BUF_B (hits)]
```

With 30% hit rate: 1.2 of 4 experts are pre-loaded. Saves ~30% of pread time.
With 50% hit rate: 2 of 4 pre-loaded. Saves ~50%.

The difference from before: we're not polluting any CACHE. We're loading into scratch buffers. Predictions that miss are just overwritten. No cache eviction.

The cost: gate_proj matvec (~0.1ms) + predicted pread during CMD1_wait (runs during idle CPU time, so ~0ms additional). Net cost is ~0.1ms per layer.

## Key Files
- `infer.m`: `fused_layer_forward()` around line 4900-5230
- `async_pread_start/wait` at line ~3011-3050
- `g_prefetch_experts` at line ~195 (temporal prediction state)
- `buf_multi_expert_data` (set A) and `buf_multi_expert_data_B` (set B) in MetalCtx

## Baseline
- 4-bit, K=4, Trust OS, no cache: 3.50-3.70 tok/s
- expert_io: 2.43ms/layer
- Target: reduce to ~1.5ms/layer → ~4.5-5.0 tok/s

## Risk
- SSD bandwidth contention between predicted and actual preads
- Previous speculative attempts all failed or were neutral
- Double-buffer complexity
- Must verify quality is preserved (same output with/without optimization)

## Previous Attempts (ALL FAILED)
- Speculative early routing on pre-attention state: 53% accuracy, cache pollution → slower
- F_RDADVISE hints: NVMe command contention → slower
- Temporal F_RDADVISE with lead time: 65-80% wrong predictions → slower
- mmap memcpy: 5.5x slower for cold data (page faults)

## What's Different This Time
- Loading into SCRATCH buffers (no cache pollution)
- Using CMD1_wait idle time (no additional CPU cost)
- Only predicting from previous token at same layer (simple, no gate_proj overhead if we just reuse stored indices)
- Only need to sync-pread the MISSES, not all 4 experts
