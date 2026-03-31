# Plan: I/O Optimization Experiments

## Baseline Reference
- Per-layer: 4.28ms total, 2.41ms expert_io (56%)
- Expert read: 4 × 7MB parallel pread from 3.4GB file
- Measured: 5.8ms cold parallel, 1.0ms warm parallel, 2.4ms mixed (71% cache hit)
- Theoretical floor: 28MB / 17.5 GB/s = 1.6ms
- Gap: 0.8ms overhead per layer (kernel VFS + page cache + NVMe scheduling)

---

## Experiment 1: dispatch_io vs pread

### Isolated test
Read 4 experts (7MB each) from a layer file using:
- (A) 4 × pread on 4 pthreads (current approach)
- (B) 4 × dispatch_io_read on a DISPATCH_IO_RANDOM channel
- Both with F_NOCACHE to force SSD reads. Both with warm cache. 50 iterations each.

Measure: wall time, throughput (GB/s logical).

### What dispatch_io does differently
- Creates a kernel-side I/O channel with optimized scheduling
- The kernel sees all reads as part of one channel → can reorder NVMe commands by LBA
- Automatic cleanup handlers (no thread join overhead)
- May use a different VFS code path optimized for random access

### Pipeline contention analysis
- dispatch_io is async with completion blocks on GCD queues
- The completion block runs on a GCD thread (same as our async_pread)
- Memory path: SSD → DMA → DRAM → Metal shared buffer (same as pread)
- **No new contention** — same memory path as current approach
- Risk: dispatch_io might add GCD overhead that exceeds VFS savings
- Risk: completion blocks might have higher latency than pthread wakeup

### Expected impact
- Best case: 10-20% expert_io reduction (eliminates per-syscall VFS overhead)
- Worst case: neutral or slight regression (GCD completion overhead)
- Pipeline impact: drop-in replacement for pread, no GPU interaction

---

## Experiment 2: GPU private buffer compression

### Isolated test
- (A) GPU matvec reading from StorageModeShared buffer (current)
- (B) GPU blit shared→private, then matvec reading from StorageModePrivate buffer
- Use same expert weight data, same kernel, same dimensions
- Measure: blit time, matvec time, total. 100 iterations.

### What GPU compression does
- StorageModePrivate buffers live in GPU-managed memory
- The GPU's memory controller can apply lossless compression (similar to console GPU
  texture compression — transparent to shaders)
- For compressible data: effective bandwidth doubles (read 64B, decompress to 128B)
- 4-bit quantized weights with 2.4-3.7 bits entropy → highly compressible
- The shader code doesn't change at all — compression is hardware-transparent

### Pipeline contention analysis
- Blit (shared→private) runs on the GPU command queue
- It would go BEFORE the matvec dispatches in CMD3
- Timeline: [pread→shared buf] → [GPU blit 0.02ms] → [GPU matvec from private]
- The blit adds ~0.02ms per expert to CMD3
- But the matvec might be 30-50% faster from doubled bandwidth
- **Key contention**: the blit and matvec are both on the same GPU queue (serial)
  The blit cannot overlap with the matvec. It's purely: does the bandwidth gain
  from compression exceed the blit cost?
- **Memory**: private buffers use GPU-managed memory. 4 × 7MB = 28MB of private
  memory per layer. The GPU manages this pool — may cause memory pressure if the
  pool grows. Need to reuse/recycle the private buffers each layer.

### Expected impact
- Best case: 15-30% cmd1_wait reduction (CMD3 expert matvec faster)
- Worst case: slight regression (blit cost exceeds compression benefit)
- Pipeline impact: affects GPU phases only, no SSD interaction

---

## Experiment 3: Expert file clustering by co-occurrence

### Isolated test
- Run 500 tokens with --freq, collect per-layer expert co-occurrence matrix
- For each layer: cluster the 512 experts so frequently co-occurring experts are adjacent
- Repack each layer file with the new ordering (+ save the permutation map)
- Measure: 4-expert parallel pread with original vs clustered ordering
- Use same expert indices (mapped through permutation), F_NOCACHE, 50 iterations

### What clustering does
- NVMe SSDs read in pages (4KB-16KB). When we read expert 37 (7MB at offset 262MB),
  the SSD reads pages 262.0-269.0 MB. Expert 38 is at 269-276 MB — adjacent.
- If the routing selects experts {37, 42, 100, 205}, those are at offsets
  {262, 297, 708, 1451} MB — widely scattered
- If we reorder so co-occurring experts are adjacent: {37, 42, 100, 205} might become
  physical positions {0, 1, 2, 3} — a 28MB sequential read instead of 4 scattered reads
- Sequential 28MB at 17.5 GB/s = 1.6ms vs scattered 4×7MB at ~5.8ms

### Pipeline contention analysis
- This changes the FILE LAYOUT only — the inference code reads the same way
- Expert indices get mapped through a permutation table (one array lookup, ~0ns)
- **No contention** — purely changes which bytes are at which file offsets
- The only risk: co-occurrence patterns change with different prompts.
  If the clustering is prompt-dependent, it might help some prompts and hurt others.
- Mitigation: use a diverse set of prompts for profiling

### Expected impact
- Best case: 30-50% expert_io reduction for cold reads (scattered → near-sequential)
- Worst case: neutral (if co-occurrence is too flat/prompt-dependent)
- Pipeline impact: pure I/O improvement, no GPU/CPU interaction

---

## Experiment 4: LZ4 DRAM expert cache

### Isolated test
- Allocate 4GB of malloc'd memory
- After each expert pread, LZ4-compress and store in the cache (hash by layer+expert_id)
- On subsequent reads: check cache first. Hit = LZ4 decompress from DRAM.
- Measure: cache hit rate over 200 tokens, avg expert read time (hit vs miss)

### What this does
- Creates a second-level cache between the OS page cache and SSD
- Stores experts in compressed form → 4GB holds ~730 experts (vs ~570 raw)
- Cache hit: decompress at 41 GB/s = 0.17ms per expert
- Cache miss: pread from SSD/page cache = 0.3-1.5ms per expert
- The cache is in USERSPACE memory — doesn't interfere with OS page cache

### Pipeline contention analysis
- Cache lookup: hash table check (~0.001ms) — negligible
- Cache hit: LZ4 decompress runs on the I/O worker thread (CPU)
  - During decompress, the CPU is busy for 0.17ms
  - This overlaps with other threads' preads (parallel, no contention)
- Cache miss: normal pread path (no change)
- **Memory contention**: 4GB of malloc'd DRAM reduces available page cache by 4GB
  (OS has 4GB less to work with). Current page cache: ~35GB → ~31GB.
  31GB / 7MB = ~4430 expert slots (vs current 5000). FEWER raw experts cached.
  BUT the userspace cache adds 730 compressed experts on top.
  Total accessible experts: 4430 (page cache) + 730 (LZ4 cache) = 5160.
  That's only 3% more than current 5000. Barely worth it.
- **If we increase to 8GB cache**: 27GB page cache (3857 slots) + 1455 LZ4 = 5312. Still marginal.
- **The math doesn't work** unless the LZ4 cache has MUCH higher hit rate than the
  page cache (e.g., by using a smarter eviction policy than LRU).

### Expected impact
- Best case: 5-10% expert_io reduction (LZ4 cache hits for hottest experts)
- Worst case: negative (reduced page cache hurts more than LZ4 cache helps)
- Pipeline impact: CPU time for decompress, but overlaps with parallel preads
- **VERDICT: probably not worth implementing given the math above**

---

## Experiment 5: aio_read batching

### Isolated test
- (A) 4 × pread on 4 pthreads (current)
- (B) 4 × aio_read, then aio_suspend to wait for all
- Both F_NOCACHE. Both warm cache. 50 iterations.

### What aio_read does differently
- Submits I/O requests to the kernel without blocking the calling thread
- The kernel sees all 4 requests at once and can batch NVMe commands
- aio_suspend blocks until all 4 complete (single wait vs 4 thread joins)
- Eliminates per-thread overhead (no pthread_create/join or dispatch_group)

### Pipeline contention analysis
- aio_read uses kernel-level async I/O (not userspace threads)
- The kernel's I/O scheduler has full visibility into all pending reads
- **No new contention** — same SSD path, potentially better NVMe scheduling
- Risk: macOS aio implementation might be less optimized than GCD
  (Apple generally prefers dispatch_io over POSIX aio)
- The completion notification (SIGEV_THREAD or SIGEV_SIGNAL) has latency

### Expected impact
- Best case: 5-15% expert_io reduction (better NVMe command batching)
- Worst case: neutral or regression (aio overhead on macOS)
- Pipeline impact: drop-in replacement for pread

---

## Execution Priority

| # | Experiment | Expected Impact | Effort | Risk | Priority |
|---|-----------|----------------|--------|------|----------|
| 3 | Expert clustering | 30-50% cold I/O | Medium | Low | **1st** |
| 1 | dispatch_io | 10-20% I/O | Low | Low | **2nd** |
| 2 | GPU private compression | 15-30% GPU | Medium | Medium | **3rd** |
| 5 | aio_read | 5-15% I/O | Low | Low | **4th** |
| 4 | LZ4 DRAM cache | 5-10% I/O | High | High | Skip |

## Key Prediction: Compound Effects

The reason to test each in isolation FIRST: on unified memory, improvements that look
good alone can cancel each other (like F_RDADVISE's expert_io -31% + cmd2_wait +73% = net 0%).

After isolated tests, the compound analysis:
- Experiment 1 + 3 (dispatch_io + clustering): better I/O scheduling + fewer scattered reads.
  These should compound because clustering reduces scatter and dispatch_io optimizes remaining scatter.
- Experiment 2 is GPU-only, orthogonal to I/O experiments. Should compound cleanly.
- Experiment 5 is alternative to experiment 1 — test both, pick winner.
