# I/O and GPU Exploration: What We Learned Running a 397B Model from SSD

## The Problem

We're streaming a 397 billion parameter Mixture-of-Experts model from NVMe SSD on a MacBook Pro with 48GB RAM. The model's expert weights total 120GB at 2-bit quantization (209GB at 4-bit). Only 6GB fits in memory. Every generated token requires reading ~600MB of expert data from disk — 4 experts × 3.9MB × 60 layers.

The question that drove months of optimization: **where does the time actually go, and what can we do about it?**

## Part 1: The GPU Story

### What the profiler showed us

We captured a Metal GPU trace of the expert forward pass (the most compute-intensive per-token operation). The results were surprising:

- **Total GPU compute time for 20 expert matvecs: 747µs** (37µs each)
- **Wall clock time: 31.5ms** (1.58ms each)
- **GPU utilization: 2.4%**

The GPU finishes its actual math in microseconds, then sits idle waiting for the next batch of data. The "compute" bottleneck is really a **data delivery** bottleneck.

### Where GPU cycles go

The instruction cost breakdown from Metal's performance counters:

| Category | % of GPU Time | What it is |
|----------|:---:|---|
| Math (FMA/MUL/ADD) | 63.9% | The actual dequant + multiply-accumulate |
| Conversion | 25.0% | `bf16_to_f32()` calls for scale/bias lookup |
| Data Movement | 3.3% | Moving data between register files |
| Bit Manipulation | 19.0% | Extracting 2-bit values from uint32 |

**25% of GPU time is type conversion.** Every scale and bias value is stored as bfloat16 and converted to float32 on the fly. Storing them as float32 would double the scale/bias storage (negligible overall) but eliminate a quarter of the GPU work.

**19% is bit manipulation.** Extracting 16 × 2-bit values from each uint32 requires shift-and-mask operations. A lookup table or wider SIMD approach could reduce this.

But here's the kicker: even if we eliminated ALL of this overhead, saving 44% of GPU time would save 44% of 37µs = 16µs per matvec. At 720 matvecs per token (4 experts × 3 projections × 60 layers), that's 11.5ms per token. Not nothing — but the I/O bottleneck at 90ms/token makes it a secondary concern.

### Cache behavior

- **L1 Cache Read Hit Rate: 93.4%** — our threadgroup shared memory (`x_shared[4096]`) works perfectly for caching the input vector
- **L1 Cache Write Hit Rate: 100%** — all output writes hit L1
- **Last Level Cache Bandwidth: ~418 GB/s** — nearly saturating unified memory bandwidth

The GPU cache hierarchy is working well. The 6.6% L1 read miss rate corresponds to expert weight data that doesn't fit in L1 (each expert is 3.9MB, L1 is ~192KB per core). These misses go to the shared L2 / memory fabric, which runs at near-theoretical bandwidth.

### The GPU cluster affinity experiment

Apple's M3 Max has 40 GPU cores organized into clusters, each with its own L2 cache (~4MB). Our 2-bit experts are 3.9MB — almost exactly one cluster's L2 capacity.

**Hypothesis:** If we encode all 4 operations for one expert (gate → up → SwiGLU → down) into a single Metal command encoder, the GPU scheduler would keep that work on one cluster, and the expert's weight data would stay hot in that cluster's L2.

**Result:** 2% slower. The fused single-encoder approach reduced parallelism — Metal's scheduler couldn't overlap work across experts anymore. The existing 2-encoder-per-expert approach (gate+up together, SwiGLU+down together) lets the GPU interleave expert computations across clusters, which provides better throughput than L2 locality.

**Lesson:** GPU schedulers are smarter than manual NUMA pinning. Don't fight the hardware scheduler unless you have profiling data showing it's making bad decisions.

### What doesn't matter on the GPU

- **Superpages (2MB pages):** Apple Silicon ARM64 uses fixed 16KB pages. `vm_allocate` with `VM_FLAGS_SUPERPAGE_SIZE_2MB` returns `KERN_INVALID_ARGUMENT`. Not available.
- **Command buffer type:** `commandBufferWithUnretainedReferences` (skip ARC retain/release) vs `commandBuffer` — zero measurable difference.
- **Encoder count:** Batching all experts' gate+up into one encoder vs separate encoders per expert — zero difference. Metal handles both patterns efficiently.

## Part 2: The I/O Story

### The landscape

Our I/O benchmark measured raw SSD performance for the expert read pattern:

| Access Pattern | Throughput | Latency (4 experts) |
|---|:---:|:---:|
| Sequential, warm page cache | 32.1 GB/s | 0.49 ms |
| Parallel 4T, warm cache | 29.2 GB/s | 0.97 ms |
| Parallel 4T, cold (F_NOCACHE) | 5.5 GB/s | 2.84 ms |
| Sequential, cold | 4.5 GB/s | 3.46 ms |
| mmap + memcpy, cold | 0.12 GB/s | varies |

The gap between warm (32 GB/s) and cold (5.5 GB/s) is the entire optimization story. Everything we tried was about moving more data from cold to warm.

### The mmap disaster

**What:** Replace `pread()` with `mmap()` + `memcpy()` for zero-syscall access to cached data.

**Result:** 0.56 tok/s — **5x slower** than pread.

**Why:** Each 3.9MB expert spans 240 × 16KB pages. For uncached data, mmap triggers 240 individual page faults, each requiring a separate kernel trap → I/O request → page table update. A single `pread()` call issues one large NVMe command for the entire 3.9MB range.

**Lesson:** `mmap()` is designed for random access to already-cached data. For bulk reads of potentially uncached data, `pread()` is dramatically better because it lets the kernel optimize the I/O pattern.

### The custom cache trap

We built increasingly sophisticated expert caching systems:

| Cache Type | Entries | Memory | Hit Rate | tok/s | Verdict |
|---|:---:|:---:|:---:|:---:|---|
| None (pread only) | 0 | 0 | 0% | 2.86 | Baseline |
| Metal LRU (500) | 500 | 3.5 GB | 35% | 3.14 | Small win |
| Metal LRU (1000) | 1000 | 7.1 GB | 44% | 2.24 | **Worse** |
| Metal LRU (2500) | 2500 | 9.8 GB | 55% | 2.24 | **Worse** |
| Metal LRU (3000) | 3000 | 21 GB | 55% | 1.99 | **Much worse** |
| Malloc zero-copy (2581) | 2581 | 18 GB | 52% | 2.10 | **Worse** |
| **No cache, trust OS** | **0** | **0** | **OS-managed** | **5.74** | **Best** |

**The breakthrough:** Deleting the entire custom cache system and letting macOS manage the page cache yielded a **38% speedup** over our best custom implementation.

**Why custom caches hurt:**
1. **Metal buffer caches wire memory.** Every Metal buffer allocation is pinned in physical RAM (wired pages). Our 9.8GB cache wired 9.8GB, leaving only ~25GB for the OS page cache instead of ~35GB.
2. **The OS page cache is smarter.** macOS uses CLOCK-Pro (an adaptive replacement algorithm that balances recency and frequency). Our LRU cache was strictly recency-based.
3. **Zero lookup overhead.** The OS page cache operates at the virtual memory level — a cache "hit" is just a normal memory access through the MMU. Our cache required hash table lookups, pointer chasing, and LRU bookkeeping.
4. **Memory pressure compounds.** `vm_stat` monitoring showed that with the Metal cache active, the compressor was doing 60,000-130,000 decompressions per second. Without it: near zero. The wired cache pages forced the OS to compress other data, and decompressing it on access added latency everywhere.

**The database analogy:** PostgreSQL recommends keeping `shared_buffers` at 25% of RAM and letting the OS cache handle the rest. We were doing the equivalent of setting shared_buffers to 60% of RAM — squeezing out the OS cache that handles the long tail of access patterns better than any application-level cache.

### The kernel hint experiments

We tried every macOS I/O hint available:

| Hint | Purpose | Result | Why |
|---|---|:---:|---|
| `F_NOCACHE` | Bypass page cache | +3% (2-bit) | Avoids thrashing when working set >> cache. But prevents warm hits. |
| `F_RDAHEAD` | Enable readahead | 0% | Kernel already does readahead for pread. |
| `F_RDADVISE` (immediate) | Pre-hint reads | -8% | Creates NVMe command contention — double-issues reads. |
| `F_RDADVISE` (with lead time) | Pre-hint from previous token | -4% | 65-80% of predictions wrong (different routing). Wrong advises waste bandwidth. |
| `MADV_RANDOM` | Disable readahead | **Harmful** | Fragments 3.9MB reads into 5.7 × 512KB disk ops. |
| `MADV_SEQUENTIAL` | Large readahead | 0% | Fragmentation is physical page layout, not readahead policy. |
| `MADV_WILLNEED` | Pre-populate cache | 0% on steady state | Only helps first access, not sustained generation. |
| No hint (default) | Let kernel decide | **Best** | Kernel's default behavior is already well-tuned for Apple hardware. |

**The pattern:** Every hint we tried either made no difference or made things worse. The macOS kernel is already optimized for Apple's NVMe controller. Application-level hints add overhead (each `fcntl` / `madvise` is a syscall) without providing information the kernel doesn't already have.

### The fragmentation discovery

`fs_usage` profiling revealed the kernel's internal behavior:

```
pread calls:     45,414
RdData ops:     260,845
Reads per pread: 5.7x
```

Each 3.9MB pread is broken into ~5.7 separate NVMe commands, mostly 512KB (0x80000) and smaller. The reason: the page cache stores data in scattered 16KB virtual pages that map to non-contiguous physical pages. The kernel can't coalesce them into a single DMA transfer.

**Block size distribution from fs_usage:**
```
76,549 × 512KB   (0x80000)
38,441 × 8KB     (0x2000)
20,064 × 16KB    (0x4000)
17,647 × 12KB    (0x3000)
14,651 × 256KB   (0x40000)
```

This fragmentation adds ~46µs of kernel overhead per pread (240 pread calls/token × 46µs = 11ms/token). It's inherent to the virtual memory system and can't be fixed from userspace.

### Buffer alignment matters

Our isolated benchmark showed a dramatic difference based on destination buffer alignment:

| Buffer Alignment | Avg Latency | Throughput |
|---|:---:|:---:|
| 2MB-aligned (`posix_memalign`) | 234 µs | 16.8 GB/s |
| 16KB-aligned (default Metal) | 836 µs | 4.7 GB/s |

**3.6x faster with 2MB alignment** for page-cache-resident data. The DMA controller can do larger, more efficient burst transfers when the destination is aligned to large boundaries.

In the full pipeline, the improvement was more modest (5%) because most reads hit SSD (cold data), where the DMA controller's performance is dominated by NAND flash latency rather than buffer alignment. But it's a free optimization — `posix_memalign` + `newBufferWithBytesNoCopy` costs nothing at runtime.

### The tiered I/O experiment

**Hypothesis:** Use two file descriptors per layer file — one with `F_NOCACHE` for first-time reads (avoid polluting page cache with one-off data), one without for repeat reads (benefit from page cache). Track "seen" experts with a 3.8KB bitset.

**Result:** Marginally better tok/s, but `vm_stat` showed identical memory pressure. The memory pressure was from Metal buffers and model weights, not from page cache behavior. The tiered approach added complexity without meaningful benefit.

### What actually worked for I/O

1. **2-bit expert quantization** — 44% smaller files. Reduced expert_io from 2.6ms to 1.5ms per layer. The single biggest improvement.
2. **Trust the OS page cache** — Delete custom caches. Let macOS manage memory. 38% speedup.
3. **2MB-aligned DMA buffers** — 5% improvement on expert_io. Free optimization.
4. **Parallel pread (4 threads)** — 9.2x speedup over sequential (superlinear due to NVMe command queuing).
5. **No kernel hints** — Default behavior is already optimal. Every hint we tried was neutral or harmful.

## Part 3: The Bigger Picture

### The SSD bandwidth wall

At 2-bit precision, the theoretical I/O floor for our workload is:

```
60 layers × ~2.6 cache misses × 3.9MB = 608MB per token
608MB ÷ 5.5 GB/s (random read throughput) = 110ms
110ms → 9.1 tok/s theoretical maximum (I/O limited)
```

Our measured performance of 5.5 tok/s (182ms/token) is split roughly 50/50 between I/O (90ms) and compute (90ms). We're at 82% of the I/O-limited theoretical maximum.

The remaining 18% gap is the page cache fragmentation overhead (5.7 ops/pread) and the kernel's per-read overhead (46µs/pread). These are architectural limitations of macOS's virtual memory system.

### Systems thinking beats micro-optimization

The single most impactful change in this entire project was **deleting code**: removing the 9.8GB Metal buffer cache. It wasn't that the cache was poorly implemented — it was that the cache's existence created system-level effects (memory pressure, compressor thrashing, reduced page cache) that outweighed its direct benefits.

This is a classic systems engineering lesson: optimizing one component in isolation can degrade the whole system. The GPU profiling showed us the compute isn't the bottleneck. The I/O profiling showed us the kernel is already doing a good job. The `vm_stat` monitoring showed us our "optimization" was causing the real problem.

### What the database world already knew

Dan Woods brought the key insight: **treat the model weights like a database.** Databases have solved the problem of accessing datasets larger than memory for decades:

- **Don't build your own buffer pool.** PostgreSQL learned this — `shared_buffers` should be 25% of RAM, not 100%. The OS buffer cache handles the long tail better.
- **Respect the hardware cache hierarchy.** Don't bypass caches (F_NOCACHE) unless you have measured evidence of thrashing. The caches exist for a reason.
- **Profile before optimizing.** `fs_usage` and Metal GPU traces told us exactly where time goes. Without them, we would have optimized the wrong thing.
- **Alignment matters for DMA.** Database systems align I/O buffers to page boundaries. We found 2MB alignment gives 3.6x better DMA throughput.

### Remaining frontiers

1. **Batch prefill** — Process multiple prompt tokens simultaneously. The sequential GatedDeltaNet recurrence limits parallelism, but projection matmuls and expert I/O can be batched.
2. **C tokenizer** — Done. Eliminated the 3.5s Python overhead, bringing setup from 4s to 180ms.
3. **The page cache fragmentation** — 5.7 disk ops per pread is a kernel limitation. The only userspace mitigation would be to use `mincore()` to detect cached pages and `memcpy()` from mmap for hits, falling back to pread for misses.
4. **Expert file layout optimization** — Co-locating frequently co-accessed experts could reduce the number of distinct NVMe commands per token. This requires offline analysis of routing patterns.
