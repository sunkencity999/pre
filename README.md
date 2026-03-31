# PRE — Personal Reasoning Engine

> A 397-billion-parameter language model running on your laptop. No cloud. No API keys. No data leaves your machine.

PRE is a purpose-built deployment of **Qwen3.5-397B-A17B** on Apple Silicon Macs. It combines a hand-tuned Metal inference engine with a rich CLI interface, delivering a near-state-of-the-art reasoning model at 4–9+ tokens/second entirely on local hardware.

![Progress](progress.png)

## Why This Exists

Cloud AI services are fast and convenient, but they come with trade-offs: your data crosses the wire, you pay per token, you're subject to content policies, and you're dependent on someone else's uptime. Small local models avoid these problems but sacrifice capability — a 7B model can't reason through complex problems the way a 400B model can.

PRE eliminates that trade-off. By streaming a 397-billion-parameter Mixture-of-Experts model from SSD through a custom Metal compute pipeline, it delivers reasoning quality comparable to frontier models while keeping everything local. The model's MoE architecture means only 17 billion parameters activate per token, making the compute tractable on Apple Silicon. The remaining 380 billion parameters live on your SSD and stream in on demand at NVMe speeds.

**This is not a toy.** Qwen3.5-397B-A17B is a production-grade model that handles complex multi-step reasoning, code analysis, technical writing, and tool calling. Running it locally means you can:

- Analyze proprietary source code without it touching any server
- Process sensitive documents (legal, medical, financial) with zero exposure
- Work on air-gapped or restricted networks
- Reason through complex problems with no rate limits or per-token costs
- Have a private, uncensored thinking partner available 24/7

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Chip** | Apple M1 Pro/Max | Apple M3/M4 Max or Ultra |
| **Unified Memory** | 48 GB | 128 GB |
| **SSD** | 500 GB free | 1 TB+ free |
| **macOS** | 14.0 (Sonoma) | 15.0+ |

The engine is architected around Apple Silicon's unified memory — GPU and CPU share the same address space, eliminating PCIe transfer bottlenecks. Performance scales with SSD bandwidth and GPU core count:

| Machine | RAM | tok/s | Notes |
|---------|-----|-------|-------|
| M3 Max, 40-core GPU | 48 GB | ~4.4 | Reference platform |
| M4 Max, 40-core GPU | 128 GB | ~9.3 | More RAM = larger OS page cache |
| M2/M3 Ultra | 192 GB | ~12+ | Expected (most experts cached in RAM) |

## Quick Start

```bash
git clone https://github.com/sunkencity999/pre.git
cd pre
./install.sh
```

The installer handles everything: model download (214 GB), weight extraction, expert repacking, compilation, and PATH setup. It's resumable — safe to interrupt and re-run.

Once installed:

```bash
pre-launch
```

That's it. From any directory. The launcher auto-starts the inference server, waits for it to load, and drops you into the PRE interface. On exit, the server shuts down automatically.

## The Interface

```
╔══════════════════════════════════════════════════╗
║  Personal Reasoning Engine (PRE)                ║
║  Qwen3.5-397B-A17B                             ║
╚══════════════════════════════════════════════════╝
  Server:  http://localhost:8000
  Session: pre-12345-1774969604
  CWD:     /Users/you/project

pre>
```

PRE is designed for the way engineers actually work — jumping between directories, reading files, running commands, and asking deep questions. It's not a chatbot with a text box. It's a reasoning engine wired into your filesystem.

### Commands

| Command | Description |
|---------|-------------|
| `/file <path>` | Attach a file for the model to analyze |
| `/edit` | Open `$EDITOR` for multi-line input |
| `/run <cmd>` | Run a shell command, optionally feed output to model |
| `/cd <path>` | Change working directory |
| `/ls [path]` | List directory contents |
| `/tree [path]` | Show directory tree |
| `/save <path>` | Save last response to file |
| `/export [path]` | Export full conversation to markdown |
| `/summary` | Ask the model to summarize the session |
| `/context` | Visualize context window usage |
| `/stats` | Detailed session statistics (tokens, speed, timing) |
| `/think` | Toggle visibility of `<think>` reasoning blocks |
| `/sessions` | List saved sessions |
| `/resume <id>` | Resume a previous session |
| `/rename <name>` | Name the current session |
| `/rewind [N]` | Remove last N conversation turns |
| `/new` | Start a fresh session |
| `/clear` | Clear screen |
| `/status` | Show current state |
| `/help` | Show all commands |
| `/quit` or `/exit` | Exit PRE |

### Shell Integration

Prefix any line with `!` to execute it as a shell command without leaving PRE:

```
pre> !git log --oneline -5
pre> !wc -l src/**/*.py
pre> !brew update
```

### File Analysis

Attach files before asking your question:

```
pre> /file src/auth.py
  [attached: /Users/you/project/src/auth.py (12.4K)]

pre> Is there a vulnerability in the session handling?
```

Multiple files can be attached before a single question:

```
pre> /file Cargo.toml
pre> /file src/main.rs
pre> /file src/lib.rs
pre> Explain this project's architecture
```

### Multi-Line Input

Use `/edit` to open your `$EDITOR` (vi, nvim, code, etc.) for composing complex prompts:

```
pre> /edit
  [attached 15 lines from editor — type your message or press Enter to send]

pre> Analyze the above and suggest improvements
```

### Tool Calling

The model can request tool execution during conversations. Read-only commands (ls, cat, grep, find, etc.) execute automatically. Write operations require your approval:

```
pre> What's in my project directory?

  [listing /Users/you/project]

The project contains...

  $ find . -name "*.test.js" | wc -l
  [execute? y/n/a(lways)]
```

### Session Management

Every conversation is automatically saved. Resume any session later:

```bash
pre-launch --sessions          # List all sessions
pre-launch --resume pre-12345  # Resume specific session
```

Inside PRE:
```
pre> /rename "auth refactor analysis"
pre> /export ~/Desktop/analysis.md
```

## How It Works

### The 60-Layer Pipeline

Each token passes through 60 transformer layers: 45 **GatedDeltaNet** layers (linear attention) and 15 standard full-attention layers. Each layer follows this pipeline:

```
CMD3(prev) → CMD1: attention projections + delta-net/attention  [GPU, 1.2ms]
           → CMD2: output proj + residual + norm + routing      [GPU, 0.6ms]
           → CPU:  softmax + top-K expert selection              [0.003ms]
           → I/O:  parallel pread 4 experts from SSD             [2.4ms]
           → CMD3: expert forward + combine + norm               [GPU, deferred]
                                                        Total: ~4.3ms/layer
```

At 60 layers, that's ~258ms per token, or ~3.9 tokens/second at baseline. The optimizations below push this significantly higher.

### Key Optimizations

**FMA Dequantized Matrix-Vector Multiply** — The inner loop of the 4-bit dequant kernel rearranges `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)`. Pre-computing `scale*x` and `bias*x` lets the GPU's fused multiply-add unit do dequant + multiply in one instruction. **+12% throughput.**

**CMD1+CMD2 Fusion** — For GatedDeltaNet layers, the attention projection (CMD1) and output projection + routing (CMD2) are merged into a single Metal command buffer. This eliminates a GPU command buffer boundary and the CPU synchronization overhead between them. The residual connection uses `buf_moe_hidden` directly from the previous layer's GPU output, avoiding a CPU round-trip. **+7% throughput.**

**Deferred GPU Expert Compute** — CMD3 (the expert MoE forward pass) is submitted to the GPU without waiting. The GPU executes expert projections, SwiGLU activation, MoE weighted combine, residual addition, and RMS normalization asynchronously while the CPU prepares the next layer's attention. **Eliminates 0.8ms idle time per layer.**

**BLAS-Accelerated Delta-Net** — The GatedDeltaNet recurrence maintains a 128×128 state matrix per head (64 heads). State updates use Apple Accelerate's `cblas_sscal`, `cblas_sgemv`, and `cblas_sger` for the decay, read, and rank-1 update operations. **+64% faster than scalar code** (0.78ms → 0.28ms per layer).

**Trust the OS Page Cache** — Expert weights (209GB at 4-bit) stream from SSD via parallel `pread()`. No custom cache. The macOS page cache manages LRU eviction across ~35GB of available memory, achieving ~71% hit rate naturally. Every custom caching approach tested (Metal LRU, malloc cache, LZ4 compression) was slower due to memory pressure or overhead. **+38% vs. Metal LRU.**

**Parallel Expert I/O** — Four experts load simultaneously via GCD dispatch groups on `QOS_CLASS_USER_INTERACTIVE` threads. Each expert is 6.75MB at 4-bit quantization. On warm cache, expert loading drops from 2.4ms to near-zero.

**System Prompt Caching** — On server start, the system prompt is pre-tokenized and forwarded through all 60 layers. The resulting KV caches, conv states, and delta-net states are snapshotted. Each new request restores from this snapshot instead of re-processing the system prompt, saving 2-4 seconds of prefill per conversation.

### Metal Compute Shaders

All GPU-side computation uses hand-written Metal shaders (~1,200 lines):

- **4-bit dequantized matvec** — Tiled, SIMD-reduced, shared input cache, FMA-optimized. Two variants: standard (4096-dim) and 8K (for output projection).
- **Fused SwiGLU** — `gate * silu(gate) * up` in a single kernel pass.
- **Two-pass RMS normalization** — Sum-of-squares reduction → apply with bfloat16 weight support.
- **Batched GPU attention** — Q@K^T → softmax → scores@V for full-attention layers.
- **GPU RoPE** — Fused with Q deinterleave and K normalization.
- **MoE combine + residual + gate** — Weighted sum of K expert outputs + sigmoid-gated shared expert + residual, in one kernel.
- **GatedDeltaNet kernels** — Conv1d step, decay/beta computation, RMS norm per head, z-gated output normalization.

### Unified Memory Architecture

Apple Silicon's unified memory is both an advantage and a constraint. The GPU and SSD DMA share the same memory controller, which means:

- **No PCIe bottleneck** — GPU reads from the same physical memory as CPU, at ~400 GB/s bandwidth.
- **No profitable overlap** — SSD DMA and GPU compute cannot run simultaneously without memory controller arbitration causing GPU latency spikes. The serial pipeline (GPU → SSD → GPU) is hardware-optimal.
- **Page cache is king** — On a 128GB machine, ~90GB is available for OS page cache. Experts that were recently used are already in RAM. On a 48GB machine, ~35GB of cache still achieves 71% hit rate.

## What This Model Is Good At

### Deep Technical Analysis

With 397 billion parameters and extended thinking, the model excels at problems that require holding many facts in context simultaneously:

- **Code review** — Attach source files and get line-by-line analysis with security, correctness, and performance insights.
- **Architecture design** — Describe constraints, get reasoned trade-off analysis.
- **Debugging** — Paste error logs and stack traces. The model reasons through causality chains.
- **Research synthesis** — Attach papers or technical docs and get structured summaries with critical analysis.

### Privacy-Sensitive Work

Nothing leaves your machine. Use cases that are uncomfortable or impossible with cloud AI:

- Analyzing proprietary algorithms or trade secrets
- Processing PII, medical records, or legal documents
- Security research and vulnerability analysis
- Internal incident response and forensics
- Drafting sensitive communications

### Knowledge Work

The model's 397B parameter count gives it a deep knowledge base:

- Technical writing and documentation
- Explaining complex systems at varying levels of detail
- Comparative analysis (technologies, approaches, designs)
- Historical and domain-specific research

### Extended Reasoning

The `<think>` blocks allow the model to reason through problems step-by-step before responding. Toggle visibility with `/think`:

```
pre> /think
  [thinking blocks: visible]

pre> Prove that there are infinitely many primes
```

## What This Model Is Not

PRE generates 4–9 tokens per second. That's fast enough for conversation but not for:

- **Real-time agentic loops** — If you need sub-second tool call cycles (like Claude Code's edit-test-fix loop), a cloud API is better.
- **Bulk document processing** — Processing 1000 files at 9 tok/s would take days. Use batch APIs.
- **Multi-user serving** — The engine handles one conversation at a time. It's a personal tool.

## Project Structure

```
pre/
├── install.sh              # Full automated setup
├── system.md               # Default system prompt (copied to ~/.flash-moe/)
├── README.md               # This file
├── progress.png            # Performance results chart
├── results.tsv             # Detailed experiment log
├── engine/
│   ├── infer.m             # Complete inference engine (~7,000 lines)
│   ├── pre.m               # PRE CLI application (~1,900 lines)
│   ├── chat.m              # Lightweight chat client
│   ├── shaders.metal       # Metal GPU kernels (~1,200 lines)
│   ├── tokenizer.h         # Single-header C BPE tokenizer
│   ├── linenoise.c/h       # Line editor library
│   ├── main.m              # MoE benchmark harness
│   ├── Makefile             # Build system
│   ├── pre-launch           # Universal launcher script
│   ├── extract_weights.py   # HF safetensors → model_weights.bin
│   ├── export_tokenizer.py  # HF tokenizer → tokenizer.bin
│   ├── repack_experts.py    # HF safetensors → packed_experts/
│   ├── repack_experts_2bit.py  # Optional 2-bit requantization
│   └── expert_index.json    # Expert location manifest
├── paper/
│   └── flash_moe.pdf       # Technical paper (90+ experiments)
└── docs/
    └── *.md                 # Optimization research notes
```

### Generated at Runtime (not in git)

```
engine/
├── model_weights.bin       # 5.5 GB — non-expert weights (mmap'd)
├── model_weights.json      # Tensor manifest
├── vocab.bin               # 3.2 MB — vocabulary for decoding
└── tokenizer.bin           # 7.8 MB — BPE tokenizer data

packed_experts/
├── layer_00.bin            # 3.6 GB — 512 experts for layer 0
├── layer_01.bin            # 3.6 GB
├── ...
└── layer_59.bin            # 3.6 GB
                            # Total: 209 GB
```

## Configuration

### System Prompt

Edit `~/.flash-moe/system.md` to customize the model's behavior. The system prompt is cached at server startup — changes take effect on next server restart.

### Server Port

```bash
PRE_PORT=9000 pre-launch     # Use a different port
pre-launch --port 9000       # Same thing
```

### Visible Reasoning

```bash
pre-launch --show-think      # See <think> blocks in output
```

Or toggle inside PRE with `/think`.

### Max Tokens

```bash
pre-launch --max-tokens 16384   # Longer responses (default: 8192)
```

## Building from Source

If you prefer to build manually instead of using `install.sh`:

```bash
cd engine
make                    # Builds infer + pre
make install            # Symlinks pre-launch into ~/.local/bin
```

Requirements: Apple Clang (from Xcode CLI tools), macOS frameworks (Metal, Foundation, Accelerate). No external dependencies.

## The Technical Paper

The full technical details — including 90+ experiments, discarded approaches, and the complete optimization story — are in [`paper/flash_moe.pdf`](paper/flash_moe.pdf).

## Acknowledgments

- **Qwen Team** (Alibaba) for the Qwen3.5-397B-A17B model
- **MLX Community** for the 4-bit quantized weights
- **Apple** for unified memory architecture and the Metal compute framework
- Built on the [Flash-MoE](https://github.com/sunkencity999/flash-moe) inference engine

## License

MIT License. See [LICENSE](LICENSE) for details.

The Qwen3.5-397B-A17B model weights are subject to the [Qwen License Agreement](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/LICENSE).
