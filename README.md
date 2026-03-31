# PRE — Personal Reasoning Engine

> A 397-billion-parameter language model, running on a MacBook Pro. No cloud. No API keys. No data leaves your machine.

PRE is a fit-for-purpose deployment of **Qwen3.5-397B-A17B** — one of the most capable open-weight language models in existence — on a single Apple Silicon laptop. Powered by the [Flash-MoE](https://github.com/sunkencity999/flash-moe) inference engine, PRE streams a 209 GB model from your SSD through hand-tuned Metal compute shaders, delivering near-state-of-the-art reasoning at conversational speed with complete privacy.

The reference system is a **MacBook Pro with an M4 Max (128 GB unified memory)**, where it achieves **9+ tokens/second**. It also runs on machines with as little as 48 GB of RAM — slower, but with the same full-quality output from all 397 billion parameters.

![Progress](progress.png)

---

## Table of Contents

- [What Makes This Possible](#what-makes-this-possible)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Tutorial: Your First Session](#tutorial-your-first-session)
- [Command Reference](#command-reference)
- [Best Uses](#best-uses)
- [How the Engine Works](#how-the-engine-works)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## What Makes This Possible

Running a 397-billion-parameter model on a laptop sounds impossible. The entire model would need **~800 GB of memory** at full precision. Even at 4-bit quantization, the expert weights alone occupy **209 GB** — far more than any laptop's RAM.

The Flash-MoE engine solves this through a combination of techniques purpose-built for Apple Silicon hardware:

**Mixture-of-Experts architecture** — Qwen3.5-397B-A17B has 512 experts per layer, but only activates 4 per token. That means only ~17 billion parameters are computed for each token. The other 380 billion sit on your SSD until needed.

**SSD expert streaming** — The 4 active experts per layer (~27 MB total) are loaded from NVMe SSD on demand via parallel `pread()` system calls. Apple's internal SSDs deliver 7–17 GB/s sequential read throughput, making this feasible in real time.

**OS page cache as LRU** — Rather than building a custom cache (we tried — 58 experiments, all slower), the engine trusts macOS to manage expert caching in unused RAM. On the reference 128 GB system, ~90 GB of page cache yields high hit rates. On a 48 GB system, ~35 GB of cache still achieves ~71% hits.

**Hand-tuned Metal shaders** — Every GPU operation uses custom Metal compute kernels: FMA-optimized 4-bit dequantization, fused SwiGLU activation, two-pass RMS normalization, batched attention, and a single-kernel MoE combine+residual+norm. No framework overhead. No generic GEMM. Just the exact operations this model needs.

**Deferred GPU pipelining** — Expert computation is submitted to the GPU asynchronously. While the GPU processes one layer's experts, the CPU is already preparing the next layer's attention. This eliminates idle gaps in the pipeline.

The result: a model that would cost ~$0.50–$2.00 per conversation through a cloud API runs for free, as fast as you can type, with zero data exposure. Every optimization in the Flash-MoE engine exists to make this specific model practical on this specific class of hardware.

## Hardware Requirements

PRE is developed and tested on:

> **MacBook Pro** — Apple M4 Max, 16-core CPU, 40-core GPU, **128 GB unified memory**, 1 TB SSD
> **Performance:** 9.3 tokens/second, ~2s time-to-first-token

That's the reference configuration. Here's how other systems compare:

| Machine | Unified Memory | Speed | Notes |
|---------|---------------|-------|-------|
| **M4 Max MacBook Pro** | **128 GB** | **~9.3 tok/s** | **Reference system** — ~90 GB page cache, most experts warm |
| M3 Max MacBook Pro | 48 GB | ~4.4 tok/s | Fully functional, ~35 GB page cache, 71% hit rate |
| M3 Max Mac Studio | 96 GB | ~7 tok/s | Expected — sweet spot of price/performance |
| M2/M3/M4 Ultra Mac | 192 GB | ~12+ tok/s | Expected — nearly all experts cached in RAM |

**Minimum requirements:**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Chip** | Apple M1 Pro/Max | M3/M4 Max or Ultra |
| **Unified Memory** | 48 GB | 128 GB |
| **Free Disk Space** | 430 GB (install) / 215 GB (runtime) | 1 TB SSD |
| **macOS** | 14.0 (Sonoma) | 15.0+ |
| **Xcode CLI Tools** | Required | `xcode-select --install` |

The install process needs ~430 GB temporarily (HuggingFace cache + packed weights). After installation, you can clear the HuggingFace cache to reclaim ~214 GB if needed.

## Installation

### One-Line Install

```bash
git clone https://github.com/sunkencity999/pre.git
cd pre
./install.sh
```

The installer is fully automated and idempotent — safe to interrupt and re-run at any point. It will:

1. **Check your system** — verifies Apple Silicon, RAM, disk space, Xcode tools
2. **Install Python dependencies** — numpy, safetensors, huggingface-hub (for weight processing only)
3. **Download the model** — 214 GB from HuggingFace (resumable)
4. **Extract non-expert weights** → `engine/model_weights.bin` (5.5 GB)
5. **Export tokenizer** → `engine/tokenizer.bin` (7.8 MB)
6. **Repack expert weights** → `packed_experts/` (209 GB, 30–60 minutes)
7. **Compile the inference engine** — pure C/Objective-C/Metal, no external dependencies
8. **Install `pre-launch`** into your PATH

Total install time: **30–90 minutes** depending on internet speed and SSD performance.

### Manual Build (if model already downloaded)

```bash
cd engine
make              # Builds the inference server and PRE CLI
make install      # Symlinks pre-launch into ~/.local/bin
```

### Launching

```bash
pre-launch
```

From any directory. The launcher:
- Starts the inference server if it isn't already running
- Waits for the model to load (~10–15 seconds)
- Opens the PRE interface with your current directory as context
- Shuts down the server when you exit

You can also pass options:

```bash
pre-launch --show-think        # Show the model's reasoning process
pre-launch --max-tokens 16384  # Allow longer responses
PRE_PORT=9000 pre-launch       # Use a different port
```

---

## Tutorial: Your First Session

### 1. Start PRE

```bash
cd ~/my-project
pre-launch
```

You'll see the banner and a `pre>` prompt. PRE already knows your working directory and can see its files.

### 2. Ask a question

```
pre> What is this project? Describe its structure.
```

The model reads the directory listing injected as context on the first turn and responds. You'll see a spinner while it thinks, then streaming output with rendered markdown.

### 3. Attach a file for deep analysis

```
pre> /file src/main.py
  [attached: /Users/you/my-project/src/main.py (8.2K)]

pre> Review this code for bugs and security issues
```

The file contents are sent along with your message. The model analyzes them in full.

### 4. Attach multiple files

```
pre> /file config.yaml
pre> /file src/database.py
pre> /file src/auth.py
pre> How do these components interact? Is the auth flow secure?
```

### 5. Run a shell command and feed output to the model

```
pre> /run git log --oneline -20
  $ git log --oneline -20
  a1b2c3d Fix token refresh race condition
  ...
  [feed to model? y/n] y
  [output attached — type your message]

pre> Summarize the recent changes and identify any risky commits
```

### 6. Quick shell commands with `!`

```
pre> !grep -r "TODO" src/
pre> !docker ps
pre> !python3 -m pytest tests/ -q
```

These execute immediately and print output — the model doesn't see them unless you use `/run`.

### 7. Compose complex prompts with your editor

```
pre> /edit
```

This opens `$EDITOR` (vim, nvim, VS Code, etc.). Write a multi-line prompt, save, and quit. The text is attached to your next message.

### 8. Check how much context you've used

```
pre> /context

  Context Window
  ─────────────────────────────
  [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 28%
  Used:      ~9,100 tokens
  Budget:    32,768 tokens
  Remaining: ~23,668 tokens
```

### 9. Save and export your work

```
pre> /save analysis.md                  # Save last response to a file
pre> /export ~/Desktop/full-session.md  # Export entire conversation
pre> /rename "auth security review"     # Name the session for later
```

### 10. Resume later

```bash
pre-launch --sessions
#   pre-12345  auth security review  (14 turns)
#   pre-67890  (6 turns)  What is this project?

pre-launch --resume pre-12345
```

---

## Command Reference

### Chat & Input

| Command | Arguments | Description |
|---------|-----------|-------------|
| *(just type)* | | Send a message to the model |
| `!<command>` | | Execute a shell command directly (output not sent to model) |
| `/edit` | | Open `$EDITOR` for composing multi-line prompts |
| `/think` | | Toggle visibility of the model's `<think>` reasoning blocks |

### Files & Context

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/file` | `<path>` | Attach a file to your next message (supports multiple) |
| `/run` | `<command>` | Execute a command and optionally feed its output to the model |
| `/cd` | `<path>` | Change working directory (model sees new file listing on next turn) |
| `/ls` | `[path]` | List directory contents |
| `/tree` | `[path]` | Show directory tree (depth 3, skips .git/node_modules/etc.) |

### Session Management

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/new` | | Start a fresh session (resets context) |
| `/sessions` | | List all saved sessions with titles and turn counts |
| `/resume` | `<id>` | Resume a previous session by ID |
| `/rename` | `<name>` | Give the current session a human-readable name |
| `/save` | `<path>` | Save the model's last response to a file |
| `/export` | `[path]` | Export the full conversation to a markdown file |
| `/rewind` | `[N]` | Remove the last N turns from the session (default: 1) |
| `/summary` | | Ask the model to generate a bullet-point summary of the session |

### Status & Diagnostics

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/context` | | Visualize context window usage with a progress bar |
| `/stats` | | Show detailed statistics: tokens, speed, timing, session duration |
| `/status` | | Show current session state: model, server, CWD, settings |
| `/help` | | Display all available commands |

### Application

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/clear` | | Clear the screen |
| `/quit` | | Exit PRE (also: `/exit`, Ctrl-C, Ctrl-D) |

### Tool Calling (Automatic)

The model can request tool execution during conversations. These are handled automatically:

| Tool | Approval | Description |
|------|----------|-------------|
| `read_file` | Auto-approved | Read file contents (read-only) |
| `list_dir` | Auto-approved | List directory contents (read-only) |
| `bash` | Requires `y`/`n`/`a` | Execute a shell command (`a` = auto-approve for session) |

---

## Best Uses

### What PRE excels at

PRE puts a near-state-of-the-art reasoning model at your fingertips with zero compromise on privacy. These are the use cases where it truly shines:

**Deep code analysis and review** — Attach source files and get thorough, line-by-line analysis. The model's 397B parameter count gives it strong understanding of patterns across languages. Attach your auth module, your database layer, and your config — ask it to trace a request through the stack and find where things could break.

**Security and vulnerability research** — Analyze binaries, review network configurations, audit access controls, examine suspicious logs. Nothing leaves your machine. No content policy prevents you from discussing exploit techniques in a defensive context.

**Private document analysis** — Legal contracts, medical records, financial reports, HR documents, board communications. Feed them to a model that cannot leak them because it physically cannot reach the internet.

**Architecture and design reasoning** — Describe your constraints, attach your existing code, and get reasoned trade-off analysis. The model can hold complex multi-component systems in context and reason about their interactions.

**Research and knowledge synthesis** — Ask questions that require connecting ideas across domains. The 397B parameter count encodes a deep knowledge base — ask it to compare distributed consensus algorithms, explain the implications of a new paper, or reason through a complex technical decision.

**Extended thinking through hard problems** — Toggle `/think` to watch the model reason step-by-step through proofs, debugging sessions, or multi-factor decisions. The thinking tokens let it work through problems methodically before committing to an answer.

**Drafting and technical writing** — First drafts of documentation, incident reports, design proposals, and technical blog posts. Iterate locally without uploading your proprietary context to a cloud service.

### What PRE is not designed for

**Real-time agentic workflows** — At 4–9 tok/s, the edit-compile-test cycle that tools like Claude Code excel at would be too slow. PRE is for thinking, not for rapid-fire tool execution.

**High-throughput batch processing** — Processing thousands of documents at inference speed isn't practical. Use cloud batch APIs for scale.

**Multi-user serving** — PRE runs one conversation at a time. It's a personal tool, not a server you share.

**Tasks requiring very long output** — Generating 10,000+ token responses at 9 tok/s takes 18+ minutes. For bulk generation, cloud APIs are more practical.

---

## How the Engine Works

PRE inherits all of its inference performance from the [Flash-MoE](https://github.com/sunkencity999/flash-moe) engine — a pure C/Metal implementation built specifically for running large Mixture-of-Experts models on Apple Silicon. Here's what makes it fast:

### The Problem

Qwen3.5-397B-A17B has 60 transformer layers. Each layer contains 512 experts at 4-bit quantization (~6.75 MB each). Only 4 experts activate per token, but the engine must load them from SSD, dequantize them, and run them through the GPU — 60 times per token, every token.

### The Pipeline (4.3ms per layer)

```
CMD3(prev) → CMD1: attention projections + delta-net  [GPU, 1.2ms]
           → CMD2: output proj + residual + norm       [GPU, 0.6ms]
           → CPU:  softmax + top-K expert routing       [0.003ms]
           → I/O:  parallel pread K=4 experts from SSD  [2.4ms]
           → CMD3: expert forward + combine + norm      [GPU, deferred]
```

### Flash-MoE Optimizations

These are the specific techniques from the Flash-MoE project that make the 397B model viable on a laptop:

| Optimization | Impact | Description |
|-------------|--------|-------------|
| **FMA Dequant Kernel** | +12% tok/s | Rearranges 4-bit dequantization to use GPU fused multiply-add units: `fma(nibble, scale*x, bias*x)` instead of `(nibble*scale+bias)*x` |
| **CMD1+CMD2 Fusion** | +7% tok/s | Merges attention projection and output projection into a single Metal command buffer, eliminating GPU synchronization overhead |
| **Deferred GPU Expert Compute** | -0.8ms/layer | Expert forward pass runs on GPU asynchronously while CPU prepares the next layer |
| **BLAS Delta-Net** | +64% attn speed | GatedDeltaNet 128x128 state matrix updates use Apple Accelerate BLAS instead of scalar code |
| **Trust the OS Page Cache** | +38% vs custom cache | No custom LRU — macOS page cache manages expert data caching. Tested 58 alternatives; all were slower |
| **Parallel Expert I/O** | ~4x I/O speed | 4 experts loaded simultaneously via GCD dispatch groups at interactive QoS |
| **System Prompt Caching** | -2–4s/conversation | System prompt KV cache is snapshotted at startup and restored per request |

These optimizations were discovered through **90+ experiments** documented in the [technical paper](paper/flash_moe.pdf). Many plausible ideas (LZ4 compression, temporal prediction, GPU LUT dequant, mmap, speculative decoding) were tested and discarded — the paper explains why.

### Metal Compute Shaders (~1,200 lines)

Every GPU operation uses hand-written Metal kernels — no frameworks, no generic GEMM:

- 4-bit dequantized matrix-vector multiply (tiled, SIMD-reduced, FMA-optimized)
- Fused SwiGLU activation
- Two-pass RMS normalization (bfloat16 weight support)
- Batched GPU attention (Q@K^T → softmax → scores@V)
- GPU RoPE (fused with Q deinterleave and K normalization)
- MoE combine + residual + sigmoid gate (single kernel)
- GatedDeltaNet: conv1d, decay/beta, per-head RMS norm, z-gated output

### Why Apple Silicon

The engine is designed around Apple Silicon's **unified memory architecture**:

- **No PCIe bottleneck** — GPU and CPU share the same physical memory at ~400 GB/s
- **SSD as extended memory** — Apple's NVMe delivers 7–17 GB/s, fast enough to stream experts in real-time
- **Page cache scales with RAM** — More unified memory means more of the 209 GB expert pool stays cached. This is why the M4 Max 128 GB system is 2x faster than the M3 Max 48 GB: more experts are already in RAM

---

## Configuration

### System Prompt

The model's behavior is configured via `~/.flash-moe/system.md`, which is loaded and cached at server startup. Edit it to customize the model's persona, priorities, or knowledge about your projects.

```bash
$EDITOR ~/.flash-moe/system.md    # Edit, then restart the server
```

### Options

```bash
pre-launch --show-think            # See <think> reasoning blocks
pre-launch --max-tokens 16384     # Allow longer responses (default: 8192)
pre-launch --port 9000            # Use a different server port
PRE_PORT=9000 pre-launch          # Same, via environment variable
pre-launch --dir /path/to/project # Override working directory
```

### Session Data

All session data lives in `~/.flash-moe/`:

```
~/.flash-moe/
├── system.md           # System prompt (editable)
├── pre_history         # Input history (up-arrow recall)
└── sessions/
    ├── pre-12345.jsonl   # Conversation turns
    ├── pre-12345.title   # Session name
    └── ...
```

---

## Project Structure

```
pre/
├── install.sh              # Automated setup (download, extract, compile, install)
├── system.md               # Default system prompt
├── README.md
├── LICENSE
├── progress.png            # Performance benchmark chart
├── results.tsv             # Detailed experiment log
├── engine/
│   ├── infer.m             # Inference engine (~7,000 lines of C/Objective-C)
│   ├── pre.m               # PRE CLI (~1,900 lines)
│   ├── chat.m              # Lightweight chat client
│   ├── shaders.metal       # Metal GPU kernels (~1,200 lines)
│   ├── tokenizer.h         # Single-header BPE tokenizer
│   ├── linenoise.c/h       # Terminal line editor
│   ├── Makefile             # Build system (no external dependencies)
│   ├── pre-launch           # Universal launcher script
│   ├── extract_weights.py   # HF safetensors → model_weights.bin
│   ├── export_tokenizer.py  # HF tokenizer → tokenizer.bin
│   ├── repack_experts.py    # HF safetensors → packed_experts/
│   └── expert_index.json    # Expert location manifest
├── paper/
│   └── flash_moe.pdf       # Technical paper (90+ experiments)
└── docs/
    └── *.md                 # Optimization research notes
```

**Generated at runtime** (not committed to git):

| File | Size | Description |
|------|------|-------------|
| `engine/model_weights.bin` | 5.5 GB | Non-expert transformer weights (mmap'd) |
| `engine/model_weights.json` | 371 KB | Tensor offset manifest |
| `engine/tokenizer.bin` | 7.8 MB | Pre-compiled BPE tokenizer |
| `engine/vocab.bin` | 3.2 MB | Token vocabulary for decoding |
| `packed_experts/layer_*.bin` | 209 GB total | 60 files, 512 experts each at 4-bit |

---

## Contributing

PRE is built and maintained by **Christopher Bradford** — systems administrator, AI engineer, and the kind of person who thinks running a 400-billion-parameter model on a laptop is a reasonable thing to attempt.

This project is at an early stage and there's a lot of room to grow:

- **Performance** — There are more optimizations to find. The technical paper documents 58 experiments; there are surely more waiting to be discovered. Speculative decoding, better expert prediction, GPU-side routing, 2-bit quantization with intact tool calling — all open problems.
- **CLI features** — The PRE interface is functional but young. Better markdown rendering, syntax highlighting, image support, conversation branching, and more commands are all welcome.
- **Hardware profiles** — If you have an M2/M3/M4 Ultra or a Mac Studio/Pro with different RAM configurations, benchmarks and tuning for your hardware would be valuable.
- **Model support** — The engine is built for Qwen3.5-397B-A17B, but the architecture could support other large MoE models (Mixtral, DeepSeek, etc.) with weight repacking scripts.
- **Documentation** — Better tutorials, video walkthroughs, and use-case guides.

**How to contribute:**

1. Fork the repo and create a branch
2. Make your changes
3. Open a pull request with a clear description

Or just open an issue to discuss ideas, report bugs, or share performance numbers from your hardware.

**Contact:**
- GitHub: [@sunkencity999](https://github.com/sunkencity999)
- Issues: [github.com/sunkencity999/pre/issues](https://github.com/sunkencity999/pre/issues)

If this project is useful to you, a star helps others find it.

---

## Acknowledgments

- **Qwen Team** (Alibaba) for the remarkable Qwen3.5-397B-A17B model
- **MLX Community** for the 4-bit quantized weight distribution
- **Apple** for unified memory architecture and the Metal compute framework
- Inference engine built on [Flash-MoE](https://github.com/sunkencity999/flash-moe) — the research project behind the 90+ experiments that made this possible

## License

MIT License. See [LICENSE](LICENSE) for details.

The Qwen3.5-397B-A17B model weights are subject to the [Qwen License Agreement](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/LICENSE).
