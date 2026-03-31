#!/usr/bin/env python3
"""
repack_experts_3bit.py — TurboQuant 3-bit repacking with Hadamard rotation.

Reads packed_experts/layer_XX.bin files (512 experts x 7,077,888 bytes each)
and writes packed_experts_3bit/layer_XX.bin (512 experts x 5,505,024 bytes each).

Key difference from naive requantization: applies Walsh-Hadamard Transform (WHT)
to weight matrix columns before quantization. This distributes outlier values
more uniformly, significantly reducing quantization error.

3-bit format (per expert, 5,505,024 bytes):
  gate_proj: weights [1024, 384] u32 (planar 3-bit) + scales [1024, 64] bf16 + biases [1024, 64] bf16
  up_proj:   weights [1024, 384] u32 (planar 3-bit) + scales [1024, 64] bf16 + biases [1024, 64] bf16
  down_proj: weights [4096, 96]  u32 (planar 3-bit) + scales [4096, 16] bf16 + biases [4096, 16] bf16
  Total: 5,505,024 bytes  (22.2% reduction from 4-bit)

Planar 3-bit packing: 32 values stored in 3 x uint32 (bit-plane layout).
  word0: bit 0 of each value at position [0..31]
  word1: bit 1 of each value
  word2: bit 2 of each value
  val[i] = ((word0>>i)&1) | (((word1>>i)&1)<<1) | (((word2>>i)&1)<<2)  -> [0,7]

Usage:
    python repack_experts_3bit.py [--layer N] [--verify]
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy.linalg import hadamard


# ============================================================================
# 4-bit expert layout (from repack_experts_2bit.py)
# ============================================================================

EXPERT_SIZE_4BIT = 7_077_888
NUM_EXPERTS = 512
GROUP_SIZE = 64

GATE_W_OFF_4 = 0
GATE_W_SIZE_4 = 2_097_152
GATE_S_OFF_4 = 2_097_152
GATE_S_SIZE_4 = 131_072
GATE_B_OFF_4 = 2_228_224
GATE_B_SIZE_4 = 131_072

UP_W_OFF_4 = 2_359_296
UP_W_SIZE_4 = 2_097_152
UP_S_OFF_4 = 4_456_448
UP_S_SIZE_4 = 131_072
UP_B_OFF_4 = 4_587_520
UP_B_SIZE_4 = 131_072

DOWN_W_OFF_4 = 4_718_592
DOWN_W_SIZE_4 = 2_097_152
DOWN_S_OFF_4 = 6_815_744
DOWN_S_SIZE_4 = 131_072
DOWN_B_OFF_4 = 6_946_816
DOWN_B_SIZE_4 = 131_072

PROJS_4BIT = [
    ("gate", 1024, 4096, GATE_W_OFF_4, GATE_S_OFF_4, GATE_B_OFF_4),
    ("up",   1024, 4096, UP_W_OFF_4,   UP_S_OFF_4,   UP_B_OFF_4),
    ("down", 4096, 1024, DOWN_W_OFF_4, DOWN_S_OFF_4,  DOWN_B_OFF_4),
]

# ============================================================================
# 3-bit expert layout (TurboQuant planar packing)
# ============================================================================

GATE_W_SIZE_3 = 1_572_864   # [1024, 384] uint32  (384 = 4096/32*3)
UP_W_SIZE_3   = 1_572_864
DOWN_W_SIZE_3 = 1_572_864   # [4096, 96]  uint32  (96 = 1024/32*3)

GATE_W_OFF_3 = 0
GATE_S_OFF_3 = GATE_W_OFF_3 + GATE_W_SIZE_3                # 1,572,864
GATE_B_OFF_3 = GATE_S_OFF_3 + GATE_S_SIZE_4                # 1,703,936
UP_W_OFF_3   = GATE_B_OFF_3 + GATE_B_SIZE_4                # 1,835,008
UP_S_OFF_3   = UP_W_OFF_3   + UP_W_SIZE_3                  # 3,407,872
UP_B_OFF_3   = UP_S_OFF_3   + UP_S_SIZE_4                  # 3,538,944
DOWN_W_OFF_3 = UP_B_OFF_3   + UP_B_SIZE_4                  # 3,670,016
DOWN_S_OFF_3 = DOWN_W_OFF_3 + DOWN_W_SIZE_3                # 5,242,880
DOWN_B_OFF_3 = DOWN_S_OFF_3 + DOWN_S_SIZE_4                # 5,373,952
EXPERT_SIZE_3BIT = DOWN_B_OFF_3 + DOWN_B_SIZE_4             # 5,505,024

assert EXPERT_SIZE_3BIT == 5_505_024, f"Got {EXPERT_SIZE_3BIT}"

PROJS_3BIT_OFFSETS = {
    "gate": (GATE_W_OFF_3, GATE_S_OFF_3, GATE_B_OFF_3),
    "up":   (UP_W_OFF_3,   UP_S_OFF_3,   UP_B_OFF_3),
    "down": (DOWN_W_OFF_3, DOWN_S_OFF_3, DOWN_B_OFF_3),
}


# ============================================================================
# bf16 helpers
# ============================================================================

def bf16_to_f32(bf16_u16):
    return (bf16_u16.astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16(f32):
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


# ============================================================================
# Unpack 4-bit
# ============================================================================

def unpack_4bit(packed):
    shape = packed.shape
    flat = packed.ravel()
    n = flat.size
    out = np.empty(n * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = ((flat >> (i * 4)) & 0xF).astype(np.uint8)
    return out.reshape(shape[:-1] + (shape[-1] * 8,))


# ============================================================================
# Pack 3-bit (planar): 32 values -> 3 x uint32
# ============================================================================

def pack_3bit_planar(vals):
    """
    Pack 3-bit values into planar uint32 format.
    Input:  [out_dim, in_dim] uint8, values in [0, 7], in_dim must be multiple of 32
    Output: [out_dim, in_dim/32 * 3] uint32
    """
    shape = vals.shape
    assert shape[-1] % 32 == 0, f"Last dim {shape[-1]} not divisible by 32"
    out_dim = shape[0]
    in_dim = shape[-1]
    n_chunks = in_dim // 32
    packed_cols = n_chunks * 3

    flat = vals.reshape(out_dim, n_chunks, 32).astype(np.uint32)

    # Bit-plane extraction
    plane0 = np.zeros((out_dim, n_chunks), dtype=np.uint32)
    plane1 = np.zeros((out_dim, n_chunks), dtype=np.uint32)
    plane2 = np.zeros((out_dim, n_chunks), dtype=np.uint32)

    for i in range(32):
        plane0 |= ((flat[:, :, i] >> 0) & 1) << i
        plane1 |= ((flat[:, :, i] >> 1) & 1) << i
        plane2 |= ((flat[:, :, i] >> 2) & 1) << i

    # Interleave: [chunk0_p0, chunk0_p1, chunk0_p2, chunk1_p0, ...]
    out = np.empty((out_dim, packed_cols), dtype=np.uint32)
    out[:, 0::3] = plane0
    out[:, 1::3] = plane1
    out[:, 2::3] = plane2

    return out


# ============================================================================
# Fast Walsh-Hadamard Transform (matches Metal kernel)
# ============================================================================

def fwht_inplace(x):
    """In-place Fast Walsh-Hadamard Transform on last axis. Normalizes by 1/sqrt(N)."""
    N = x.shape[-1]
    assert N & (N - 1) == 0, "N must be power of 2"
    h = 1
    while h < N:
        for i in range(0, N, h * 2):
            a = x[..., i:i+h].copy()
            b = x[..., i+h:i+2*h].copy()
            x[..., i:i+h] = a + b
            x[..., i+h:i+2*h] = a - b
        h <<= 1
    x /= np.sqrt(N)
    return x


# ============================================================================
# Requantize one projection: 4-bit -> dequant -> Hadamard -> 3-bit
# ============================================================================

def requantize_projection_3bit(packed_4bit, scales_bf16, biases_bf16, out_dim, in_dim):
    """
    Requantize a single projection from 4-bit to 3-bit with Hadamard rotation.
    Returns: (packed_3bit, new_scales_bf16, new_biases_bf16, rmse)
    """
    num_groups = in_dim // GROUP_SIZE

    # 1. Unpack 4-bit -> float32
    vals_4bit = unpack_4bit(packed_4bit)
    assert vals_4bit.shape == (out_dim, in_dim)

    scales_f32 = bf16_to_f32(scales_bf16)
    biases_f32 = bf16_to_f32(biases_bf16)

    vals_grouped = vals_4bit.reshape(out_dim, num_groups, GROUP_SIZE).astype(np.float32)
    s = scales_f32[:, :, np.newaxis]
    b = biases_f32[:, :, np.newaxis]
    dequant = (vals_grouped * s + b).reshape(out_dim, in_dim)

    # 2. Apply Hadamard rotation to columns (right-multiply by H)
    # W_rot = W @ H  (H is self-inverse, so at inference: y = W_rot @ (H @ x) = W @ x)
    dequant_rot = fwht_inplace(dequant.copy())

    # 3. Compute optimal 3-bit quantization per group (8 levels)
    dequant_rot_grouped = dequant_rot.reshape(out_dim, num_groups, GROUP_SIZE)

    f_min = dequant_rot_grouped.min(axis=2, keepdims=True)
    f_max = dequant_rot_grouped.max(axis=2, keepdims=True)

    s3 = (f_max - f_min) / 7.0
    b3 = f_min

    degenerate = (s3 == 0.0)
    s3_safe = np.where(degenerate, 1.0, s3)

    vals_3bit_f = (dequant_rot_grouped - b3) / s3_safe
    vals_3bit = np.clip(np.round(vals_3bit_f), 0, 7).astype(np.uint8)

    # 4. Compute RMSE (vs rotated dequantized values)
    recon = vals_3bit.astype(np.float32) * s3 + b3
    error = dequant_rot_grouped - recon
    rmse = float(np.sqrt(np.mean(error ** 2)))

    # 5. Pack 3-bit planar
    vals_3bit_flat = vals_3bit.reshape(out_dim, in_dim)
    packed_3bit = pack_3bit_planar(vals_3bit_flat)

    # 6. Convert scales/biases to bf16
    new_scales_bf16 = f32_to_bf16(s3.squeeze(axis=2).astype(np.float32))
    new_biases_bf16 = f32_to_bf16(b3.squeeze(axis=2).astype(np.float32))

    return packed_3bit, new_scales_bf16, new_biases_bf16, rmse


# ============================================================================
# Process one expert
# ============================================================================

def requantize_expert(expert_blob):
    assert len(expert_blob) == EXPERT_SIZE_4BIT

    output = bytearray(EXPERT_SIZE_3BIT)
    proj_rmses = {}

    for name, out_dim, in_dim, w_off, s_off, b_off in PROJS_4BIT:
        packed_cols_4 = in_dim // 8
        num_groups = in_dim // GROUP_SIZE

        w_end = w_off + out_dim * packed_cols_4 * 4
        s_end = s_off + out_dim * num_groups * 2
        b_end = b_off + out_dim * num_groups * 2

        packed_4bit = np.frombuffer(
            expert_blob[w_off:w_end], dtype=np.uint32
        ).reshape(out_dim, packed_cols_4)
        scales_bf16 = np.frombuffer(
            expert_blob[s_off:s_end], dtype=np.uint16
        ).reshape(out_dim, num_groups)
        biases_bf16 = np.frombuffer(
            expert_blob[b_off:b_end], dtype=np.uint16
        ).reshape(out_dim, num_groups)

        packed_3bit, new_scales, new_biases, rmse = requantize_projection_3bit(
            packed_4bit, scales_bf16, biases_bf16, out_dim, in_dim
        )
        proj_rmses[name] = rmse

        w_off_3, s_off_3, b_off_3 = PROJS_3BIT_OFFSETS[name]

        w_data = packed_3bit.tobytes()
        s_data = new_scales.tobytes()
        b_data = new_biases.tobytes()

        output[w_off_3 : w_off_3 + len(w_data)] = w_data
        output[s_off_3 : s_off_3 + len(s_data)] = s_data
        output[b_off_3 : b_off_3 + len(b_data)] = b_data

    return bytes(output), proj_rmses


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TurboQuant: Requantize 4-bit packed experts to 3-bit with Hadamard rotation')
    parser.add_argument('--model', type=str,
                        default=os.path.expanduser(
                            '~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit'
                            '/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3'),
                        help='Path to model directory (containing packed_experts/)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: MODEL/packed_experts_3bit)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Process only this layer (0-59). Default: all layers.')
    parser.add_argument('--verify', action='store_true',
                        help='Verify by comparing reconstruction error')
    parser.add_argument('--experts', type=int, default=NUM_EXPERTS,
                        help=f'Number of experts per layer (default: {NUM_EXPERTS})')
    args = parser.parse_args()

    model_path = Path(args.model)
    input_dir = model_path / 'packed_experts'
    output_dir = Path(args.output) if args.output else model_path / 'packed_experts_3bit'

    if not input_dir.exists():
        # Try relative to script dir
        script_dir = Path(__file__).parent.parent
        input_dir = script_dir / 'packed_experts'
        if not input_dir.exists():
            print(f"ERROR: packed_experts/ not found at {model_path} or {script_dir}", file=sys.stderr)
            sys.exit(1)
        output_dir = script_dir / 'packed_experts_3bit'

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.layer is not None:
        layers = [args.layer]
    else:
        layers = []
        for i in range(60):
            if (input_dir / f'layer_{i:02d}.bin').exists():
                layers.append(i)
        if not layers:
            print(f"ERROR: No layer_XX.bin files found in {input_dir}", file=sys.stderr)
            sys.exit(1)

    num_experts = args.experts

    print(f"Model:       {model_path}")
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"Layers:      {layers}")
    print(f"Experts:     {num_experts}")
    print(f"4-bit size:  {EXPERT_SIZE_4BIT:,} bytes/expert  "
          f"({num_experts * EXPERT_SIZE_4BIT / 1e9:.2f} GB/layer)")
    print(f"3-bit size:  {EXPERT_SIZE_3BIT:,} bytes/expert  "
          f"({num_experts * EXPERT_SIZE_3BIT / 1e9:.2f} GB/layer)")
    print(f"Savings:     {1 - EXPERT_SIZE_3BIT / EXPERT_SIZE_4BIT:.1%}")
    print(f"Method:      TurboQuant (Hadamard rotation + 3-bit affine)")
    print()

    total_t0 = time.time()

    for layer_idx in layers:
        input_path = input_dir / f'layer_{layer_idx:02d}.bin'
        output_path = output_dir / f'layer_{layer_idx:02d}.bin'

        expected_size = num_experts * EXPERT_SIZE_4BIT
        actual_size = input_path.stat().st_size
        if actual_size != expected_size:
            num_experts_actual = actual_size // EXPERT_SIZE_4BIT
            if actual_size % EXPERT_SIZE_4BIT != 0:
                print(f"ERROR: File size not a multiple of EXPERT_SIZE_4BIT, skipping",
                      file=sys.stderr)
                continue
            print(f"  Adjusting to {num_experts_actual} experts based on file size")
        else:
            num_experts_actual = num_experts

        print(f"=== Layer {layer_idx:02d} ({num_experts_actual} experts, "
              f"{actual_size / 1e9:.2f} GB -> "
              f"{num_experts_actual * EXPERT_SIZE_3BIT / 1e9:.2f} GB) ===")

        layer_t0 = time.time()
        rmse_accum = {"gate": 0.0, "up": 0.0, "down": 0.0}

        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            for eidx in range(num_experts_actual):
                fin.seek(eidx * EXPERT_SIZE_4BIT)
                expert_4bit = fin.read(EXPERT_SIZE_4BIT)
                if len(expert_4bit) != EXPERT_SIZE_4BIT:
                    print(f"  ERROR: Short read for expert {eidx}", file=sys.stderr)
                    break

                expert_3bit, proj_rmses = requantize_expert(expert_4bit)
                assert len(expert_3bit) == EXPERT_SIZE_3BIT

                for p in ("gate", "up", "down"):
                    rmse_accum[p] += proj_rmses[p]

                fout.write(expert_3bit)

                if (eidx + 1) % 32 == 0 or eidx == num_experts_actual - 1:
                    elapsed = time.time() - layer_t0
                    rate = (eidx + 1) / elapsed
                    eta = (num_experts_actual - eidx - 1) / rate if rate > 0 else 0
                    print(f"  [{eidx+1:3d}/{num_experts_actual}] "
                          f"{elapsed:.1f}s elapsed, {rate:.1f} experts/s, "
                          f"ETA {eta:.0f}s")

        layer_elapsed = time.time() - layer_t0
        avg_rmse = {p: rmse_accum[p] / num_experts_actual for p in rmse_accum}
        print(f"\n  Layer {layer_idx:02d} done in {layer_elapsed:.1f}s "
              f"({num_experts_actual / layer_elapsed:.1f} experts/s)")
        print(f"  Avg RMSE:  gate={avg_rmse['gate']:.6f}  "
              f"up={avg_rmse['up']:.6f}  down={avg_rmse['down']:.6f}")

        out_size = output_path.stat().st_size
        print(f"  Output: {output_path} ({out_size / 1e9:.2f} GB)")
        print()

    total_elapsed = time.time() - total_t0
    print(f"Total time: {total_elapsed:.1f}s")
    print()
    print("3-bit expert layout offsets (for C/Metal code):")
    print(f"  #define EXPERT_SIZE_3BIT  {EXPERT_SIZE_3BIT}")
    print(f"  #define GATE_W_OFF_3  {GATE_W_OFF_3}")
    print(f"  #define GATE_S_OFF_3  {GATE_S_OFF_3}")
    print(f"  #define GATE_B_OFF_3  {GATE_B_OFF_3}")
    print(f"  #define UP_W_OFF_3    {UP_W_OFF_3}")
    print(f"  #define UP_S_OFF_3    {UP_S_OFF_3}")
    print(f"  #define UP_B_OFF_3    {UP_B_OFF_3}")
    print(f"  #define DOWN_W_OFF_3  {DOWN_W_OFF_3}")
    print(f"  #define DOWN_S_OFF_3  {DOWN_S_OFF_3}")
    print(f"  #define DOWN_B_OFF_3  {DOWN_B_OFF_3}")


if __name__ == '__main__':
    main()
