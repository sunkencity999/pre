#!/usr/bin/env python3
"""
repack_experts_rot4bit.py — TurboQuant rotated 4-bit repacking.

Reads packed_experts/layer_XX.bin files, applies Hadamard rotation to weight
columns, and requantizes at 4-bit. Output format is identical to the original
4-bit format (same EXPERT_SIZE, same layout), so existing kernels work unchanged.

The rotation distributes outlier weight values more uniformly across channels,
reducing quantization error at the same bit rate. At inference, the Hadamard
transform is applied to the input vector before the matvec (H is self-inverse).

Usage:
    python repack_experts_rot4bit.py [--layer N] [--verify]
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path


# ============================================================================
# 4-bit expert layout
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
# Helpers
# ============================================================================

def bf16_to_f32(bf16_u16):
    return (bf16_u16.astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16(f32):
    return (f32.view(np.uint32) >> 16).astype(np.uint16)

def unpack_4bit(packed):
    shape = packed.shape
    flat = packed.ravel()
    n = flat.size
    out = np.empty(n * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = ((flat >> (i * 4)) & 0xF).astype(np.uint8)
    return out.reshape(shape[:-1] + (shape[-1] * 8,))

def pack_4bit(vals):
    """Pack 4-bit values into uint32 (8 values per word, LSB first)."""
    shape = vals.shape
    assert shape[-1] % 8 == 0
    n_packed = shape[-1] // 8
    flat = vals.reshape(-1, shape[-1])
    rows = flat.shape[0]
    out = np.zeros((rows, n_packed), dtype=np.uint32)
    for i in range(8):
        out |= flat[:, i::8].astype(np.uint32) << (i * 4)
    return out.reshape(shape[:-1] + (n_packed,))


def fwht_inplace(x):
    """In-place Fast Walsh-Hadamard Transform on last axis, normalized."""
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
# Requantize one projection: 4-bit -> dequant -> Hadamard -> 4-bit
# ============================================================================

def requantize_projection_rot4bit(packed_4bit, scales_bf16, biases_bf16, out_dim, in_dim):
    num_groups = in_dim // GROUP_SIZE

    # 1. Unpack and dequantize
    vals_4bit = unpack_4bit(packed_4bit)
    scales_f32 = bf16_to_f32(scales_bf16)
    biases_f32 = bf16_to_f32(biases_bf16)

    vals_grouped = vals_4bit.reshape(out_dim, num_groups, GROUP_SIZE).astype(np.float32)
    s = scales_f32[:, :, np.newaxis]
    b = biases_f32[:, :, np.newaxis]
    dequant = (vals_grouped * s + b).reshape(out_dim, in_dim)

    # 2. Apply Hadamard rotation to columns
    dequant_rot = fwht_inplace(dequant.copy())

    # 3. Re-quantize to 4-bit (16 levels)
    dequant_rot_grouped = dequant_rot.reshape(out_dim, num_groups, GROUP_SIZE)
    f_min = dequant_rot_grouped.min(axis=2, keepdims=True)
    f_max = dequant_rot_grouped.max(axis=2, keepdims=True)

    s4 = (f_max - f_min) / 15.0
    b4 = f_min

    degenerate = (s4 == 0.0)
    s4_safe = np.where(degenerate, 1.0, s4)

    vals_4bit_f = (dequant_rot_grouped - b4) / s4_safe
    vals_4bit_new = np.clip(np.round(vals_4bit_f), 0, 15).astype(np.uint8)

    # 4. RMSE
    recon = vals_4bit_new.astype(np.float32) * s4 + b4
    error = dequant_rot_grouped - recon
    rmse = float(np.sqrt(np.mean(error ** 2)))

    # 5. Pack back to 4-bit
    vals_4bit_flat = vals_4bit_new.reshape(out_dim, in_dim)
    packed_4bit_new = pack_4bit(vals_4bit_flat)

    new_scales_bf16 = f32_to_bf16(s4.squeeze(axis=2).astype(np.float32))
    new_biases_bf16 = f32_to_bf16(b4.squeeze(axis=2).astype(np.float32))

    return packed_4bit_new, new_scales_bf16, new_biases_bf16, rmse


# ============================================================================
# Process one expert
# ============================================================================

def requantize_expert(expert_blob):
    assert len(expert_blob) == EXPERT_SIZE_4BIT
    output = bytearray(EXPERT_SIZE_4BIT)
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

        packed_new, new_scales, new_biases, rmse = requantize_projection_rot4bit(
            packed_4bit, scales_bf16, biases_bf16, out_dim, in_dim
        )
        proj_rmses[name] = rmse

        w_data = packed_new.tobytes()
        s_data = new_scales.tobytes()
        b_data = new_biases.tobytes()

        output[w_off : w_off + len(w_data)] = w_data
        output[s_off : s_off + len(s_data)] = s_data
        output[b_off : b_off + len(b_data)] = b_data

    return bytes(output), proj_rmses


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TurboQuant: Repack 4-bit experts with Hadamard rotation (same format, better quality)')
    parser.add_argument('--model', type=str,
                        default=os.path.expanduser(
                            '~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit'
                            '/snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3'),
                        help='Path to model directory (containing packed_experts/)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: MODEL/packed_experts_rot)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Process only this layer (0-59). Default: all layers.')
    parser.add_argument('--experts', type=int, default=NUM_EXPERTS,
                        help=f'Number of experts per layer (default: {NUM_EXPERTS})')
    args = parser.parse_args()

    model_path = Path(args.model)
    input_dir = model_path / 'packed_experts'
    output_dir = Path(args.output) if args.output else model_path / 'packed_experts_rot'

    if not input_dir.exists():
        script_dir = Path(__file__).parent.parent
        input_dir = script_dir / 'packed_experts'
        if not input_dir.exists():
            print(f"ERROR: packed_experts/ not found", file=sys.stderr)
            sys.exit(1)
        output_dir = script_dir / 'packed_experts_rot'

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
    print(f"Format:      4-bit (same size, rotated weights for lower error)")
    print(f"Method:      TurboQuant Hadamard rotation + 4-bit affine")
    print()

    total_t0 = time.time()

    for layer_idx in layers:
        input_path = input_dir / f'layer_{layer_idx:02d}.bin'
        output_path = output_dir / f'layer_{layer_idx:02d}.bin'

        actual_size = input_path.stat().st_size
        num_experts_actual = actual_size // EXPERT_SIZE_4BIT

        print(f"=== Layer {layer_idx:02d} ({num_experts_actual} experts) ===")
        layer_t0 = time.time()
        rmse_accum = {"gate": 0.0, "up": 0.0, "down": 0.0}

        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            for eidx in range(num_experts_actual):
                fin.seek(eidx * EXPERT_SIZE_4BIT)
                expert_4bit = fin.read(EXPERT_SIZE_4BIT)
                if len(expert_4bit) != EXPERT_SIZE_4BIT:
                    break

                expert_rot, proj_rmses = requantize_expert(expert_4bit)
                for p in ("gate", "up", "down"):
                    rmse_accum[p] += proj_rmses[p]
                fout.write(expert_rot)

                if (eidx + 1) % 32 == 0 or eidx == num_experts_actual - 1:
                    elapsed = time.time() - layer_t0
                    rate = (eidx + 1) / elapsed
                    eta = (num_experts_actual - eidx - 1) / rate if rate > 0 else 0
                    print(f"  [{eidx+1:3d}/{num_experts_actual}] "
                          f"{elapsed:.1f}s, {rate:.1f} exp/s, ETA {eta:.0f}s")

        layer_elapsed = time.time() - layer_t0
        avg_rmse = {p: rmse_accum[p] / num_experts_actual for p in rmse_accum}
        print(f"  Done in {layer_elapsed:.1f}s — RMSE: gate={avg_rmse['gate']:.6f} "
              f"up={avg_rmse['up']:.6f} down={avg_rmse['down']:.6f}")
        print()

    total_elapsed = time.time() - total_t0
    print(f"Total time: {total_elapsed:.1f}s")
    print("Output uses same 4-bit format — use with --rotated flag at inference.")


if __name__ == '__main__':
    main()
