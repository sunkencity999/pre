#!/bin/bash
# benchmark.sh — Compare 4-bit, 3-bit TurboQuant, and rotated 4-bit inference
#
# Runs the same prompts through each quantization path and reports:
#   - Tokens/second (sustained)
#   - Time to first token (TTFT)
#   - Output quality (human review)
#
# Since the engine auto-detects expert format by directory presence
# (3-bit > rot4bit > 2-bit > 4-bit), we temporarily hide directories
# to force each configuration.

set -euo pipefail

ENGINE="$(dirname "$0")/engine/infer"
RESULTS_DIR="$(dirname "$0")/benchmark_results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="$RESULTS_DIR/benchmark_${TIMESTAMP}.txt"

# Number of tokens to generate per test
TOKENS=200

# Model path
MODEL_PATH=$(grep -o '"model_path": "[^"]*"' "$(dirname "$0")/engine/expert_index.json" | cut -d'"' -f4)

# Test prompts covering different capability areas
declare -a PROMPTS=(
    "Explain the difference between TCP and UDP in networking. Be thorough."
    "Write a Python function that finds the two numbers in a list that add up to a target sum. Include error handling."
    "What is 17 * 23? Show your work step by step."
    "Write a JSON object representing a person with name, age, address (street, city, state, zip), and a list of 3 hobbies."
)
declare -a PROMPT_LABELS=(
    "reasoning"
    "code_generation"
    "math"
    "json_structured"
)

# Hide/unhide expert directories to force specific quantization paths
hide_dirs() {
    # Hide 3-bit and rotated dirs so auto-detect falls through to plain 4-bit
    for dir in packed_experts_3bit packed_experts_rot packed_experts_2bit; do
        if [ -d "$MODEL_PATH/$dir" ]; then
            mv "$MODEL_PATH/$dir" "$MODEL_PATH/.hidden_${dir}"
        fi
    done
}

unhide_dirs() {
    # Restore all hidden directories
    for dir in packed_experts_3bit packed_experts_rot packed_experts_2bit; do
        if [ -d "$MODEL_PATH/.hidden_${dir}" ]; then
            mv "$MODEL_PATH/.hidden_${dir}" "$MODEL_PATH/$dir"
        fi
    done
}

setup_4bit() {
    hide_dirs
}

setup_3bit() {
    unhide_dirs
    # Hide rot so auto-detect picks 3-bit (which has highest priority)
    if [ -d "$MODEL_PATH/packed_experts_rot" ]; then
        mv "$MODEL_PATH/packed_experts_rot" "$MODEL_PATH/.hidden_packed_experts_rot"
    fi
}

setup_rot4bit() {
    unhide_dirs
    # Hide 3-bit so auto-detect picks rotated 4-bit
    if [ -d "$MODEL_PATH/packed_experts_3bit" ]; then
        mv "$MODEL_PATH/packed_experts_3bit" "$MODEL_PATH/.hidden_packed_experts_3bit"
    fi
}

# Ensure cleanup on exit
trap 'unhide_dirs; echo "Restored all expert directories"' EXIT

# Verify all expert directories exist
for dir in packed_experts packed_experts_3bit packed_experts_rot; do
    if [ ! -d "$MODEL_PATH/$dir" ]; then
        echo "ERROR: $MODEL_PATH/$dir not found"
        exit 1
    fi
done

echo "================================================================" | tee "$REPORT"
echo "PRE Quantization Benchmark — $(date)" | tee -a "$REPORT"
echo "Machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')" | tee -a "$REPORT"
echo "Memory:  $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB", $1/1024/1024/1024}')" | tee -a "$REPORT"
echo "Tokens per test: $TOKENS" | tee -a "$REPORT"
echo "Model path: $MODEL_PATH" | tee -a "$REPORT"
echo "================================================================" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"

# Configurations: setup function + label
declare -a CONFIG_SETUPS=(
    "setup_4bit"
    "setup_3bit"
    "setup_rot4bit"
)
declare -a CONFIG_LABELS=(
    "4bit_baseline"
    "3bit_turboquant"
    "rot4bit_turboquant"
)

# Summary table header
SUMMARY="$RESULTS_DIR/summary_${TIMESTAMP}.tsv"
printf "config\tprompt\ttok_s\tttft_ms\ttotal_s\n" > "$SUMMARY"

for ci in "${!CONFIG_SETUPS[@]}"; do
    setup_fn="${CONFIG_SETUPS[$ci]}"
    config_label="${CONFIG_LABELS[$ci]}"

    echo "================================================================" | tee -a "$REPORT"
    echo "Configuration: $config_label" | tee -a "$REPORT"
    echo "================================================================" | tee -a "$REPORT"

    # Setup the right directories
    $setup_fn

    for pi in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$pi]}"
        prompt_label="${PROMPT_LABELS[$pi]}"
        outfile="$RESULTS_DIR/${config_label}_${prompt_label}_${TIMESTAMP}.txt"

        echo "" | tee -a "$REPORT"
        echo "--- $prompt_label ---" | tee -a "$REPORT"
        echo "Prompt: ${prompt:0:60}..." | tee -a "$REPORT"

        # Run inference with timing, capture full output
        "$ENGINE" \
            --prompt "$prompt" \
            --tokens "$TOKENS" \
            --timing \
            2>&1 | tee "$outfile"

        # Extract key metrics from output
        tok_s=$(grep -oE '[0-9]+\.[0-9]+ tok/s' "$outfile" | tail -1 | grep -oE '[0-9]+\.[0-9]+' || echo "N/A")
        ttft=$(grep -oE 'TTFT: *[0-9]+' "$outfile" | grep -oE '[0-9]+' || echo "N/A")
        total_s=$(grep -oE 'Total time: *[0-9]+\.[0-9]+' "$outfile" | grep -oE '[0-9]+\.[0-9]+' || echo "N/A")

        echo "  tok/s: $tok_s" | tee -a "$REPORT"
        echo "  TTFT:  ${ttft} ms" | tee -a "$REPORT"
        echo "  Total: ${total_s} s" | tee -a "$REPORT"
        echo "  Output saved: $outfile" | tee -a "$REPORT"

        # Write to summary TSV
        printf "%s\t%s\t%s\t%s\t%s\n" "$config_label" "$prompt_label" "$tok_s" "$ttft" "$total_s" >> "$SUMMARY"

        # Brief cooldown between runs to let SSD cache normalize
        sleep 5
    done

    echo "" | tee -a "$REPORT"
    # Longer cooldown between configs to flush page cache state
    echo "Cooldown before next config (30s to normalize page cache)..." | tee -a "$REPORT"
    sleep 30
done

# Restore directories
unhide_dirs

echo "" | tee -a "$REPORT"
echo "================================================================" | tee -a "$REPORT"
echo "BENCHMARK COMPLETE" | tee -a "$REPORT"
echo "================================================================" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"

# Print summary table
echo "Summary:" | tee -a "$REPORT"
echo "" | tee -a "$REPORT"
column -t -s $'\t' "$SUMMARY" | tee -a "$REPORT"

echo "" | tee -a "$REPORT"
echo "Full report: $REPORT" | tee -a "$REPORT"
echo "Summary TSV: $SUMMARY" | tee -a "$REPORT"
echo "Individual outputs: $RESULTS_DIR/" | tee -a "$REPORT"
