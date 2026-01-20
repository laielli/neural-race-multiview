#!/bin/bash
# CIFAR-100 Hierarchical KD Experiment
# Validates Finding 2 on real data
#
# Usage:
#   ./run_cifar_experiment.sh          # Full experiment
#   ./run_cifar_experiment.sh --quick  # Quick test
#
# Requirements: GPU with CUDA or MPS

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_BASE="../log/results/cifar"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"

# Detect device
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    echo "Using CUDA GPU"
elif python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    DEVICE="mps"
    echo "Using Apple MPS"
else
    DEVICE="cpu"
    echo "WARNING: No GPU detected, using CPU (will be slow)"
fi

# Check for quick mode
QUICK_FLAG=""
if [[ "$1" == "--quick" ]]; then
    QUICK_FLAG="--quick"
    echo "Running in QUICK mode (reduced epochs/seeds)"
fi

echo "=============================================="
echo "CIFAR-100 Hierarchical KD Experiment"
echo "=============================================="
echo "Device: $DEVICE"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run main experiment
echo ""
echo "Starting main experiment..."
python exp_cifar_hierarchical.py \
    --device "$DEVICE" \
    --teacher-epochs 200 \
    --student-epochs 200 \
    --n-teachers 3 \
    --n-seeds 3 \
    --temperature 4.0 \
    --alpha 0.7 \
    --batch-size 128 \
    --output "$OUTPUT_DIR" \
    $QUICK_FLAG \
    2>&1 | tee "${OUTPUT_DIR}/experiment.log"

echo ""
echo "=============================================="
echo "Experiment complete!"
echo "Results: $OUTPUT_DIR"
echo "=============================================="

# Display summary if results exist
if [[ -f "${OUTPUT_DIR}/cifar_hierarchical_results.json" ]]; then
    echo ""
    echo "Key results:"
    python -c "
import json
with open('${OUTPUT_DIR}/cifar_hierarchical_results.json') as f:
    r = json.load(f)
print(f\"  Hard Labels: {r['hard_labels']['avg_accuracy']:.1f}% +/- {r['hard_labels']['std_accuracy']:.1f}%\")
print(f\"  KD:          {r['kd']['avg_accuracy']:.1f}% +/- {r['kd']['std_accuracy']:.1f}%\")
if r['speed_comparison'].get('80', {}).get('speedup'):
    print(f\"  Speedup at 80%: {r['speed_comparison']['80']['speedup']:.2f}x\")
print(f\"  Success: {r['success']}\")
"
fi
