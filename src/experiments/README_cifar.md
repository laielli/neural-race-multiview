# CIFAR-100 Hierarchical KD Experiment

This experiment validates Finding 2 (hierarchical structure enables acceleration) on real data.

## Overview

- **Teacher**: ResNet-18 trained on CIFAR-100 (100 fine classes)
- **Student**: ResNet-18 trained on CIFAR-100 superclasses (20 coarse classes)
- **Comparison**: Hard labels vs KD with collapsed soft labels

## Requirements

```bash
pip install torch torchvision numpy matplotlib tqdm Pillow
```

GPU recommended (CUDA or MPS).

## Quick Test

```bash
# ~30 min on GPU, verifies everything works
python experiments/exp_cifar_hierarchical.py --quick --device cuda
```

## Full Experiment (for paper)

```bash
# ~4-6 hours on GPU
python experiments/exp_cifar_hierarchical.py \
    --device cuda \
    --teacher-epochs 200 \
    --student-epochs 200 \
    --n-teachers 3 \
    --n-seeds 3 \
    --temperature 4.0 \
    --alpha 0.7 \
    --output ../log/results/cifar
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--teacher-epochs` | 200 | Epochs to train teacher on 100-class task |
| `--student-epochs` | 200 | Epochs to train student on 20-class task |
| `--n-teachers` | 3 | Number of teachers in ensemble |
| `--n-seeds` | 3 | Number of random seeds for student |
| `--temperature` | 4.0 | KD softmax temperature |
| `--alpha` | 0.7 | Weight for soft labels (1-alpha for hard) |
| `--lr` | 0.1 | Learning rate |
| `--batch-size` | 128 | Batch size |
| `--device` | auto | cuda, mps, or cpu |
| `--quick` | false | Quick test mode (reduced epochs/seeds) |

## Expected Results

Based on synthetic experiments (Finding 2):
- KD should reach 80% accuracy faster than hard labels
- Expected speedup: 2-5x
- Soft labels should encode same-superclass confusion (~15-20%)

## Output

Results saved to `--output` directory:
- `cifar_hierarchical_results.json` - Full metrics
- `figures/cifar_hierarchical_results.png` - Learning curves and speedup comparison

## Ablations (Optional)

After main experiment, run ablations:

```bash
# Temperature sweep
for T in 1 2 4 8 16; do
    python experiments/exp_cifar_hierarchical.py \
        --device cuda --temperature $T \
        --output ../log/results/cifar_temp_$T
done

# Alpha sweep
for A in 0.3 0.5 0.7 0.9 1.0; do
    python experiments/exp_cifar_hierarchical.py \
        --device cuda --alpha $A \
        --output ../log/results/cifar_alpha_$A
done
```
