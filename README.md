# DIPCYT-X40s

## Domain-Independent Scene Text Localization with Multi-Scale Polygon Transformers

DIPCYT-X40s is a complete PyTorch implementation of a DIPCYT-style scene-text localization system. It combines a YOLOv5-inspired feature extractor, partial convolution, Fourier-enhanced attention, Swin Transformer feature aggregation, and multi-scale Transformer polygon heads.

The project is designed for text localization in challenging images containing:

- Arbitrarily oriented text
- Curved or irregular text
- Small text instances
- Occlusion and incomplete visual information
- Low-contrast text
- Complex backgrounds

## Highlights

- **Multi-scale polygon localization** with four Transformer detection heads.
- **Six-point contour prediction** for flexible scene-text shapes.
- **Partial convolution backbone** for robust feature extraction.
- **C3TR transformer blocks** for long-range visual context.
- **FFT-enhanced FPCBAM attention** for high-frequency text and edge information.
- **PANet/FPN feature aggregation** with local-window Swin Transformer blocks.
- **Hungarian matching** with center, size, and GIoU assignment costs.
- **Composite polygon loss** combining coordinates, GIoU, focal confidence, and shape regularization.
- **ImageNet-compatible 640 x 640 preprocessing** with polygon-aware augmentation.
- **Mixed-precision training** with gradient scaling and clipping.
- **Automatic VRAM probing** and hardware-aware batch-size selection.
- **Checkpoint resume support** for long-running experiments.
- **Optuna hyperparameter search** with TPE sampling and Hyperband pruning.
- **Polygon IoU, instance F1, raster area metrics, and confidence sweeps**.
- **Modular command-line workflow** for training, evaluation, and experimentation.

## Architecture

```text
Input image (640 x 640)
        |
        v
TPH-YOLOv5++-style backbone
  - Focus layer
  - Partial convolution
  - CSP bottlenecks
  - SPPF
  - C3TR transformer blocks
        |
        v
FPCBAM-enhanced PANet/FPN
  - Fourier high-frequency features
  - Channel and spatial attention
  - Optional position attention
  - Swin Transformer windows
        |
        v
Four Transformer polygon heads
  - Learnable queries
  - 2D positional encoding
  - Six-point polygon regression
  - Text confidence prediction
        |
        v
Polygon detections and confidence scores
```

## Project Structure

```text
.
├── pipeline/
│   ├── model.py              # Backbone, FPCBAM, PANet, Swin, polygon heads
│   ├── dataset.py            # COCO-Text loading, preprocessing, augmentation
│   ├── losses.py             # Hungarian matching and composite polygon loss
│   ├── eval.py               # Polygon IoU, F1, area metrics, threshold sweeps
│   ├── train.py              # Mixed-precision production training
│   ├── config.py             # Dataclass-based configuration
│   ├── pretrained.py         # YOLOv5-compatible weight transfer
│   └── optuna_search.py      # Hyperparameter optimization
├── run.py                    # Unified CLI
├── default_config.yaml       # Baseline configuration
├── production_config.yaml    # Tuned production configuration
├── extended_config.yaml      # Extended training configuration
├── build_production_config.py
├── run_optuna.py
├── run_progressive_test.py
├── inspect_sample_blocks.py
├── test_loader_step.py
├── coco-dipcyt-x40s (1).ipynb # Original experimental notebook
└── block_outputs_inspection/  # Saved feature and detection visualizations
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for your CUDA version from the official PyTorch installation selector.

## Data Setup

The pipeline uses COCO-Text v2 annotations and COCO 2014 images.

Place the following files/directories in the project directory:

```text
train2014/
cocotext.v2.json
```

The annotation loader creates a compact cache automatically:

```text
cocotext_compact.pkl
```

Large datasets, archives, checkpoints, and generated logs are intentionally excluded from Git tracking through `.gitignore`.

## Quick Start

### Train

```bash
python run.py train --config production_config.yaml
```

### Resume from a checkpoint

```bash
python run.py train ^
  --config extended_config.yaml ^
  --resume checkpoints\best_model.pth
```

### Evaluate a checkpoint

```bash
python run.py eval ^
  --checkpoint checkpoints\best_model.pth ^
  --config production_config.yaml ^
  --split val ^
  --conf-thresh 0.3
```

### Sweep confidence thresholds

```bash
python run.py sweep ^
  --checkpoint checkpoints\best_model.pth ^
  --config production_config.yaml ^
  --split val
```

### Run hyperparameter optimization

```bash
python run.py optuna ^
  --config default_config.yaml ^
  --n-trials 30 ^
  --pruner hyperband ^
  --out-dir optuna_results
```

## Data Processing

The data pipeline provides:

- Legible-text annotation filtering
- Compact annotation caching
- Polygon and mask loading
- Bounding-box fallback
- Uniform arc-length contour resampling
- Six-point polygon representation
- Clockwise polygon canonicalization
- Deterministic target padding
- Valid-target masks
- Minimum-size filtering
- ImageNet normalization
- Polygon-aware geometric augmentation

The default augmentation policy includes:

```text
Scale range: 0.7 to 1.3
Horizontal flip probability: 0.5
Rotation range: +/-10 degrees
Color jitter probability: 0.5
Gaussian blur probability: 0.2
```

## Training Features

The production training loop includes:

- Automatic GPU memory probing
- Safe micro-batch selection
- Gradient accumulation
- CUDA mixed precision
- Gradient scaling
- Gradient clipping
- Backbone freeze/unfreeze scheduling
- Cosine learning-rate scheduling
- NaN/Inf protection
- CUDA OOM recovery
- Best-model checkpointing
- Atomic checkpoint persistence
- Optimizer and scheduler state restoration
- Persistent training logs
- Validation after every epoch

The recorded production configuration used:

```text
Input size: 640 x 640
Optimizer: AdamW
Scheduler: cosine
AMP: enabled
Effective batch size: 6
Backbone unfreeze epoch: 2
Polygon points: 6
Queries per head: 32
Detection heads: 4
```

## Evaluation

The project supports two complementary metric families.

### Instance-level polygon metrics

- Polygon IoU
- Precision
- Recall
- F1 score
- True positives
- False positives
- False negatives

### Raster area-overlap metrics

```text
Area precision
Area recall
Area F1
```

The evaluator includes robust polygon handling using Shapely, including invalid-polygon repair and degenerate-shape protection.

## Experiment Results

The completed 30-epoch production run demonstrates successful end-to-end optimization on COCO-Text.

### Training progression

| Metric | Initial recorded value | Best/final recorded value |
|---|---:|---:|
| Training loss | 2.0549 | 0.8891 best epoch |
| Validation recall | 0.0128 | 0.3628 |
| Validation instance F1 | 0.0010 | 0.0791 |
| Validation area F1 | 0.0381 | 0.2907 |

The best checkpoint was saved at:

```text
checkpoints/best_model.pth
```

The complete epoch-level training history is available in:

```text
production_training.log
```

## Reproducibility

The project exposes model, data, loss, matcher, hardware, and training settings through YAML configuration files.

Important reproducibility controls include:

- Fixed random seed
- Deterministic dataset split
- Explicit optimizer and scheduler settings
- Explicit augmentation parameters
- Saved checkpoint state
- Persistent epoch logs
- Exported hyperparameter configurations

## Reference

This implementation is inspired by:

> Ayush Roy, Shivakumara Palaiahnakote, Umapada Pal, and Cheng-Lin Liu,  
> **A Novel Domain Independent Scene Text Localizer**, Pattern Recognition, 2024.

The local reference PDF is used for research traceability and is excluded from Git tracking by default.

## License

Add the license appropriate for your intended use before publishing the repository.
