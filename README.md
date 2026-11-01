# DIPCYT-X40s

## Domain-Independent Scene Text Localization with Multi-Scale Polygon Transformers

DIPCYT-X40s is a complete, modular PyTorch pipeline for developing a DIPCYT-style scene-text localization system. The project combines convolutional feature extraction, partial convolution, Fourier-enhanced attention, Swin Transformer feature aggregation, and Transformer-based polygon detection heads.

The repository provides the complete software pipeline and research infrastructure. It is not presented as a fully completed or final reproduction of every experimental result from the reference paper. The implementation is intended as a strong foundation for continued model development, training, evaluation, and ablation studies.

The goal is to localize text directly as flexible polygons instead of restricting every text instance to an axis-aligned rectangle. This makes the system suitable for difficult visual conditions such as rotated text, curved text, small text, cluttered backgrounds, occlusion, and low-contrast scenes.

The repository includes:

- A modular model implementation
- COCO-Text v2 data loading and preprocessing
- Polygon-aware data augmentation
- Multi-scale Hungarian matching
- Composite polygon losses
- Mixed-precision training
- Automatic GPU memory probing
- Checkpointing and resume support
- Optuna hyperparameter optimization
- Polygon IoU and area-overlap evaluation
- A unified command-line interface
- An original experimental notebook

---

## Why Polygon-Based Text Localization?

Text in natural images is rarely perfectly horizontal or rectangular. A word may be:

- Rotated by an arbitrary angle
- Curved along an object or sign
- Perspective-distorted
- Partially occluded
- Very small compared with the image
- Embedded in a visually complex background

Rectangular detection can include a large amount of background around such text. Polygon localization provides a more expressive representation by predicting points along the text boundary.

DIPCYT-X40s represents each scene-text instance with six contour points. The points are normalized to the image coordinate system and predicted by Transformer detection heads at multiple spatial resolutions.

---

## Core Contributions

### 1. Multi-scale polygon detection

The model uses four detection heads operating on different feature resolutions. Each head contains learnable Transformer queries and predicts:

- Text confidence
- Six polygon vertices
- Reference centers used during matching

The multi-scale design allows the system to handle text instances with different sizes in the same image.

### 2. Partial convolution feature extraction

Partial convolution is integrated into the YOLOv5-inspired backbone. The layer uses valid-region information while extracting features, making it well suited to scenes containing:

- Occlusion
- Missing visual information
- Degraded regions
- Low-quality imagery

### 3. Fourier-enhanced attention

The FPCBAM implementation uses Fourier features to emphasize high-frequency information such as:

- Character strokes
- Text boundaries
- Fine edge structure
- Small text details

The Fourier path is combined with channel attention, spatial attention, optional position attention, and a gated residual connection.

### 4. Transformer-based contextual reasoning

Transformer blocks are used in both the backbone and feature pyramid:

- C3TR blocks capture long-range feature relationships.
- Swin Transformer windows provide efficient local self-attention.
- Transformer polygon heads convert multi-scale features into detections.

### 5. Production-oriented training

The training system is designed for reliable, long-running experiments. It includes:

- Automatic VRAM probing
- Safe micro-batch sizing
- Mixed-precision training
- Gradient scaling
- Gradient clipping
- OOM recovery
- Checkpoint retention
- Resume-from-checkpoint support
- Validation logging

---

## Model Architecture

```text
Input image
  640 x 640 RGB image
        |
        v
TPH-YOLOv5++-style backbone
  Focus
  Partial convolution
  CSP bottlenecks
  SPPF
  C3TR transformer blocks
        |
        v
FPCBAM-enhanced PANet/FPN
  Multi-scale lateral fusion
  Fourier features
  Channel attention
  Spatial attention
  Optional position attention
  Swin Transformer windows
        |
        v
Four Transformer polygon heads
  Learnable queries
  Two-dimensional positional encoding
  Transformer encoder
  Transformer decoder
  Anchor-relative polygon regression
  Text confidence prediction
        |
        v
Polygon detections
  Six vertices per text instance
  Confidence score per query
```

### Architecture flow

1. The input image is resized and normalized.
2. The backbone extracts hierarchical visual features.
3. Partial convolution improves feature extraction in incomplete or difficult regions.
4. C3TR blocks add long-range contextual reasoning.
5. PANet/FPN combines features from multiple resolutions.
6. FPCBAM emphasizes informative channels, positions, and high-frequency details.
7. Swin Transformer blocks model local spatial relationships efficiently.
8. Four polygon heads predict text instances at different scales.
9. Hungarian matching associates predictions with ground-truth polygons during training.

---

## Backbone

The backbone is implemented in `pipeline/model.py` and follows a YOLOv5-inspired design with Transformer enhancements.

### Focus layer

The Focus layer performs space-to-depth rearrangement. It reduces spatial resolution while increasing the channel representation, allowing the network to process local neighborhoods efficiently.

### Partial convolution blocks

Each partial convolution block:

1. Receives image features and an optional validity mask.
2. Computes a convolution over valid feature regions.
3. Normalizes the response according to the active mask.
4. Applies GroupNorm and SiLU activation.

When no explicit mask is supplied, the implementation uses an all-valid mask, so the layer remains compatible with ordinary image inputs.

### CSP feature blocks

CSP bottlenecks split the feature stream into transformed and shortcut paths. The two paths are then concatenated and projected. This preserves efficient feature reuse while increasing representational depth.

### SPPF

The SPPF module uses repeated max pooling to provide multiple receptive-field sizes. It helps the network combine local text information with broader scene context.

### C3TR

C3TR replaces part of the conventional convolutional bottleneck with Transformer encoder layers:

1. Spatial features are converted into tokens.
2. Multi-head self-attention models relationships between locations.
3. Feed-forward layers transform the token features.
4. The tokens are restored to a spatial feature map.
5. Transformer and convolutional paths are fused.

---

## FPCBAM Attention

FPCBAM is implemented as a Fourier-augmented attention module.

### Fourier branch

The Fourier branch:

1. Converts features to a stable floating-point representation.
2. Computes a two-dimensional FFT.
3. Removes the DC component.
4. Applies the inverse FFT.
5. Combines the high-frequency result with the original feature.

This exposes high-frequency structure that can help distinguish text strokes and boundaries from smooth backgrounds.

### Channel attention

Channel attention uses both global average pooling and global maximum pooling. The pooled descriptors pass through a lightweight MLP to generate learned channel gates.

### Spatial attention

Spatial attention combines channel-wise mean and maximum maps. A dilated convolution then produces a spatial attention mask that highlights useful image locations.

### Position attention

The optional position-attention module models pairwise relationships between spatial locations. It is selectively enabled by configuration to balance contextual modeling and memory usage.

### Gated residual fusion

FPCBAM uses a learnable residual gate initialized at zero. This starts the module close to an identity mapping and allows the attention contribution to grow smoothly during training.

---

## PANet and Swin Transformer Feature Pyramid

The feature pyramid combines top-down and bottom-up paths:

- Top-down fusion transfers semantic information to higher-resolution features.
- Bottom-up aggregation strengthens localization information.
- Lateral projections align channel dimensions.
- FPCBAM modules refine the fused features.
- Swin blocks provide local self-attention.

### Swin windows

The Swin implementation uses:

- Local attention windows
- Relative positional biases
- Regular windows
- Shifted windows
- Layer normalization
- MLP transformations

Regular and shifted windows allow neighboring regions to exchange information without requiring full global attention at every layer.

---

## Transformer Polygon Heads

Each detection head is a DETR-style Transformer module.

The production configuration uses:

```text
Detection heads: 4
Queries per head: 32
Total queries: 128
Polygon points: 6
Decoder layers: 6
Attention heads: 8
Feed-forward dimension: 1024
```

### Query embeddings

Each head contains learnable query embeddings. A query represents a possible text instance and attends to the encoded feature map.

### Two-dimensional positional encoding

The feature map receives sinusoidal positional information before being passed into the Transformer encoder. This preserves the spatial meaning of each token.

### Reference-point polygon regression

Each query contains learnable reference polygon points. The model predicts a bounded offset from those references:

```text
predicted point = reference point + learned offset
```

The result is clamped to normalized image coordinates.

### Confidence prediction

The confidence head predicts whether each query corresponds to text. The initial confidence bias is negative, allowing the model to begin with conservative predictions while the localization heads learn.

---

## Data Pipeline

The data pipeline is implemented in `pipeline/dataset.py`.

The default dataset is COCO-Text v2 paired with COCO 2014 images.

### Annotation processing

The loader:

- Reads COCO-Text annotations.
- Keeps legible text annotations.
- Builds a compact annotation cache.
- Loads polygons or masks when available.
- Falls back to bounding boxes when polygon data is unavailable.
- Converts annotations to six-point contours.
- Normalizes coordinates to `[0, 1]`.

### Polygon canonicalization

Equivalent polygons can have different point orderings. To make matching and coordinate loss more stable, the implementation:

- Enforces a consistent winding direction.
- Selects a deterministic starting vertex.
- Rotates the contour sequence into canonical order.

### Arc-length resampling

Polygon contours are uniformly resampled along their perimeter. This distributes the six points around the text boundary rather than selecting arbitrary annotation vertices.

### Batch collation

Images can contain different numbers of text instances. The data loader:

- Pads target polygons to a configured maximum.
- Creates a boolean valid-target mask.
- Excludes padded targets from loss and evaluation.

### Preprocessing

The default image preprocessing is:

```text
Input resolution: 640 x 640
Color format: RGB
Interpolation: bilinear
Normalization: ImageNet mean and standard deviation
```

### Augmentation

The training pipeline supports:

- Horizontal flipping
- Random scaling and cropping
- Rotation
- HSV color jitter
- Gaussian blur

Default values:

```text
Scale range: 0.7 to 1.3
Horizontal flip probability: 0.5
Rotation range: +/-10 degrees
Color jitter probability: 0.5
Gaussian blur probability: 0.2
Rotation probability: 0.4
```

---

## Hungarian Matching and Loss

The training objective is implemented in `pipeline/losses.py`.

The composite loss is:

```text
L = lambda_coord * L_coord
  + lambda_giou * L_giou
  + lambda_conf * L_focal
  + lambda_shape * L_shape
```

### Coordinate loss

The coordinate term measures the absolute difference between matched predicted and ground-truth polygon points.

### GIoU loss

The GIoU term uses polygon-derived bounding boxes to provide a spatial-overlap signal in addition to point-wise coordinate supervision.

### Focal confidence loss

The focal confidence term handles the imbalance between positive text queries and the much larger number of negative queries. Positive and negative contributions are normalized independently.

### Shape regularization

The shape term encourages stable polygon geometry by discouraging degenerate, self-intersecting, or poorly formed predictions.

### Multi-scale matching

Ground-truth polygons are assigned to a detection scale using normalized size buckets:

```text
0.08
0.16
0.30
```

The Hungarian assignment cost combines:

- Center distance
- Size difference
- GIoU cost

This creates an explicit one-to-one relationship between text instances and prediction queries.

---

## Training System

The training loop is implemented in `pipeline/train.py`.

### Mixed precision

CUDA autocasting and gradient scaling reduce memory usage and improve training throughput while preserving stable optimization.

### VRAM-aware batch sizing

Before training, the system can probe feasible batch sizes and select a safe micro-batch for the available GPU memory.

### Gradient accumulation

Gradient accumulation allows the project to reach a larger effective batch size even when the GPU cannot fit a large micro-batch.

### Backbone scheduling

The production configuration freezes the backbone until epoch 2 and then enables end-to-end training. This provides a short stabilization phase for the detection heads and feature pyramid.

### Checkpointing

Checkpoints store:

- Model parameters
- Optimizer state
- Scheduler state
- AMP scaler state
- Epoch number
- Training step
- Validation score metadata

The best checkpoint from a training run is saved as:

```text
checkpoints/best_model.pth
```

### Logging

The logger records:

- Epoch progress
- Training loss
- Learning rate
- GPU memory usage
- Validation metrics
- Area-overlap metrics

---

## Hyperparameter Optimization

Hyperparameter search is implemented in `pipeline/optuna_search.py`.

The search uses:

- Optuna TPE sampling
- Hyperband pruning
- Fixed random seeds
- VRAM-feasible batch-size options
- Proxy training for rapid experimentation

The search space includes:

- Optimizer
- Learning rate
- Weight decay
- Scheduler
- Warmup steps
- Batch size
- Focal-loss parameters
- GIoU and shape-loss weights
- Hungarian matching costs
- FPCBAM reduction ratio
- Backbone unfreeze epoch

The exported configuration is stored in:

```text
optuna_results/best_params.yaml
```

The production configuration generated from the search includes:

```text
Optimizer: AdamW
Learning rate: 2.284455685002053e-05
Weight decay: 0.0007128188058401369
Scheduler: cosine
Warmup steps: 160
FPCBAM reduction ratio: 32
Focal alpha: 0.26652243238366186
Focal gamma: 0.7697285674832611
GIoU loss weight: 0.6414313355903041
Shape loss weight: 1.9455901926649632
Center matching cost: 3.82920382968694
Size matching cost: 2.6411393175905733
GIoU matching cost: 4.584049132667419
Backbone unfreeze epoch: 2
```

---

## Evaluation

Evaluation is implemented in `pipeline/eval.py`.

### Polygon IoU

The evaluator uses Shapely with:

- Axis-aligned bounding-box pre-filtering
- Invalid-polygon repair
- Convex-hull fallback
- Degenerate-polygon protection

### Instance metrics

Predictions are greedily matched to ground truth using polygon IoU. The evaluator reports:

- Precision
- Recall
- F1
- True positives
- False positives
- False negatives

### Raster area-overlap metrics

The project also computes pixel-area metrics:

```text
Area precision = intersection area / predicted area
Area recall    = intersection area / ground-truth area
Area F1        = harmonic mean of area precision and area recall
```

### Confidence threshold sweep

The sweep command evaluates multiple confidence thresholds and identifies the strongest operating point for F1.

---

## Dataset Setup

Download or prepare the COCO-Text v2 annotations and COCO 2014 images, then place them in the project directory:

```text
train2014/
cocotext.v2.json
```

The annotation loader automatically creates:

```text
cocotext_compact.pkl
```

Large datasets, checkpoints, and generated outputs are excluded from Git tracking through `.gitignore`.

---

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux or macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA environment using the official PyTorch installation instructions.

---

## Usage

### Train the model

```bash
python run.py train --config production_config.yaml
```

### Resume training

#### Windows

```powershell
python run.py train `
  --config extended_config.yaml `
  --resume checkpoints\best_model.pth
```

#### Linux or macOS

```bash
python run.py train \
  --config extended_config.yaml \
  --resume checkpoints/best_model.pth
```

### Evaluate a checkpoint

```bash
python run.py eval \
  --checkpoint checkpoints/best_model.pth \
  --config production_config.yaml \
  --split val \
  --conf-thresh 0.3
```

### Find the best confidence threshold

```bash
python run.py sweep \
  --checkpoint checkpoints/best_model.pth \
  --config production_config.yaml \
  --split val
```

### Run Optuna search

```bash
python run.py optuna \
  --config default_config.yaml \
  --n-trials 30 \
  --pruner hyperband \
  --out-dir optuna_results
```

---

## Project Structure

```text
.
├── pipeline/
│   ├── model.py
│   ├── dataset.py
│   ├── losses.py
│   ├── eval.py
│   ├── train.py
│   ├── config.py
│   ├── pretrained.py
│   └── optuna_search.py
├── run.py
├── default_config.yaml
├── production_config.yaml
├── extended_config.yaml
├── build_production_config.py
├── export_study.py
├── run_optuna.py
├── run_progressive_test.py
├── inspect_sample_blocks.py
├── test_loader_step.py
├── coco-dipcyt-x40s (1).ipynb
├── requirements.txt
└── .gitignore
```

### Important modules

| Module | Responsibility |
|---|---|
| `pipeline/model.py` | Backbone, FPCBAM, PANet, Swin blocks, polygon heads |
| `pipeline/dataset.py` | Annotation loading, polygon processing, augmentation, loaders |
| `pipeline/losses.py` | Hungarian assignment and composite polygon loss |
| `pipeline/train.py` | Production training, AMP, checkpoints, logging |
| `pipeline/eval.py` | Polygon IoU, instance metrics, raster metrics |
| `pipeline/config.py` | Typed configuration dataclasses and YAML serialization |
| `pipeline/pretrained.py` | Compatible YOLOv5 weight transfer |
| `pipeline/optuna_search.py` | Optuna sampler, pruner, and parameter export |
| `run.py` | Unified training, evaluation, sweep, and HPO CLI |

---

## Implementation Status

This repository contains a complete and runnable software pipeline for the DIPCYT-X40s project:

- Model construction
- Dataset loading
- Polygon preprocessing
- Data augmentation
- Multi-scale matching
- Loss computation
- Training
- Checkpointing
- Evaluation
- Confidence-threshold sweeps
- Hyperparameter search

The research implementation is still under active development. The repository should therefore be understood as a complete pipeline and implementation foundation, not as a claim that the original paper has been fully reproduced or that all final experiments have been completed.

Training outputs, checkpoints, logs, datasets, and generated experiment artifacts are intentionally kept outside the public source tree and are excluded by `.gitignore`.

---

## Reproducibility

The implementation exposes model, data, loss, matcher, hardware, and training settings through YAML configuration files.

Reproducibility features include:

- Fixed random seed
- Deterministic dataset splitting
- Explicit augmentation parameters
- Explicit optimizer and scheduler settings
- Saved model checkpoints
- Persistent training logs
- Exported hyperparameter configurations
- Configurable hardware behavior

The main configuration files are:

```text
default_config.yaml
production_config.yaml
extended_config.yaml
```

---

## Research Reference

This project is inspired by:

> Ayush Roy, Shivakumara Palaiahnakote, Umapada Pal, and Cheng-Lin Liu,  
> **A Novel Domain Independent Scene Text Localizer**, Pattern Recognition, 2024.

The implementation is intended for research, experimentation, and educational use.

---

## License

Add the license appropriate for your intended use before distributing the repository.
