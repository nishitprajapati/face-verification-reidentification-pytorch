# Face Verification and Re-Identification

## 1. Overview

This project implements a face-verification and gallery–probe identification pipeline using a ResNet-18 embedding model. Three training objectives were evaluated: cross-entropy, contrastive loss, and triplet loss.

The model produces 512-dimensional L2-normalized embeddings. Verification compares embeddings using cosine similarity; classifier logits are not used as verification scores.

## 2. Dataset and split protocol

- Dataset subset: MorphFace, 100 identities × 25 images (2,500 images).
- Split definitions: `data/splits/train.json`, `data/splits/validation.json`, and `data/splits/test.json`.
- The split protocol is identity-disjoint: training, validation, and test identities are separated.
- Pair generation uses deterministic seeds and balanced genuine/impostor pairs.
- The test split is held out from model/threshold tuning.

Dataset source: https://huggingface.co/datasets/LuxPolaris/MorphFace  
The source records a CC-BY-4.0 license. Verify the dataset's current license terms and attribution requirements before redistributing dataset images.

## 3. Preprocessing

Preprocessing was considered as five separate stages. The implementation choices and their rationale are documented below. 

### 3.1 Face presence verification

The selected dataset consists of images organized by identity. Representative images were manually inspected to verify that they contained recognizable facial content.

The inspected samples exhibit face-focused framing, with facial features occupying most of the image area.

### 3.2 Face cropping

Additional face-specific cropping was deliberately omitted because the inspected images already contain tightly framed faces. Applying another crop could unnecessarily truncate facial regions, including the forehead, chin, or sides of the face.

Retaining the source framing avoids introducing an additional, unvalidated transformation and preserves the facial information already present in the images. This decision is specific to the inspected dataset and does not imply that cropping is unnecessary for unconstrained images containing substantial background.

### 3.3 Face alignment

Explicit landmark-based face alignment was not applied. This preserves the original variation in pose, orientation, and expression rather than imposing an additional geometric transformation.

Retaining pose variation can encourage the embedding model to learn identity-discriminative representations under varied image conditions.


### 3.4 Resizing to model input resolution

**Decision:** Images are resized to 224 × 224 pixels.

**Reasoning:** This provides a consistent tensor shape for batching and matches the input resolution used by the ResNet-18/ImageNet preprocessing pipeline. Directly resizing to a square can alter the original aspect ratio and distort facial geometry. Aspect-ratio-preserving resizing with padding, or detector-based face crops, would be alternatives, but they are not used by the current implementation.

### 3.5 Normalization

**Decision:** Images are converted to RGB, converted to tensors, and normalized using the ImageNet channel means and standard deviations.

**Reasoning:** This follows the input-normalization convention used with ImageNet-pretrained ResNet weights and standardizes channel scales. Normalization does not correct lighting variation, blur, occlusion, or dataset-to-deployment domain shift.

**Implemented transform summary:** RGB conversion → resize to 224 × 224 → tensor conversion → ImageNet normalization. Automated face detection, detector-driven cropping, and landmark alignment are not performed by the current preprocessing code.

## 4. Model and training

- Backbone: ResNet-18, initialized with ImageNet-pretrained weights during training.
- Embedding dimension: 512.
- Embeddings are L2-normalized.
- Objectives: cross-entropy, contrastive, and triplet.
- Training logs are retained under `results/`.
- The submission retains the best checkpoint for each objective: `resnet18_cross_entropy_best.pt`, `resnet18_contrastive_best.pt`, and `resnet18_triplet_best.pt`.
- `checkpoints/best_model.pth` is the assessment-required filename and a convenience copy of the triplet best checkpoint.

## 5. Environment and setup

The recorded environment uses Python 3.11.9 and CUDA-enabled PyTorch. Exact package pins are recorded in `requirements.txt`.

From the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\submission\requirements.txt
```

Use a compatible NVIDIA driver/CUDA-enabled PyTorch installation for GPU execution. CPU execution may be slower.

## 6. Verification evaluation

Run from the project root:

```powershell
python .\submission\evaluate_pairs.py
```

The evaluator creates balanced genuine/impostor pairs independently within validation and test splits. It extracts embeddings, computes cosine similarities, calculates ROC-AUC and an EER estimate, and selects a threshold on validation data by maximizing TAR subject to empirical FAR ≤ 1%. That validation-selected threshold is then applied to test scores.

### Latest recorded run

| Objective | Best checkpoint epoch | Validation ROC-AUC | Test ROC-AUC | Test EER estimate |
|---|---:|---:|---:|---:|
| Cross-entropy | 8 | 0.8118 | 0.8424 | 0.2375 |
| Contrastive | 11 | 0.7938 | 0.8325 | 0.2520 |
| Triplet | 40 | 0.8147 | 0.8454 | 0.2310 |

These values are from the latest run of `submission/evaluate_pairs.py`

### Verification outputs

- `results/evaluation_results.json`
- `results/evaluation_results.csv`
- `results/verification_metrics.json`
- `results/verification_comparison.csv`
- `results/verification_scores.csv` — per-pair scores generated by the submission evaluator
- `results/roc_curve_points.csv` — ROC coordinates from actual test scores
- `results/roc_curve.png`
- `results/similarity_distribution.png`
- `results/test_roc_comparison.png` — compatibility copy of the ROC comparison plot

## 7. Gallery–probe identification

Run from the project root:

```powershell
python .\submission\gallery_probe.py
```

The evaluation uses five gallery images per test identity and the remaining 20 images as probes: 20 identities, 100 gallery images, and 400 probe images. Gallery and probe images do not overlap. Gallery-image similarities are aggregated per identity using the maximum similarity, so each identity occupies one ranking position.

### Recorded identity-level results

| Objective | Rank-1 | Rank-5 |
|---|---:|---:|
| Cross-entropy | 46.00% | 84.50% |
| Contrastive | 38.75% | 82.75% |
| Triplet | 47.50% | 85.50% |

Triplet achieved the highest Rank-1 and Rank-5 values under this protocol.

Outputs include:
- `results/gallery_probe_results.csv`
- `results/evaluation/gallery_probe_cmc.png` — cumulative matching characteristic plot.

## 8. Required scripts

- `dataset_preparation.py`: dataset and split audit/preparation utilities.
- `dataset.py`: PyTorch `FaceIdentityDataset` loader and image transforms.
- `train.py`: model training.
- `model.py`: embedding model.
- `generate_pairs.py`: balanced pair generation from split records.
- `evaluate_pairs.py`: verification evaluation and score/plot exports.
- `gallery_probe.py`: gallery–probe identification.
- `roc_analysis.py`: ROC analysis utility.

The exact script arguments and expected split JSON schema should be checked against each script before running it independently. The primary verification artifact-generation path is `evaluate_pairs.py`.

## 9. Checkpoints and training evidence

`checkpoints/` contains `best_model.pth` plus the best checkpoint for each of the three training objectives: `resnet18_cross_entropy_best.pt`, `resnet18_contrastive_best.pt`, and `resnet18_triplet_best.pt`. Periodic and last-epoch checkpoint files were excluded from the submission folder to keep the deliverable focused. The original project checkpoints remain outside the submission folder.

Training logs:
- `results/training_cross_entropy.jsonl`
- `results/training_contrastive.jsonl`
- `results/training_triplet.jsonl`

