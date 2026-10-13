"""Face verification evaluation and artifact generation for the assessment.

Run from the project root:
    python submission/evaluate_pairs.py

Writes actual model-derived pair scores, metrics, ROC and similarity plots.
No training is performed by this script.
"""
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader

from dataset import FaceIdentityDataset
from model import FaceEmbeddingModel

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent if (HERE.parent / "data" / "splits").exists() else HERE
DATA_ROOT = PROJECT_ROOT
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
CHECKPOINT_DIR = HERE / "checkpoints"
RESULTS_DIR = HERE / "results"
SEED = 42
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "cross_entropy": CHECKPOINT_DIR / "resnet18_cross_entropy_best.pt",
    "contrastive": CHECKPOINT_DIR / "resnet18_contrastive_best.pt",
    "triplet": CHECKPOINT_DIR / "resnet18_triplet_best.pt",
}

def make_pairs(dataset, seed, pairs_per_class=1000):
    """Create deterministic balanced genuine/impostor pairs from one split."""
    rng = random.Random(seed)
    by_identity = {}
    for idx, record in enumerate(dataset.records):
        by_identity.setdefault(str(record["identity"]), []).append(idx)
    identities = sorted(by_identity)
    if len(identities) < 2 or any(len(v) < 2 for v in by_identity.values()):
        raise ValueError("Pair generation requires at least two identities and two images per identity.")

    positive_candidates = []
    for identity, indices in by_identity.items():
        candidates = [(indices[i], indices[j]) for i in range(len(indices)) for j in range(i + 1, len(indices))]
        positive_candidates.extend(candidates)
    negative_candidates = []
    for i, ident_a in enumerate(identities):
        for ident_b in identities[i + 1:]:
            negative_candidates.extend((a, b) for a in by_identity[ident_a] for b in by_identity[ident_b])
    rng.shuffle(positive_candidates)
    rng.shuffle(negative_candidates)
    count = min(pairs_per_class, len(positive_candidates), len(negative_candidates))
    pairs = [(a, b, 1) for a, b in positive_candidates[:count]]
    pairs += [(a, b, 0) for a, b in negative_candidates[:count]]
    rng.shuffle(pairs)
    return pairs

@torch.no_grad()
def extract_embeddings(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    outputs = []
    for images, _labels in loader:
        images = images.to(DEVICE)
        embeddings = model(images)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        outputs.append(embeddings.cpu().numpy())
    return np.concatenate(outputs, axis=0)

def score_pairs(embeddings, pairs):
    a = np.asarray([p[0] for p in pairs], dtype=np.int64)
    b = np.asarray([p[1] for p in pairs], dtype=np.int64)
    return np.sum(embeddings[a] * embeddings[b], axis=1)

def choose_threshold_for_far(labels, scores, target_far=0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eligible = np.flatnonzero(fpr <= target_far)
    if not len(eligible):
        return float(thresholds[int(np.argmin(fpr))])
    return float(thresholds[eligible[np.argmax(tpr[eligible])]])

def metrics(labels, scores, threshold):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eligible = np.flatnonzero(fpr <= 0.01)
    tar_far_1 = float(np.max(tpr[eligible])) if len(eligible) else 0.0
    pred = scores >= threshold
    genuine = labels == 1
    impostor = labels == 0
    tp = int(np.sum(pred & genuine))
    fn = int(np.sum((~pred) & genuine))
    fp = int(np.sum(pred & impostor))
    tn = int(np.sum((~pred) & impostor))
    return {
        "roc_auc": auc, "eer_estimate": eer, "tar_at_far_1_percent": tar_far_1,
        "threshold": float(threshold), "accuracy_at_threshold": float(np.mean(pred == labels)),
        "far_at_threshold": float(fp / max(1, np.sum(impostor))),
        "frr_at_threshold": float(fn / max(1, np.sum(genuine))),
        "true_accept_rate_at_threshold": float(tp / max(1, np.sum(genuine))),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "fpr": fpr, "tpr": tpr, "roc_thresholds": thresholds,
    }

def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = FaceEmbeddingModel(pretrained=False)
    if checkpoint.get("objective") == "cross_entropy":
        model.classifier = torch.nn.Linear(model.embedding_dim, len(checkpoint["identity_names"]))
    else:
        model.classifier = None
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE).eval()
    return model, checkpoint

def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in CHECKPOINTS.values():
        if not path.exists():
            raise FileNotFoundError(f"Required checkpoint not found: {path}")
    for split in ("validation", "test"):
        if not (SPLITS_DIR / f"{split}.json").exists():
            raise FileNotFoundError(f"Required split JSON not found: {SPLITS_DIR / (split + '.json')}")

    print(f"Device: {DEVICE}")
    validation = FaceIdentityDataset(SPLITS_DIR / "validation.json")
    test = FaceIdentityDataset(SPLITS_DIR / "test.json")
    val_pairs = make_pairs(validation, SEED)
    test_pairs = make_pairs(test, SEED + 1)
    val_labels = np.asarray([p[2] for p in val_pairs], dtype=np.int64)
    test_labels = np.asarray([p[2] for p in test_pairs], dtype=np.int64)

    metric_rows = []
    details = {}
    roc_curves = {}
    score_rows = []
    distribution_scores = {}

    for objective, checkpoint_path in CHECKPOINTS.items():
        print(f"Evaluating {objective} from {checkpoint_path.name}")
        model, checkpoint = load_model(checkpoint_path)
        val_embeddings = extract_embeddings(model, validation)
        test_embeddings = extract_embeddings(model, test)
        val_scores = score_pairs(val_embeddings, val_pairs)
        test_scores = score_pairs(test_embeddings, test_pairs)
        threshold = choose_threshold_for_far(val_labels, val_scores, 0.01)
        val_metrics = metrics(val_labels, val_scores, threshold)
        test_metrics = metrics(test_labels, test_scores, threshold)
        roc_curves[objective] = (test_metrics["fpr"], test_metrics["tpr"], test_metrics["roc_auc"])
        distribution_scores[objective] = (test_scores[test_labels == 1], test_scores[test_labels == 0])

        metric_rows.append({
            "objective": objective,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "validation_roc_auc": val_metrics["roc_auc"],
            "validation_eer_estimate": val_metrics["eer_estimate"],
            "validation_tar_at_far_1_percent": val_metrics["tar_at_far_1_percent"],
            "validation_selected_threshold_for_far_1_percent": threshold,
            "test_roc_auc": test_metrics["roc_auc"],
            "test_eer_estimate": test_metrics["eer_estimate"],
            "test_tar_at_far_1_percent": test_metrics["tar_at_far_1_percent"],
            "test_accuracy_at_validation_threshold": test_metrics["accuracy_at_threshold"],
            "test_far_at_validation_threshold": test_metrics["far_at_threshold"],
            "test_frr_at_validation_threshold": test_metrics["frr_at_threshold"],
            "test_tar_at_validation_threshold": test_metrics["true_accept_rate_at_threshold"],
            "test_pair_count": len(test_labels),
        })
        details[objective] = {
            "checkpoint": str(checkpoint_path.relative_to(HERE)),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "validation": {k: v for k, v in val_metrics.items() if k not in {"fpr", "tpr", "roc_thresholds"}},
            "test": {k: v for k, v in test_metrics.items() if k not in {"fpr", "tpr", "roc_thresholds"}},
            "threshold_selection": "Selected on validation pairs only: highest TAR at empirical FAR <= 1%.",
            "validation_pair_count": len(val_labels), "test_pair_count": len(test_labels), "seed": SEED,
        }
        for split_name, pairs, labels, scores in (
            ("validation", val_pairs, val_labels, val_scores),
            ("test", test_pairs, test_labels, test_scores),
        ):
            for (a, b, _), label, score in zip(pairs, labels, scores):
                score_rows.append({
                    "objective": objective, "split": split_name,
                    "image1": str((validation if split_name == "validation" else test).records[a]["image_path"]),
                    "image2": str((validation if split_name == "validation" else test).records[b]["image_path"]),
                    "same_identity": int(label), "cosine_similarity": float(score),
                    "validation_selected_threshold": threshold,
                })
        print(f"  validation AUC={val_metrics['roc_auc']:.4f}; test AUC={test_metrics['roc_auc']:.4f}; test EER~={test_metrics['eer_estimate']:.4f}")
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(RESULTS_DIR / "verification_comparison.csv", metric_rows)
    write_csv(RESULTS_DIR / "evaluation_results.csv", metric_rows)
    write_csv(RESULTS_DIR / "verification_scores.csv", score_rows)
    metrics_json = {
        "dataset": "MorphFace subset: 100 identities x 25 images",
        "seed": SEED,
        "validation_protocol": "Balanced fixed pairs from validation identities, disjoint from training.",
        "test_protocol": "Balanced fixed pairs from test identities, disjoint from training and validation.",
        "models": details,
    }
    (RESULTS_DIR / "verification_metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    (RESULTS_DIR / "evaluation_results.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    # Required ROC curve, based on held-out test scores.
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Chance")
    plt.xlim(0, 1); plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate (FAR)"); plt.ylabel("True Positive Rate (TAR)")
    plt.title("Face Verification ROC — Held-out Test Identities")
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=160)
    plt.savefig(RESULTS_DIR / "test_roc_comparison.png", dpi=160)
    plt.close()

    # Required similarity distribution, using the actual test-pair scores.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (name, (genuine, impostor)) in zip(axes, distribution_scores.items()):
        ax.hist(impostor, bins=35, alpha=0.65, density=True, label="Different identity")
        ax.hist(genuine, bins=35, alpha=0.65, density=True, label="Same identity")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Cosine similarity")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Density")
    axes[-1].legend()
    fig.suptitle("Genuine vs. Impostor Similarity — Held-out Test Pairs")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "similarity_distribution.png", dpi=160)
    plt.close(fig)

    # Save raw ROC coordinates as a reusable CSV.
    roc_rows = []
    for objective, (fpr, tpr, auc) in roc_curves.items():
        for x, y in zip(fpr, tpr):
            roc_rows.append({"objective": objective, "false_positive_rate": float(x),
                             "true_positive_rate": float(y), "roc_auc": float(auc)})
    write_csv(RESULTS_DIR / "roc_curve_points.csv", roc_rows)
    print("\nArtifacts saved:")
    for name in ("roc_curve.png", "similarity_distribution.png", "evaluation_results.json",
                 "evaluation_results.csv", "verification_scores.csv", "roc_curve_points.csv",
                 "verification_comparison.csv", "verification_metrics.json"):
        print(" -", RESULTS_DIR / name)
    print("Thresholds are selected on validation pairs only; test data is not used for threshold selection.")

if __name__ == "__main__":
    main()
