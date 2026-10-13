import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import FaceEmbeddingModel

SEED = 42
GALLERY_IMAGES_PER_IDENTITY = 5
BATCH_SIZE = 32

TEST_JSON = Path("data/splits/test.json")
RESULTS_DIR = Path("results/evaluation")

CHECKPOINTS = {
    "cross_entropy": Path("checkpoints/resnet18_cross_entropy_best.pt"),
    "contrastive": Path("checkpoints/resnet18_contrastive_best.pt"),
    "triplet": Path("checkpoints/resnet18_triplet_best.pt"),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_records():
    with TEST_JSON.open("r", encoding="utf-8-sig") as file:
        records = json.load(file)

    by_identity = {}
    for record in records:
        by_identity.setdefault(record["identity"], []).append(record)

    rng = random.Random(SEED)
    gallery, probes = [], []

    for identity in sorted(by_identity, key=lambda value: int(value)):
        images = sorted(
            by_identity[identity],
            key=lambda record: record["image_path"],
        )
        rng.shuffle(images)

        if len(images) <= GALLERY_IMAGES_PER_IDENTITY:
            raise ValueError(f"Not enough images for identity {identity}")

        gallery.extend(images[:GALLERY_IMAGES_PER_IDENTITY])
        probes.extend(images[GALLERY_IMAGES_PER_IDENTITY:])

    gallery_paths = {record["image_path"] for record in gallery}
    probe_paths = {record["image_path"] for record in probes}

    if gallery_paths & probe_paths:
        raise RuntimeError("Gallery/probe image overlap detected.")

    return gallery, probes


def embed_records(model, records):
    embeddings = []
    model.eval()

    with torch.inference_mode():
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start:start + BATCH_SIZE]
            images = []

            for record in batch:
                with Image.open(record["image_path"]) as image:
                    images.append(TRANSFORM(image.convert("RGB")))

            batch_tensor = torch.stack(images).to(DEVICE)
            vectors = model(batch_tensor)
            embeddings.append(vectors.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = FaceEmbeddingModel(pretrained=False)

    if checkpoint["objective"] == "cross_entropy":
        model.classifier = torch.nn.Linear(
            model.embedding_dim,
            len(checkpoint["identity_names"]),
        )
    else:
        model.classifier = None

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(DEVICE).eval()


def evaluate_rankings(
    gallery_embeddings,
    gallery_ids,
    probe_embeddings,
    probe_ids,
):
    # Unit-normalized embeddings make dot product equal cosine similarity.
    image_similarities = probe_embeddings @ gallery_embeddings.T

    gallery_ids = np.asarray(gallery_ids)
    probe_ids = np.asarray(probe_ids)

    identities = sorted(set(gallery_ids), key=lambda value: int(value))

    # Aggregate multiple gallery-image scores into one score per identity.
    identity_scores = np.column_stack([
        image_similarities[:, gallery_ids == identity].max(axis=1)
        for identity in identities
    ])

    ranked_identity_indices = np.argsort(-identity_scores, axis=1)
    ranked_identities = np.asarray(identities)[ranked_identity_indices]

    max_rank = len(identities)
    cmc = []

    for rank in range(1, max_rank + 1):
        hits = np.any(
            ranked_identities[:, :rank] == probe_ids[:, None],
            axis=1,
        )
        cmc.append(float(np.mean(hits)))

    return {
        "rank1": cmc[0],
        "rank5": cmc[min(4, max_rank - 1)],
        "cmc": cmc,
        "max_rank": max_rank,
        "identity_count": max_rank,
        "probe_count": len(probe_ids),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gallery, probes = load_records()
    gallery_ids = [record["identity"] for record in gallery]
    probe_ids = [record["identity"] for record in probes]

    print(f"Device: {DEVICE}")
    print(
        f"Gallery: {len(gallery)} images, "
        f"{len(set(gallery_ids))} identities"
    )
    print(f"Probes: {len(probes)} images")
    print("Gallery/probe image overlap: none")
    print("Ranking unit: unique identity, using maximum gallery-image similarity")

    all_results = {}

    for name, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\nEvaluating {name}...")
        model = load_model(checkpoint_path)

        gallery_embeddings = embed_records(model, gallery)
        probe_embeddings = embed_records(model, probes)

        result = evaluate_rankings(
            gallery_embeddings,
            gallery_ids,
            probe_embeddings,
            probe_ids,
        )
        all_results[name] = result

        print(f"Rank-1: {result['rank1']:.4f}")
        print(f"Rank-5: {result['rank5']:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = RESULTS_DIR / "gallery_probe_comparison.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "objective",
            "rank1_accuracy",
            "rank5_accuracy",
            "probe_count",
            "identity_count",
            "gallery_images_per_identity",
            "identity_score_aggregation",
        ])

        for name, result in all_results.items():
            writer.writerow([
                name,
                result["rank1"],
                result["rank5"],
                result["probe_count"],
                result["identity_count"],
                GALLERY_IMAGES_PER_IDENTITY,
                "maximum gallery-image cosine similarity",
            ])

    max_rank = min(result["max_rank"] for result in all_results.values())
    ranks = np.arange(1, max_rank + 1)

    plt.figure(figsize=(8, 6))

    for name, result in all_results.items():
        plt.plot(ranks, result["cmc"][:max_rank], label=name)

    plt.xlabel("Rank")
    plt.ylabel("Identification Rate")
    plt.title("Gallery–Probe CMC Comparison (Test Set)")
    plt.xticks(range(1, max_rank + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = RESULTS_DIR / "gallery_probe_cmc.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print("\n=== Corrected identity-level comparison ===")
    for name, result in all_results.items():
        print(
            f"{name:14s} "
            f"Rank-1={result['rank1']:.4f} "
            f"Rank-5={result['rank5']:.4f}"
        )

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
