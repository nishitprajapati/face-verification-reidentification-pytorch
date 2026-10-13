import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import FaceIdentityDataset
from model import FaceEmbeddingModel


SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
CONTRASTIVE_MARGIN = 1.0
TRIPLET_MARGIN = 0.5
EARLY_STOPPING_PATIENCE = 7
MIN_DELTA = 0.001
CHECKPOINT_EVERY = 3
VALIDATION_PAIRS_PER_CLASS = 1000

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class PairDataset(Dataset):
    """Sampled training pairs for contrastive learning."""

    def __init__(self, base_dataset, seed=SEED):
        self.base_dataset = base_dataset
        self.rng = random.Random(seed)
        self.by_identity = {}
        for index, record in enumerate(base_dataset.records):
            self.by_identity.setdefault(record["identity"], []).append(index)
        self.identities = sorted(self.by_identity, key=int)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image_a, _ = self.base_dataset[index]
        identity_a = self.base_dataset.records[index]["identity"]

        if self.rng.random() < 0.5:
            candidates = self.by_identity[identity_a]
            other_index = index
            while len(candidates) > 1 and other_index == index:
                other_index = self.rng.choice(candidates)
            same = 1.0
        else:
            other_ids = [identity for identity in self.identities if identity != identity_a]
            identity_b = self.rng.choice(other_ids)
            other_index = self.rng.choice(self.by_identity[identity_b])
            same = 0.0

        image_b, _ = self.base_dataset[other_index]
        return image_a, image_b, torch.tensor(same, dtype=torch.float32)


class TripletDataset(Dataset):
    """Sampled anchor-positive-negative triplets."""

    def __init__(self, base_dataset, seed=SEED):
        self.base_dataset = base_dataset
        self.rng = random.Random(seed)
        self.by_identity = {}
        for index, record in enumerate(base_dataset.records):
            self.by_identity.setdefault(record["identity"], []).append(index)
        self.identities = sorted(self.by_identity, key=int)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        anchor, _ = self.base_dataset[index]
        identity = self.base_dataset.records[index]["identity"]

        positives = self.by_identity[identity]
        positive_index = index
        while len(positives) > 1 and positive_index == index:
            positive_index = self.rng.choice(positives)

        other_ids = [item for item in self.identities if item != identity]
        negative_identity = self.rng.choice(other_ids)
        negative_index = self.rng.choice(self.by_identity[negative_identity])

        positive, _ = self.base_dataset[positive_index]
        negative, _ = self.base_dataset[negative_index]
        return anchor, positive, negative


def make_fixed_validation_pairs(dataset, seed=SEED, pairs_per_class=VALIDATION_PAIRS_PER_CLASS):
    """Create a fixed, balanced set of genuine/impostor pairs once before training."""
    rng = random.Random(seed)
    by_identity = {}
    for index, record in enumerate(dataset.records):
        by_identity.setdefault(record["identity"], []).append(index)
    identities = sorted(by_identity, key=int)

    if len(identities) < 2:
        raise ValueError("Validation needs at least two identities.")

    positive_pairs = []
    for identity in identities:
        indices = by_identity[identity]
        possible = [(indices[i], indices[j]) for i in range(len(indices)) for j in range(i + 1, len(indices))]
        rng.shuffle(possible)
        positive_pairs.extend(possible[:max(1, pairs_per_class // len(identities))])
    rng.shuffle(positive_pairs)
    positive_pairs = positive_pairs[:pairs_per_class]

    negative_pairs = []
    seen = set()
    attempts = 0
    max_attempts = pairs_per_class * 100
    while len(negative_pairs) < pairs_per_class and attempts < max_attempts:
        identity_a, identity_b = rng.sample(identities, 2)
        index_a = rng.choice(by_identity[identity_a])
        index_b = rng.choice(by_identity[identity_b])
        pair = (index_a, index_b)
        if pair not in seen:
            seen.add(pair)
            negative_pairs.append(pair)
        attempts += 1

    if not positive_pairs or not negative_pairs:
        raise ValueError("Could not create enough validation pairs.")
    count = min(len(positive_pairs), len(negative_pairs))
    pairs = positive_pairs[:count] + negative_pairs[:count]
    labels = [1] * count + [0] * count
    return pairs, np.asarray(labels, dtype=np.int64)


def contrastive_loss(embedding_a, embedding_b, same, margin):
    distance = F.pairwise_distance(embedding_a, embedding_b)
    positive = same * distance.pow(2)
    negative = (1.0 - same) * F.relu(margin - distance).pow(2)
    return (positive + negative).mean()


def triplet_loss(anchor, positive, negative, margin):
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin, p=2)


def train_one_epoch(model, loader, optimizer, device, objective):
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    progress = tqdm(loader, desc=objective, dynamic_ncols=True)

    for batch in progress:
        optimizer.zero_grad(set_to_none=True)

        if objective == "cross_entropy":
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            embeddings = model(images)
            logits = model.classifier(embeddings)
            loss = F.cross_entropy(logits, labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count = labels.size(0)
        elif objective == "contrastive":
            image_a, image_b, same = batch
            image_a = image_a.to(device, non_blocking=True)
            image_b = image_b.to(device, non_blocking=True)
            same = same.to(device, non_blocking=True)
            embedding_a = model(image_a)
            embedding_b = model(image_b)
            loss = contrastive_loss(embedding_a, embedding_b, same, CONTRASTIVE_MARGIN)
            count = same.size(0)
        elif objective == "triplet":
            anchor, positive, negative = batch
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)
            embedding_a = model(anchor)
            embedding_p = model(positive)
            embedding_n = model(negative)
            loss = triplet_loss(embedding_a, embedding_p, embedding_n, TRIPLET_MARGIN)
            count = anchor.size(0)
        else:
            raise ValueError(f"Unsupported objective: {objective}")

        if not torch.isfinite(loss).all():
            raise RuntimeError(f"Non-finite loss detected for {objective}; stopping training.")

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * count
        total_samples += count
        progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

    metrics = {"loss": total_loss / max(total_samples, 1)}
    if objective == "cross_entropy":
        metrics["accuracy"] = correct / max(total_samples, 1)
    return metrics


@torch.no_grad()
def compute_embeddings(model, dataset, device, batch_size=32):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings = []
    for images, _ in tqdm(loader, desc="Validation embeddings", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        embeddings.append(model(images).cpu())
    return torch.cat(embeddings, dim=0)


def evaluate_validation_auc(model, dataset, pairs, labels, device):
    embeddings = compute_embeddings(model, dataset, device)
    pair_a = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    pair_b = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    scores = (embeddings[pair_a] * embeddings[pair_b]).sum(dim=1).numpy()
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    target_far = 0.01
    eligible = np.where(fpr <= target_far)[0]
    tar_at_far = float(tpr[eligible[-1]]) if len(eligible) else 0.0
    return auc, tar_at_far, scores


def save_checkpoint(path, model, optimizer, epoch, best_auc, epochs_without_improvement,
                    args, dataset, scheduler):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation_auc": best_auc,
        "epochs_without_improvement": epochs_without_improvement,
        "objective": args.objective,
        "seed": args.seed,
        "embedding_dim": model.embedding_dim,
        "architecture": "ResNet18",
        "identity_names": dataset.identity_names,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
    }
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["cross_entropy", "contrastive", "triplet"],
                        default="cross_entropy")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=MIN_DELTA)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size, and --patience must be positive")
    if args.min_delta < 0:
        parser.error("--min-delta cannot be negative")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Objective: {args.objective}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Precision: FP32")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = FaceIdentityDataset("data/splits/train.json")
    validation_dataset = FaceIdentityDataset("data/splits/validation.json")
    train_generator = torch.Generator().manual_seed(args.seed)

    if args.objective == "cross_entropy":
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, generator=train_generator, pin_memory=device.type == "cuda")
    elif args.objective == "contrastive":
        pair_dataset = PairDataset(train_dataset, seed=args.seed)
        loader = DataLoader(pair_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, generator=train_generator, pin_memory=device.type == "cuda")
    else:
        triplet_dataset = TripletDataset(train_dataset, seed=args.seed)
        loader = DataLoader(triplet_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, generator=train_generator, pin_memory=device.type == "cuda")

    model = FaceEmbeddingModel(pretrained=args.resume is None)
    model.classifier = (nn.Linear(model.embedding_dim, len(train_dataset.identity_names))
                        if args.objective == "cross_entropy" else None)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, threshold=args.min_delta
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"resnet18_{args.objective}_best.pt"
    last_path = CHECKPOINT_DIR / f"resnet18_{args.objective}_last.pt"
    log_path = RESULTS_DIR / f"training_{args.objective}.jsonl"
    periodic_pattern = f"resnet18_{args.objective}_epoch_{{:03d}}.pt"

    validation_pairs, validation_labels = make_fixed_validation_pairs(
        validation_dataset, seed=args.seed
    )
    print(f"Training images: {len(train_dataset)} across {len(train_dataset.identity_names)} identities")
    print(f"Validation images: {len(validation_dataset)} across {len(validation_dataset.identity_names)} identities")
    print(f"Fixed validation pairs: {len(validation_pairs)} "
          f"({int(validation_labels.sum())} genuine, {int((validation_labels == 0).sum())} impostor)")
    print(f"Max epochs: {args.epochs} | Early-stopping patience: {args.patience} | "
          f"Min AUC improvement: {args.min_delta}")
    print(f"Best checkpoint: {best_path}")
    print(f"Recovery checkpoint: {last_path}")

    start_epoch = 1
    best_auc = -float("inf")
    epochs_without_improvement = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("objective") != args.objective:
            raise ValueError("Resume checkpoint objective does not match --objective.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_auc = float(checkpoint["best_validation_auc"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        print(f"Resumed from epoch {checkpoint['epoch']}; best validation ROC-AUC={best_auc:.4f}")

    log_mode = "a" if args.resume else "w"
    stopped_early = False
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        for epoch in range(start_epoch, args.epochs + 1):
            metrics = train_one_epoch(model, loader, optimizer, device, args.objective)
            val_auc, tar_at_far, _ = evaluate_validation_auc(
                model, validation_dataset, validation_pairs, validation_labels, device
            )
            scheduler.step(val_auc)
            current_lr = float(optimizer.param_groups[0]["lr"])

            improved = val_auc > best_auc + args.min_delta
            if improved:
                best_auc = val_auc
                epochs_without_improvement = 0
                save_checkpoint(best_path, model, optimizer, epoch, best_auc,
                                epochs_without_improvement, args, train_dataset, scheduler)
                print(f"  BEST MODEL UPDATED: validation ROC-AUC={best_auc:.4f}")
            else:
                epochs_without_improvement += 1

            metrics.update({
                "epoch": epoch,
                "objective": args.objective,
                "seed": args.seed,
                "validation_roc_auc": val_auc,
                "validation_tar_at_far_1_percent": tar_at_far,
                "best_validation_roc_auc": best_auc,
                "epochs_without_improvement": epochs_without_improvement,
                "learning_rate": current_lr,
            })
            print(
                f"Epoch {epoch:02d}/{args.epochs} | train_loss={metrics['loss']:.4f}"
                + (f" | train_accuracy={metrics['accuracy']:.4f}" if "accuracy" in metrics else "")
                + f" | val_ROC_AUC={val_auc:.4f} | val_TAR@FAR1%={tar_at_far:.4f}"
                + f" | best_AUC={best_auc:.4f} | no_improve={epochs_without_improvement}/{args.patience}"
                + f" | lr={current_lr:.2e}"
            )
            log_file.write(json.dumps(metrics) + "\n")
            log_file.flush()

            # Always retain the latest full training state for recovery/resume.
            save_checkpoint(last_path, model, optimizer, epoch, best_auc,
                            epochs_without_improvement, args, train_dataset, scheduler)

            # Periodic recovery snapshots, in addition to the per-epoch last checkpoint.
            if epoch % CHECKPOINT_EVERY == 0:
                periodic_path = CHECKPOINT_DIR / periodic_pattern.format(epoch)
                save_checkpoint(periodic_path, model, optimizer, epoch, best_auc,
                                epochs_without_improvement, args, train_dataset, scheduler)
                print(f"  Periodic checkpoint saved: {periodic_path}")

            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}: validation ROC-AUC did not improve "
                      f"by {args.min_delta} for {args.patience} consecutive epochs.")
                stopped_early = True
                break

    print("Training finished." if not stopped_early else "Training finished by early stopping.")
    print(f"Best validation ROC-AUC: {best_auc:.4f}")
    print(f"Best checkpoint: {best_path.resolve()}")
    print(f"Latest recovery checkpoint: {last_path.resolve()}")
    print(f"Training log: {log_path.resolve()}")


if __name__ == "__main__":
    main()
