"""Generate reproducible balanced verification pairs from split JSON files."""
import argparse, json, random, csv
from pathlib import Path
from itertools import combinations

def load_records(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else data.get("images", data.get("samples", []))
    out = []
    for r in records:
        if not isinstance(r, dict):
            continue
        identity = r.get("identity", r.get("label", r.get("person_id")))
        image = r.get("path", r.get("image", r.get("image_path", r.get("file"))))
        if identity is not None and image is not None:
            out.append((str(identity), str(image)))
    if not out:
        raise ValueError(f"Unsupported record schema in {path}. Expected identity and image/path fields.")
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train","validation","test"], default="validation")
    p.add_argument("--num-pairs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    records = load_records(root / "data" / "splits" / f"{args.split}.json")
    by_id = {}
    for ident, path in records:
        by_id.setdefault(ident, []).append(path)
    rng = random.Random(args.seed)
    positives = []
    for ident, paths in by_id.items():
        positives.extend((ident, a, b, 1) for a,b in combinations(paths,2) if a != b)
    negatives = []
    ids = sorted(by_id)
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            negatives.extend((a, x, y, 0) for x in by_id[a] for y in by_id[b])
    if not positives or not negatives:
        raise ValueError("Need at least two images per identity and two identities.")
    n = min(args.num_pairs // 2, len(positives), len(negatives))
    rng.shuffle(positives); rng.shuffle(negatives)
    rows = []
    for ident, a, b, label in positives[:n] + negatives[:n]:
        rows.append({"image1": a, "image2": b, "same_identity": label})
    rng.shuffle(rows)
    out = root / "submission" / "results" / f"{args.split}_pairs.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image1","image2","same_identity"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved {len(rows)} pairs ({sum(r['same_identity'] for r in rows)} positive): {out}")
if __name__ == "__main__":
    main()
