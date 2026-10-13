"""Dataset preparation entry point for the face verification assessment."""
import argparse
import json
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--images-dir", default="data/raw/morphface_100x25/images")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    splits = root / args.splits_dir
    images = root / args.images_dir
    if not splits.exists():
        raise FileNotFoundError(f"Split directory not found: {splits}")
    if not images.exists():
        raise FileNotFoundError(f"Image directory not found: {images}")
    report = {}
    identity_sets = {}
    for name in ("train", "validation", "test"):
        path = splits / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("images", data.get("samples", []))
        if not records:
            raise ValueError(f"Could not identify image records in {path}; inspect its JSON schema.")
        ids = set()
        for item in records:
            if isinstance(item, dict):
                identity = item.get("identity", item.get("label", item.get("person_id")))
                if identity is not None:
                    ids.add(str(identity))
        identity_sets[name] = ids
        report[name] = {"records": len(records), "identities_detected": len(ids)}
    overlaps = {}
    names = list(identity_sets)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            overlaps[f"{a}-{b}"] = sorted(identity_sets[a] & identity_sets[b])
    output = root / "submission" / "results" / "dataset_audit.json"
    output.write_text(json.dumps({"splits": report, "identity_overlaps": overlaps}, indent=2), encoding="utf-8")
    print(json.dumps({"splits": report, "identity_overlap_counts": {k: len(v) for k, v in overlaps.items()}, "audit": str(output)}, indent=2))
if __name__ == "__main__":
    main()
