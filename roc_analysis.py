"""Create ROC comparison plot from actual evaluate_pairs.py outputs."""
import json, csv
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "results" / "evaluation" / "verification_metrics.json",
        root / "submission" / "results" / "verification_metrics.json",
        root / "submission" / "results" / "evaluation_results.json",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError("No verification metrics JSON found. Run evaluate_pairs.py first.")
    data = json.loads(src.read_text(encoding="utf-8"))
    # Metrics-only JSON may not contain per-pair ROC coordinates. Do not fabricate curves.
    outdir = root / "submission" / "results"
    (outdir / "roc_analysis_source.json").write_text(json.dumps({
        "source_file": str(src),
        "metrics": data,
        "note": "ROC coordinates require actual pair scores; this script does not invent scores."
    }, indent=2), encoding="utf-8")
    print(f"Metrics source archived: {src}")
    print("Checking for saved ROC plot from the evaluation run...")
    plots = list(src.parent.glob("*roc*.png")) + list((root / "results" / "evaluation").glob("*roc*.png"))
    if plots:
        for plot in plots:
            dest = outdir / "roc_curve.png"
            if plot.resolve() != dest.resolve():
                import shutil
                shutil.copy2(plot, dest)
            print(f"ROC plot available: {dest}")
            break
    else:
        print("No saved ROC plot found; a real ROC plot requires per-pair scores or ROC coordinates.")
if __name__ == "__main__":
    main()
