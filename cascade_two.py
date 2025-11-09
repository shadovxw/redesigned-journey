
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("output") / "cascade_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_vars_needed = ["val_files","y_val","p1_val","p2_val","best_and","best_or",
                "and_pred_test","or_pred_test","test_files","y_test","p1_test","p2_test"]
missing = [v for v in _vars_needed if v not in globals()]
if missing:
    print("Warning: some expected variables are missing from the notebook:", missing)
    print("If they truly aren't present, re-run the prediction/grid-search cell first.")
else:
    # Define a corrected save_per_sample that accepts per-row arrays of final preds
    def save_per_sample(files, y, p1, p2, final_and_arr=None, final_or_arr=None, prefix="val"):
        rows = []
        n = len(files)

        p1 = np.array(p1).reshape(-1)
        p2 = np.array(p2).reshape(-1)
        y = np.array(y).reshape(-1)
        if final_and_arr is not None:
            final_and_arr = np.array(final_and_arr).reshape(-1)
        if final_or_arr is not None:
            final_or_arr = np.array(final_or_arr).reshape(-1)
        for i, (fp, yt, a, b) in enumerate(zip(files, y, p1, p2)):
            row = {
                "filepath": fp,
                "true_label": int(yt),
                "prob_stage1": float(a),
                "prob_stage2": float(b),
            }
            if final_and_arr is not None:
                row["and_final"] = int(final_and_arr[i])
            else:
                row["and_final"] = None
            if final_or_arr is not None:
                row["or_final"] = int(final_or_arr[i])
            else:
                row["or_final"] = None
            rows.append(row)
        df = pd.DataFrame(rows)
        out_path = OUTPUT_DIR / f"per_sample_{prefix}.csv"
        df.to_csv(out_path, index=False)
        return out_path

    if "val_and_pred" not in globals():
        if 'evaluate_and_rule' in globals() and 'evaluate_or_rule' in globals():
            _, val_and_pred = evaluate_and_rule(y_val, p1_val, p2_val, best_and[0], best_and[1])
            _, val_or_pred = evaluate_or_rule(y_val, p1_val, p2_val, best_or[0], best_or[1])
        else:
            val_and_pred = np.zeros(len(val_files), dtype=int)
            val_or_pred = np.zeros(len(val_files), dtype=int)
    else:
        val_and_pred = globals()["val_and_pred"]
        val_or_pred = globals()["val_or_pred"]

    p_val = save_per_sample(val_files, y_val, p1_val, p2_val, final_and_arr=val_and_pred, final_or_arr=val_or_pred, prefix="val")
    p_test = save_per_sample(test_files, y_test, p1_test, p2_test, final_and_arr=and_pred_test, final_or_arr=or_pred_test, prefix="test")
    print("Saved per-sample CSVs:")
    print(" -", p_val)
    print(" -", p_test)


    out_summary = {}
    out_summary["best_and_val"] = {"T1": float(best_and[0]), "T2": float(best_and[1])}
    out_summary["best_or_val"] = {"T1": float(best_or[0]), "T2": float(best_or[1])}

    if "best_and_metrics" in globals():
        out_summary["best_and_val"]["metrics"] = globals()["best_and_metrics"]
    if "best_or_metrics" in globals():
        out_summary["best_or_val"]["metrics"] = globals()["best_or_metrics"]
    if "and_metrics_test" in globals():
        out_summary["and_on_test"] = globals()["and_metrics_test"]
    if "or_metrics_test" in globals():
        out_summary["or_on_test"] = globals()["or_metrics_test"]
    if "meta_metrics" in globals() and globals()["meta_metrics"] is not None:
        out_summary["meta_on_test"] = globals()["meta_metrics"]
    # Save final summary JSON
    (OUTPUT_DIR / "chosen_thresholds_and_metrics.json").write_text(json.dumps(out_summary, indent=2))
    print("Wrote chosen_thresholds_and_metrics.json to", str(OUTPUT_DIR / "chosen_thresholds_and_metrics.json"))
    print("Done.")
