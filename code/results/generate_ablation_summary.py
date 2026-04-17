from pathlib import Path

import pandas as pd


ABLATION_LABELS = {
    "full": "GAMC",
    "gmac_aug": "GMAC-Aug",
    "gamc_lrec": "GAMC-Lrec",
    "gamc_lcon": "GAMC-Lcon",
}


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    files = [
        p
        for p in sorted(out_dir.glob("ablation_*.csv"))
        if "summary" not in p.name.lower()
    ]
    if not files:
        raise FileNotFoundError("No ablation_*.csv files found in results directory.")

    rows = []
    for fp in files:
        df = pd.read_csv(fp)
        mean_row = df[df["run"].astype(str) == "mean"]
        if mean_row.empty:
            continue

        r = mean_row.iloc[0]
        dataset = str(r["dataset"]).lower()
        ablation = str(r["ablation"]).lower()
        rows.append(
            {
                "Dataset": dataset,
                "Model": ABLATION_LABELS.get(ablation, ablation),
                "Accuracy": float(r["accuracy"]) if "accuracy" in r.index else float(r["f1_micro"]),
                "F1-score": float(r["f1_micro"]),
                "Std": float(r.get("std", 0.0)),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No mean rows found in ablation CSV files.")

    order_dataset = {"politifact": 0, "gossipcop": 1}
    order_model = {"GAMC": 0, "GMAC-Aug": 1, "GAMC-Lrec": 2, "GAMC-Lcon": 3}
    summary["_d"] = summary["Dataset"].map(order_dataset).fillna(99)
    summary["_m"] = summary["Model"].map(order_model).fillna(99)
    summary = summary.sort_values(["_d", "_m"]).drop(columns=["_d", "_m"]).reset_index(drop=True)

    out_csv = out_dir / "ablation_summary.csv"
    out_md = out_dir / "ablation_summary.md"
    summary.to_csv(out_csv, index=False)

    disp = summary.copy()
    disp["Accuracy"] = disp["Accuracy"].map(lambda x: f"{x:.4f}")
    disp["F1-score"] = disp["F1-score"].map(lambda x: f"{x:.4f}")
    disp["Std"] = disp["Std"].map(lambda x: f"{x:.4f}")
    out_md.write_text(disp.to_markdown(index=False), encoding="utf-8")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
