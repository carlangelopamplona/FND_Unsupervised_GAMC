from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    src_csv = out_dir / "ablation_summary.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"Missing source file: {src_csv}")

    df = pd.read_csv(src_csv)
    if df.empty:
        raise ValueError("ablation_summary.csv is empty")

    disp = df.copy()
    disp["Dataset"] = disp["Dataset"].astype(str).str.capitalize()
    disp["Accuracy"] = disp["Accuracy"].map(lambda x: f"{x:.4f}")
    disp["F1-score"] = disp["F1-score"].map(lambda x: f"{x:.4f}")
    disp["Std"] = disp["Std"].map(lambda x: f"{x:.4f}")

    out_md = out_dir / "ablation_summary_table.md"
    out_png = out_dir / "ablation_summary_table.png"

    out_md.write_text(disp.to_markdown(index=False), encoding="utf-8")

    plt.rcParams["font.family"] = "DejaVu Serif"
    fig, ax = plt.subplots(figsize=(11.2, 4.0))
    ax.axis("off")

    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.05, 1.65)

    ax.set_title("GAMC Ablation Results", fontsize=18, pad=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_md}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
