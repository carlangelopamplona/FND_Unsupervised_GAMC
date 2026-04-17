import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_summary_df() -> pd.DataFrame:
    out_dir = Path(__file__).resolve().parent
    classwise_path = out_dir / "gamc_classwise_results.csv"
    if not classwise_path.exists():
        raise FileNotFoundError(f"Missing source metrics file: {classwise_path}")

    src = pd.read_csv(classwise_path)
    df = pd.DataFrame(
        {
            "Dataset": src["Dataset"].str.lower(),
            "Accuracy": src["Accuracy"],
            "Precision": (src["True Precision"] + src["Fake Precision"]) / 2.0,
            "Recall": (src["True Recall"] + src["Fake Recall"]) / 2.0,
            "F1-score": (src["True F1-score"] + src["Fake F1-score"]) / 2.0,
        }
    )
    return df


def save_markdown_table(df: pd.DataFrame, out_md: Path) -> None:
    disp = df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        disp[col] = disp[col].map(lambda x: f"{x:.4f}")
    out_md.write_text(disp.to_markdown(index=False), encoding="utf-8")


def save_table_figure(df: pd.DataFrame, out_png: Path) -> None:
    disp = df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        disp[col] = disp[col].map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(9.4, 2.4))
    ax.axis("off")
    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    fig.suptitle("GAMC Performance by Dataset", fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_bar_figure(df: pd.DataFrame, out_png: Path) -> None:
    datasets = df["Dataset"].tolist()
    means = df["Accuracy"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(datasets, means)
    ax.set_ylim(0.80, 0.90)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Dataset")

    for rect, val in zip(bars, means):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            val + 0.0015,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_seed_boxplot(out_png: Path) -> None:
    # Keep this artifact for compatibility with existing outputs.
    # It now serves as a compact table image with the four requested metrics.
    df = build_summary_df()
    save_table_figure(df, out_png)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    os.makedirs(out_dir, exist_ok=True)

    df = build_summary_df()
    csv_path = out_dir / "results_summary.csv"
    md_path = out_dir / "results_summary.md"
    table_png = out_dir / "results_table.png"
    bar_png = out_dir / "results_final_acc_bar.png"
    box_png = out_dir / "results_seed_boxplot.png"

    df.to_csv(csv_path, index=False)
    save_markdown_table(df, md_path)
    save_table_figure(df, table_png)
    save_bar_figure(df, bar_png)
    save_seed_boxplot(box_png)

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {table_png}")
    print(f"Saved: {bar_png}")
    print(f"Saved: {box_png}")


if __name__ == "__main__":
    main()
