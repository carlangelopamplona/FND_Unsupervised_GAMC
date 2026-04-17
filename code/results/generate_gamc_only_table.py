from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Updated GAMC results from executed runs.
# Micro-F1 from evaluation equals accuracy for single-label classification.
DATA = [
    {
        "Dataset": "Politifact",
        "Method": "GAMC",
        "Accuracy (Micro-F1)": 0.8576,
        "Std": 0.0102,
        "Runs": 9,
    },
    {
        "Dataset": "GossipCop",
        "Method": "GAMC",
        "Accuracy (Micro-F1)": 0.8760,
        "Std": 0.0050,
        "Runs": 9,
    },
]


def save_markdown(df: pd.DataFrame, out_path: Path) -> None:
    disp = df.copy()
    disp["Accuracy (Micro-F1)"] = disp["Accuracy (Micro-F1)"].map(lambda x: f"{x:.4f}")
    disp["Std"] = disp["Std"].map(lambda x: f"{x:.4f}")
    out_path.write_text(disp.to_markdown(index=False), encoding="utf-8")


def draw_single_dataset_table(ax, row: pd.Series, title: str) -> None:
    ax.axis("off")

    columns = ["Methods", "Accuracy", "Std", "Runs"]
    values = [[
        row["Method"],
        f"{row['Accuracy (Micro-F1)']:.4f}",
        f"{row['Std']:.4f}",
        f"{int(row['Runs'])}",
    ]]

    table = ax.table(
        cellText=values,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.2, 2.0)

    # Bold GAMC row values.
    for col in range(len(columns)):
        table[(1, col)].set_text_props(weight="bold")

    ax.set_title(title, fontsize=20, pad=10)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(DATA)

    md_path = out_dir / "gamc_only_results.md"
    csv_path = out_dir / "gamc_only_results.csv"
    png_path = out_dir / "gamc_only_results_table.png"

    save_markdown(df, md_path)
    df.to_csv(csv_path, index=False)

    plt.rcParams["font.family"] = "DejaVu Serif"
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    row_pol = df[df["Dataset"] == "Politifact"].iloc[0]
    row_gos = df[df["Dataset"] == "GossipCop"].iloc[0]

    draw_single_dataset_table(axes[0], row_pol, "Table 1: GAMC updated result on Politifact dataset")
    draw_single_dataset_table(axes[1], row_gos, "Table 2: GAMC updated result on GossipCop dataset")

    fig.tight_layout(h_pad=2.5)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {md_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
