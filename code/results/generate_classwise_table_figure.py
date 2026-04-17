import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from gamc.datasets.data_util import load_fake_news_graph_dataset
from gamc.models import build_model
from gamc.utils import create_optimizer, load_best_configs, set_random_seed
from main_new import pretrain, resolve_amd_device


# Fast mode keeps the same evaluation flow but shortens pretraining to finish quickly.
FAST_MAX_EPOCH = 5
FAST_BATCH_SIZE = 64


def get_default_args():
    class Args:
        pass

    args = Args()
    args.seeds = [0]
    args.dataset = "politifact"
    args.feature = "bert"
    args.device = -1
    args.max_epoch = 300
    args.warmup_steps = -1
    args.num_heads = 4
    args.num_out_heads = 1
    args.num_layers = 2
    args.num_hidden = 256
    args.residual = False
    args.in_drop = 0.2
    args.attn_drop = 0.1
    args.norm = None
    args.lr = 0.005
    args.weight_decay = 5e-4
    args.negative_slope = 0.2
    args.activation = "prelu"
    args.mask_rate = 0.5
    args.drop_edge_rate = 0.0
    args.replace_rate = 0.0
    args.encoder = "gat"
    args.decoder = "gat"
    args.loss_fn = "mse"
    args.alpha_l = 2
    args.optimizer = "adam"
    args.max_epoch_f = 30
    args.lr_f = 0.001
    args.weight_decay_f = 0.0
    args.linear_prob = True
    args.load_model = False
    args.save_model = False
    args.use_cfg = True
    args.logging = False
    args.scheduler = False
    args.concat_hidden = False
    args.pooling = "mean"
    args.deg4feat = False
    args.batch_size = 32
    return args


def get_embeddings(model, dataloader, device):
    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for batch_g in dataloader:
            batch_g = batch_g.to(device)
            out = model.embed(batch_g.x, batch_g.edge_index)
            out = global_mean_pool(out, batch_g.batch)
            x_list.append(out.cpu().numpy())
            y_list.append(batch_g.y.cpu().numpy())

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return x, y


def evaluate_classwise_metrics(embeddings, labels):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    accs = []
    true_p, true_r, true_f = [], [], []
    fake_p, fake_r, fake_f = [], [], []

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]

        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        clf = GridSearchCV(SVC(random_state=42), params)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)

        accs.append(accuracy_score(y_test, preds))

        p, r, f, _ = precision_recall_fscore_support(
            y_test,
            preds,
            labels=[0, 1],
            average=None,
            zero_division=0,
        )

        true_p.append(p[0])
        true_r.append(r[0])
        true_f.append(f[0])
        fake_p.append(p[1])
        fake_r.append(r[1])
        fake_f.append(f[1])

    return {
        "Accuracy": float(np.mean(accs)),
        "True Precision": float(np.mean(true_p)),
        "True Recall": float(np.mean(true_r)),
        "True F1-score": float(np.mean(true_f)),
        "Fake Precision": float(np.mean(fake_p)),
        "Fake Recall": float(np.mean(fake_r)),
        "Fake F1-score": float(np.mean(fake_f)),
    }


def run_dataset(dataset_name, device):
    args = get_default_args()
    args.dataset = dataset_name
    args.device = device
    args = load_best_configs(args, "configs.yml")
    args.max_epoch = min(args.max_epoch, FAST_MAX_EPOCH)
    args.batch_size = FAST_BATCH_SIZE

    set_random_seed(0)

    dataset, (num_features, num_classes) = load_fake_news_graph_dataset(
        args.dataset, args.feature, deg4feat=args.deg4feat
    )
    args.num_features = num_features

    train_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args)
    model.to(device)

    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = None

    model = pretrain(
        model,
        args.pooling,
        (train_loader, eval_loader),
        optimizer,
        args.max_epoch,
        device,
        scheduler,
        num_classes,
        args.lr_f,
        args.weight_decay_f,
        args.max_epoch_f,
        args.linear_prob,
        logger=None,
    )

    model = model.to(device)
    emb, labels = get_embeddings(model, eval_loader, device)
    metrics = evaluate_classwise_metrics(emb, labels)
    return metrics


def save_outputs(df, out_dir):
    csv_path = out_dir / "gamc_classwise_results.csv"
    md_path = out_dir / "gamc_classwise_results.md"
    png_path = out_dir / "gamc_classwise_results_table.png"

    df.to_csv(csv_path, index=False)

    disp = df.copy()
    for col in disp.columns:
        if col != "Dataset":
            disp[col] = disp[col].map(lambda x: f"{x:.4f}")
    md_path.write_text(disp.to_markdown(index=False), encoding="utf-8")

    plt.rcParams["font.family"] = "DejaVu Serif"
    fig, ax = plt.subplots(figsize=(16, 3.8))
    ax.axis("off")

    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.1, 1.8)

    ax.set_title("GAMC: Class-wise Results per Dataset", fontsize=20, pad=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {png_path}")


def main():
    # Ensure relative paths like configs.yml and ./data resolve under GAMC/code.
    code_dir = Path(__file__).resolve().parents[1]
    os.chdir(code_dir)

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = os.environ.get("GAMC_GPU_BACKEND", "auto")
    device = resolve_amd_device(0, backend)
    print(f"Using backend={backend} on device={device}")
    records = []
    for dataset in ["politifact", "gossipcop"]:
        print(f"Running dataset: {dataset}")
        metrics = run_dataset(dataset, device)
        record = {"Dataset": dataset.capitalize()}
        record.update(metrics)
        records.append(record)

    df = pd.DataFrame(records)
    save_outputs(df, out_dir)


if __name__ == "__main__":
    main()
