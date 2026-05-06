"""Bayesian-optimisation trajectory variance for regression.

For each regression dataset (ESOL, FreeSolv, Lipo) we run K independent
BO trajectories per method (ERM and twin-bootstrap lambda=3, the
regression value the paper uses).  All K trajectories of a given method
share the same random initial-N labelled subset; they differ only in
the in-loop training-data bootstraps -- the same source of variance the
paper measures elsewhere.

At each step the surrogate is retrained from scratch on the currently-
labelled subset, predicts y on the unlabelled remainder, and acquires
the argmax-predicted molecule (greedy top-1 maximisation -- chosen
because ERM has no per-prediction uncertainty, so giving twin the
disagreement-as-UCB bonus would bias the comparison).  The oracle reveals
the molecule's actual y; it joins the labelled subset.

The claim to test: std(best_so_far_final) across the K trajectories
should be smaller for twin-bootstrap than for ERM -- the cross-sample
churn reduction the paper reports translates to more reproducible BO
acquisition trajectories.

Reads: outputs/cross_sample/<dataset>/erm_train1.npz (just to find the
dataset's training pool via the canonical seed protocol), or directly
constructs (X, y) tensors from MolNetDataset.

Writes: outputs/bo_loop_regression.json (per-trajectory traces) and
outputs/bo_loop_regression.csv (summary).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = [
    ("esol_reg", "ESOL", "esol"),
    ("freesolv_reg", "FreeSolv", "freesolv"),
    ("lipo_reg", "Lipo", "lipo"),
]
METHODS = ["erm", "bagging", "twin"]
LAMBDA_REG = 3.0       # regression twin-bootstrap value used in the paper
K_BAG = 5              # bagging head count to mirror the classification BO appendix
EPOCHS = 30
BATCH = 64
LR = 1e-3
WD = 1e-4


# ---------------------------------------------------------------------------
# Model + training utilities (self-contained: do not import the chemistry-
# pipeline trainer because that one expects a DataLoader-shaped dataset).
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_indices(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, n, (n,), generator=g)


class MLP(nn.Module):
    """256-2L MLP regressor (matches the paper's regression backbone)."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_one_mlp(X: torch.Tensor, y: torch.Tensor, seed: int) -> nn.Module:
    set_seed(seed)
    model = MLP(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    idx = bootstrap_indices(X.shape[0], seed)
    Xb = X[idx].to(DEVICE)
    yb = y[idx].to(DEVICE).float()
    n = Xb.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, BATCH):
            sel = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.mse_loss(model(Xb[sel]), yb[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def train_erm_reg(X: torch.Tensor, y: torch.Tensor, seed: int):
    """Train one ERM regressor; return predict(X_test) -> np.ndarray."""
    model = _train_one_mlp(X, y, seed)

    @torch.no_grad()
    def predict(X_test: torch.Tensor) -> np.ndarray:
        return model(X_test.to(DEVICE)).cpu().numpy()

    return predict


def train_bagging_reg(X: torch.Tensor, y: torch.Tensor, seed: int,
                      K: int = K_BAG):
    """Train K-bootstrap bagging regressor (K independent ERM heads).

    Each head is an ERM MLP trained on its own independent bootstrap;
    deployment-time prediction is the mean across the K heads.
    Mirrors the bagging protocol used elsewhere in the paper.
    """
    models = [_train_one_mlp(X, y, seed=seed * 100 + k) for k in range(K)]

    @torch.no_grad()
    def predict(X_test: torch.Tensor) -> np.ndarray:
        Xt = X_test.to(DEVICE)
        return np.mean([m(Xt).cpu().numpy() for m in models], axis=0)

    return predict


def train_twin_reg(X: torch.Tensor, y: torch.Tensor, seed: int,
                   lam: float = LAMBDA_REG):
    """Train twin-bootstrap regression pair (independent bootstraps, MSE
    consistency on the union of mini-batches).

    The MSE consistency mirrors the regression branch of cross_sample_train
    `_train_twin_indep`.  Returns predict(X_test) -> averaged np.ndarray.
    """
    set_seed(seed)
    model_a = MLP(X.shape[1]).to(DEVICE)
    set_seed(seed + 1)
    model_b = MLP(X.shape[1]).to(DEVICE)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=LR, weight_decay=WD)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=LR, weight_decay=WD)
    n = X.shape[0]
    idx_a = bootstrap_indices(n, seed)
    idx_b = bootstrap_indices(n, seed + 1)
    Xa = X[idx_a].to(DEVICE); ya = y[idx_a].to(DEVICE).float()
    Xb = X[idx_b].to(DEVICE); yb = y[idx_b].to(DEVICE).float()
    for _ in range(EPOCHS):
        perm_a = torch.randperm(n)
        perm_b = torch.randperm(n)
        model_a.train(); model_b.train()
        for i in range(0, n, BATCH):
            sa = perm_a[i:i + BATCH]
            sb = perm_b[i:i + BATCH]
            opt_a.zero_grad(); opt_b.zero_grad()
            # Cross-entropy analogue for regression: MSE on each network's bootstrap.
            la_a = model_a(Xa[sa]); lb_b = model_b(Xb[sb])
            sup = F.mse_loss(la_a, ya[sa]) + F.mse_loss(lb_b, yb[sb])
            # Consistency: MSE between the two networks on the union of mini-batches.
            la_b = model_a(Xb[sb]); lb_a = model_b(Xa[sa])
            cons = 0.5 * (F.mse_loss(la_a, lb_a) + F.mse_loss(la_b, lb_b))
            (sup + lam * cons).backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
            opt_a.step(); opt_b.step()
    model_a.eval(); model_b.eval()

    @torch.no_grad()
    def predict(X_test: torch.Tensor) -> np.ndarray:
        Xt = X_test.to(DEVICE)
        pa = model_a(Xt).cpu().numpy()
        pb = model_b(Xt).cpu().numpy()
        return 0.5 * (pa + pb)

    return predict


# ---------------------------------------------------------------------------
# BO trajectory
# ---------------------------------------------------------------------------

def run_trajectory(X: torch.Tensor, y: np.ndarray, method: str,
                   trajectory_id: int, initial_subset: np.ndarray,
                   budget: int, lam: float = LAMBDA_REG):
    """One BO trajectory.

    All K trajectories of a given method share `initial_subset`; they
    diverge only because the in-loop retrainings use different bootstrap
    seeds.  Returns a dict with the acquired indices and best-so-far at
    each step.
    """
    n = len(y)
    labelled = list(int(i) for i in initial_subset)
    labelled_set = set(labelled)
    acquired: list[int] = []
    best_so_far: list[float] = []

    for step in range(budget):
        unlabelled = np.array([i for i in range(n) if i not in labelled_set],
                              dtype=np.int64)
        if len(unlabelled) == 0:
            break
        # Trajectory-and-step-keyed seed so each retraining sees a different
        # bootstrap of the labelled subset.
        seed = trajectory_id * 1_000_000 + step
        Xl = X[labelled]
        yl = torch.tensor(y[labelled], dtype=torch.float32)
        if method == "erm":
            predict = train_erm_reg(Xl, yl, seed=seed)
        elif method == "bagging":
            predict = train_bagging_reg(Xl, yl, seed=seed, K=K_BAG)
        elif method == "twin":
            predict = train_twin_reg(Xl, yl, seed=seed, lam=lam)
        else:
            raise ValueError(f"unknown method {method}")
        preds = predict(X[unlabelled])
        next_idx = int(unlabelled[int(np.argmax(preds))])
        labelled.append(next_idx)
        labelled_set.add(next_idx)
        acquired.append(next_idx)
        best_so_far.append(float(max(y[acquired])))

    return {
        "trajectory_id": int(trajectory_id),
        "method": method,
        "initial_subset": [int(i) for i in initial_subset.tolist()],
        "acquired": [int(i) for i in acquired],
        "best_so_far": [float(v) for v in best_so_far],
        "final_best": float(best_so_far[-1]) if best_so_far else float("nan"),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_xy(short_name: str, canonical_seed: int = 99,
            data_dir: str = "./data/molnet"):
    """Load Morgan fingerprints and regression targets for one dataset.

    Returns (X, y, name) where X is float32 torch.Tensor [N, 2048],
    y is float64 np.ndarray [N], and name is the dataset short label.
    """
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from data_molnet import MolNetDataset
    ds = MolNetDataset(name=short_name, split="train",
                       seed=canonical_seed, data_dir=data_dir,
                       regression=True)
    X = ds.images.float()
    y = ds.labels.numpy().astype(np.float64)
    return X, y


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--K", type=int, default=20,
                    help="Number of independent BO trajectories per method.")
    ap.add_argument("--initial_n", type=int, default=50,
                    help="Initial random labelled-subset size (shared across trajectories).")
    ap.add_argument("--budget", type=int, default=10,
                    help="BO acquisitions after the initial subset.")
    ap.add_argument("--canonical_seed", type=int, default=99,
                    help="Canonical seed pinning the train/test split (training pool = candidate library).")
    ap.add_argument("--initial_seed", type=int, default=42,
                    help="Seed for the initial-N random subset (shared across trajectories of a method).")
    ap.add_argument("--lam", type=float, default=LAMBDA_REG)
    ap.add_argument("--datasets", nargs="+", default=[d[0] for d in DATASETS])
    ap.add_argument("--out_json", default="outputs/bo_loop_regression.json")
    ap.add_argument("--out_csv", default="outputs/bo_loop_regression.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.initial_seed)
    all_records: list[dict] = []

    for short_name, label, molnet_short in DATASETS:
        if short_name not in args.datasets:
            continue
        print(f"\n=== {label} ({short_name}) ===")
        X, y = load_xy(molnet_short, canonical_seed=args.canonical_seed)
        n = len(y)
        y_min, y_max = float(y.min()), float(y.max())
        # Same initial subset for all trajectories of every method on this dataset.
        initial_subset = rng.choice(n, size=args.initial_n, replace=False)
        print(f"  N={n}, initial_n={args.initial_n}, budget={args.budget}, "
              f"y range [{y_min:.2f}, {y_max:.2f}]")

        for method in METHODS:
            for k in range(args.K):
                trajectory_id = k
                rec = run_trajectory(
                    X, y, method=method, trajectory_id=trajectory_id,
                    initial_subset=initial_subset, budget=args.budget,
                    lam=args.lam,
                )
                rec["dataset"] = short_name
                rec["dataset_label"] = label
                rec["y_min"] = y_min
                rec["y_max"] = y_max
                all_records.append(rec)
                print(f"  {method:7s} traj={k:2d}  final_best={rec['final_best']:.3f}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(all_records, indent=2))
    print(f"\nWrote {args.out_json}")

    # Summary CSV: per (dataset, method), mean and std of final_best across trajectories.
    summary_rows: list[dict] = []
    for short_name, label, _ in DATASETS:
        if short_name not in args.datasets:
            continue
        for method in METHODS:
            finals = [r["final_best"] for r in all_records
                      if r["dataset"] == short_name and r["method"] == method]
            if not finals:
                continue
            summary_rows.append({
                "dataset": short_name,
                "dataset_label": label,
                "method": method,
                "n_trajectories": len(finals),
                "final_best_mean": float(np.mean(finals)),
                "final_best_std": float(np.std(finals, ddof=1)),
                "final_best_min": float(np.min(finals)),
                "final_best_max": float(np.max(finals)),
            })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Wrote {args.out_csv}")

    # Print a compact comparison so the user sees the result in-terminal.
    print("\n=== summary: mean ± std (n traj) ===")
    print(f"{'dataset':<12}{'method':<8}{'final_best mean':>17}{'std':>9}")
    for row in summary_rows:
        print(f"{row['dataset_label']:<12}{row['method']:<8}"
              f"{row['final_best_mean']:>17.3f}{row['final_best_std']:>9.3f}")


if __name__ == "__main__":
    main()
