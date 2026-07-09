from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import json
import torch.nn.functional as F
import xarray as xr

from lightning_modelling.common_path import DATASET_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALER_PATH = DATASET_PATH / "scaler" / "scaler_full.pkl"
CLIMATOLOGY_PATH = DATASET_PATH / "climatology"

DATES = pd.date_range(start="2008-01-02", end="2023-12-31", freq="d")


def get_climatology_maps():
    climatology_maps = {}
    for season in ["winter", "spring", "summer", "autumn"]:
        lightning_clim_ds = xr.open_dataset(
            CLIMATOLOGY_PATH / f"{season}_climatology_training.nc"
        )
        climatology_map = torch.tensor(
            lightning_clim_ds["lightnings"].values, dtype=torch.float32
        )
        climatology_map = torch.flip(climatology_map, dims=[0])
        climatology_maps[season] = climatology_map.to(DEVICE)
    return climatology_maps


class StreamingAUC:
    def __init__(self, n_bins=1000):
        self.n_bins = n_bins
        self.pos_hist = np.zeros(n_bins)
        self.neg_hist = np.zeros(n_bins)
        self.bin_edges = np.linspace(0, 1, n_bins + 1)
        self.roc_auc = 0.0
        self.ap = 0.0

    def update(self, y_true, y_probs):
        # y_probs assumed to be sigmoid output, between 0 and 1
        y_true = y_true.detach().cpu().numpy()
        y_probs = y_probs.detach().cpu().numpy()
        bin_indices = np.digitize(y_probs, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        for i in range(len(y_true)):
            if y_true[i] == 1:
                self.pos_hist[bin_indices[i]] += 1
            else:
                self.neg_hist[bin_indices[i]] += 1

    def compute(self):
        self.tpr = (
            np.cumsum(self.pos_hist[::-1]) / self.pos_hist.sum()
            if self.pos_hist.sum() > 0
            else np.zeros_like(self.pos_hist)
        )
        self.fpr = (
            np.cumsum(self.neg_hist[::-1]) / self.neg_hist.sum()
            if self.neg_hist.sum() > 0
            else np.zeros_like(self.neg_hist)
        )
        self.precisions = np.cumsum(self.pos_hist[::-1]) / (
            np.cumsum(self.pos_hist[::-1]) + np.cumsum(self.neg_hist[::-1]) + 1e-10
        )

        # get the first index where tpr, fpr and precisions are not zero
        if (
            np.any(self.tpr > 0)
            and np.any(self.fpr > 0)
            and np.any(self.precisions > 0)
        ):
            self.first_nonzero_index = np.min(
                [
                    np.argmax(self.tpr > 0),
                    np.argmax(self.fpr > 0),
                    np.argmax(self.precisions > 0),
                ]
            )
        else:
            self.first_nonzero_index = 0
        self.tpr = self.tpr[self.first_nonzero_index :]
        self.fpr = self.fpr[self.first_nonzero_index :]
        self.precisions = self.precisions[self.first_nonzero_index :]

        self.tpr = np.concatenate([[0], self.tpr])
        self.fpr = np.concatenate([[0], self.fpr])
        self.precisions = np.concatenate([[self.precisions[1]], self.precisions])

        self.ap = np.trapezoid(self.precisions, self.tpr)
        self.roc_auc = np.trapezoid(self.tpr, self.fpr)


class Dice(nn.Module):
    """Streaming Dice (a.k.a. F1 / Sørensen-Dice) coefficient over a binary map.

    Accumulates the intersection and the two cardinalities across batches, then
    computes ``Dice = 2 * (I + s) / (P + T - I + s)`` where ``I`` is the
    intersection, ``P``/``T`` the predicted/target cardinalities, and ``s`` a
    smoothing term to avoid division by zero. Note this is the Dice coefficient,
    not the Jaccard index (IoU): the leading factor of 2 (and the ``- I`` in the
    denominator) is what distinguishes Dice from IoU.
    """

    def __init__(self, smooth=1e-6):
        super(Dice, self).__init__()
        self.smooth = smooth
        self.pred_card = 0
        self.target_card = 0
        self.intersection = 0
        self.dice = 0

    def update(self, pred, target):
        self.intersection += (pred * target).sum()
        self.pred_card += pred.sum()
        self.target_card += target.sum()

    def compute(self):
        self.dice = (
            2
            * (self.intersection + self.smooth)
            / (self.pred_card + self.target_card - self.intersection + self.smooth)
        ).item()


class DeterMetrics:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        # Final metrics
        self.precision = 0.0
        self.recall = 0.0
        self.accuracy = 0.0
        self.f1 = 0.0
        self.ets = 0.0
        # Total predictd lightning maps
        self.total_obs = np.zeros((101, 149))
        self.total_pred = np.zeros((101, 149))

    def update_tfpn(self, y_true, y_pred):
        self.tp += ((y_pred == 1) & (y_true == 1)).sum().item()
        self.fp += ((y_pred == 1) & (y_true == 0)).sum().item()
        self.fn += ((y_pred == 0) & (y_true == 1)).sum().item()
        self.tn += ((y_pred == 0) & (y_true == 0)).sum().item()

    def update_total_maps(self, obs_maps_batch, pred_maps_batch):
        self.total_obs += obs_maps_batch.sum(axis=0).detach().cpu().numpy()
        self.total_pred += pred_maps_batch.sum(axis=0).detach().cpu().numpy()

    def update(self, y_true, y_pred):
        self.update_tfpn(y_true.flatten(), y_pred.flatten() > 0.5)
        self.update_total_maps(y_true, y_pred)

    def compute(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        self.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self.accuracy = (
            (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        )
        self.f1 = (
            2 * (self.precision * self.recall) / (self.precision + self.recall)
            if (self.precision + self.recall) > 0
            else 0.0
        )
        tr = (
            (tp + fp) * (tp + fn) / (tp + fp + fn + tn)
            if (tp + fp + fn + tn) > 0
            else 0.0
        )
        self.ets = (tp - tr) / (tp + fp + fn - tr) if (tp + fp + fn - tr) > 0 else 0.0


class FractionalScores:
    def __init__(self, kernel_size=1, n_batches=1, climatology_maps=None):
        # Hyperparameters
        self.n_batches = n_batches
        self.kernel_size = kernel_size
        self.climatology_maps = climatology_maps
        # Fractional scores
        self.squared_diff = 0.0
        self.squared_diff_baseline = 0.0
        self.fbs = 0.0
        self.fbss = 0.0

    def update(self, y_true, y_pred, season=None):
        window = 2 * self.kernel_size + 1
        random_pred = self.climatology_maps[season].expand(
            y_pred.shape[0], *self.climatology_maps[season].shape
        )
        y_fractions = F.avg_pool2d(
            y_true.float(), kernel_size=window, stride=1, padding=self.kernel_size
        )
        p_fractions = F.avg_pool2d(
            y_pred.float(), kernel_size=window, stride=1, padding=self.kernel_size
        )
        p_fraction_baseline = F.avg_pool2d(
            random_pred.float(), kernel_size=window, stride=1, padding=self.kernel_size
        ).squeeze()
        self.squared_diff += (
            (p_fractions - y_fractions) ** 2
        ).mean().item() / self.n_batches
        self.squared_diff_baseline += (
            (p_fraction_baseline - y_fractions) ** 2
        ).mean().item() / self.n_batches

    def compute(self):
        self.fbs = self.squared_diff
        self.fbss = (
            1 - self.squared_diff / self.squared_diff_baseline
            if self.squared_diff_baseline > 0
            else 0.0
        )


class DevianceScore:
    def __init__(self, climatology_maps, reduction="mean", n_batches=1):
        self.dev_hat = 0.0
        self.dev_null = 0.0
        self.score = 0.0
        self.climatology_maps = climatology_maps
        self.reduction = reduction
        self.n_batches = n_batches
        # Log loss
        self.log_losses_hat = []
        self.log_losses_null = []

    def update(self, y_true, y_pred_probas, batch_size, season=None):
        # random_pred is a tensor of same shape as y_true filled with the constant value self.random_proba
        # random_pred = torch.ones_like(y_true) * self.random_proba
        random_pred = (
            self.climatology_maps[season]
            .expand(batch_size, *self.climatology_maps[season].shape)
            .flatten()
        )

        self.log_losses_hat.append(
            F.binary_cross_entropy(
                y_pred_probas, y_true, reduction=self.reduction
            ).item()
        )

        self.log_losses_null.append(
            F.binary_cross_entropy(random_pred, y_true, reduction=self.reduction).item()
        )

    def compute(self):
        if self.reduction == "mean":
            log_loss_hat = sum(self.log_losses_hat) / self.n_batches
            log_loss_null = sum(self.log_losses_null) / self.n_batches
        else:
            log_loss_hat = sum(self.log_losses_hat)
            log_loss_null = sum(self.log_losses_null)
        self.dev_hat = 2 * log_loss_hat
        self.dev_null = 2 * log_loss_null
        self.score = 1 - self.dev_hat / self.dev_null if self.dev_null > 0 else 0.0


class Metrics:
    def __init__(
        self,
        save_path,
        kernel_size=1,
        reduction="mean",
        n_batches=1,
        auc_bins=1000,
        extremes=False,
    ):
        self.save_path = save_path
        self.extremes = extremes
        self.climatology_maps = get_climatology_maps()
        # Deterministic metrics
        self.deter_metrics = DeterMetrics()
        # ROC and PR AUC
        self.auc_calc = StreamingAUC(n_bins=auc_bins)
        # Fractional scores
        self.dice = Dice()
        self.fractional_scores = FractionalScores(
            kernel_size=kernel_size,
            n_batches=n_batches,
            climatology_maps=self.climatology_maps,
        )
        # Deviance score
        self.deviance_score = DevianceScore(
            climatology_maps=self.climatology_maps,
            reduction=reduction,
            n_batches=n_batches,
        )

    def update_all(self, y_true, y_pred, season=None):
        self.deter_metrics.update(y_true, y_pred)
        self.dice.update(y_pred.flatten(), y_true.flatten())
        self.auc_calc.update(y_true.flatten(), y_pred.flatten())
        self.fractional_scores.update(
            y_true.unsqueeze(1), y_pred.unsqueeze(1), season=season
        )
        self.deviance_score.update(
            y_true.flatten(), y_pred.flatten(), y_true.shape[0], season=season
        )

    def compute_all(
        self,
    ):
        self.deter_metrics.compute()
        self.auc_calc.compute()
        self.dice.compute()
        self.fractional_scores.compute()
        self.deviance_score.compute()

    def print(self):
        print("Metrics Summary:")
        print("-" * 40)

        print("    Deterministic Metrics:")
        print(
            f"Precision: {self.deter_metrics.precision:.3f}                | Recall: {self.deter_metrics.recall:.5f}"
        )
        print(
            f"F1: {self.deter_metrics.f1:.5f}                     | Accuracy: {self.deter_metrics.accuracy:.3f}"
        )
        print(f"ETS: {self.deter_metrics.ets:.5f}")
        print("-" * 40)

        print("    Probabilistic Metrics:")
        print(
            f"ROC AUC: {self.auc_calc.roc_auc:.3f}                  | Average Precision: {self.auc_calc.ap:.4f}"
        )
        print(
            f"Deviance: {self.deviance_score.score:.3f}                 | Dice: {self.dice.dice:.5f}"
        )
        print(
            f"Fractional Brier Score: {self.fractional_scores.fbs:.5f} | Fractional Brier Skill Score: {self.fractional_scores.fbss:.5f}"
        )
        print("-" * 40)

        print("    Total Counts:")
        print(f"Total observations: {self.deter_metrics.total_obs.sum()}")
        print(f"Total predictions: {self.deter_metrics.total_pred.sum()}")

    def save(self: Path):
        metrics_json = {
            "precision": self.deter_metrics.precision,
            "recall": self.deter_metrics.recall,
            "f1": self.deter_metrics.f1,
            "accuracy": self.deter_metrics.accuracy,
            "ets": self.deter_metrics.ets,
            "roc_auc": self.auc_calc.roc_auc,
            "average_precision": self.auc_calc.ap,
            "deviance_score": self.deviance_score.score,
            "Dice": self.dice.dice,
            "Fractional_Brier_Score": self.fractional_scores.fbs,
            "Fractional_Brier_Skill_Score": self.fractional_scores.fbss,
            "total_obs": self.deter_metrics.total_obs.sum(),
            "total_pred": self.deter_metrics.total_pred.sum(),
            "tpr_binned": self.auc_calc.tpr.tolist(),
            "fpr_binned": self.auc_calc.fpr.tolist(),
            "precision_binned": self.auc_calc.precisions.tolist(),
        }

        file_name = "metrics_extremes.json" if self.extremes else "metrics.json"
        with open(self.save_path / file_name, "w") as f:
            json.dump(metrics_json, f, indent=4)

    def plot_roc_pr_curves(self):
        print(
            f"Number of bins for ROC and AP curves : TPR {len(self.auc_calc.tpr)}, FPR {len(self.auc_calc.fpr)}, Precision {len(self.auc_calc.precisions)}"
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(
            self.auc_calc.fpr,
            self.auc_calc.tpr,
            label=f"ROC-AUC = {self.auc_calc.roc_auc:.3f}",
        )
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(
            self.auc_calc.tpr,
            self.auc_calc.precisions,
            label=f"Average Precision = {self.auc_calc.ap:.3f}",
        )
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall (PR) Curve")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        file_name = (
            "roc_pr_curves_extremes.png" if self.extremes else "roc_pr_curves.png"
        )
        fig.savefig(self.save_path / file_name, dpi=300)
        plt.close(fig)
