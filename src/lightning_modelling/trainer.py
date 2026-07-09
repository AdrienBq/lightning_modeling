import os
from abc import ABC
import torch
import torch.nn as nn

# from torchvision.transforms.functional import gaussian_blur
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time

from lightning_modelling.metrics import Metrics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MiB = 1024**2


def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    params = 0
    buffers = 0
    for param in model.parameters():
        params += param.nelement() * param.element_size()
    for buf in model.buffers():
        buffers += buf.nelement() * buf.element_size()
    return params, params + buffers


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        full_dataset=None,
        train_dataloader=None,
        calibration_dataloader=None,
        test_dataloader=None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.full_dataset = full_dataset
        self.train_dataloader = train_dataloader
        self.calibration_dataloader = calibration_dataloader
        self.test_dataloader = test_dataloader
        self.test_extremes_dataloader = kwargs.get("test_extremes_dataloader", None)
        self.seasons = kwargs.get("seasons", None)
        self.device = kwargs.get("device", "cpu")
        self.events_start_date = kwargs.get("events_start_date", None)
        self.events_start_hour = kwargs.get("events_start_hour", None)
        self.geopt_path = kwargs.get("geopt_path", None)
        self.data_path = kwargs.get("data_path", None)
        self.model_name = (
            self.model.name
            if hasattr(self.model, "name")
            else "model_name_not_specified"
        )
        self.eval_path = kwargs.get("eval_path", None)
        self.n_years = kwargs.get("n_years", 1)
        self.calibration_early_stopping = kwargs.get("calibration_early_stopping", True)
        self.test_early_stopping = kwargs.get("test_early_stopping", False)
        self.n_early = kwargs.get("n_early", 365)  # number

    def get_optimizer(
        self,
    ):
        return torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )

    def train(self, **kwargs) -> torch.Tensor:
        self.lr = kwargs.get("lr", 1e-4)
        self.num_epochs = kwargs.get("num_epochs", 10)
        self.step_size = kwargs.get("step_size", 1)
        self.gamma = kwargs.get("gamma", 0.5)
        self.batches_per_epoch = kwargs.get("batches_per_epoch", 20)
        self.pos_weight = kwargs.get("pos_weight", 1)
        self.loss_fn = kwargs.get("loss_fn", nn.MSELoss())

        # Report model size
        n_params, size_b = model_size_b(self.model)
        print(
            f"Training model with {n_params} parameters (size: {size_b / MiB:.3f} MiB)"
        )

        # Start
        self.model.to(self.device)
        print(self.device)
        self.optimizer = self.get_optimizer()
        self.scheduler = StepLR(
            self.optimizer, step_size=self.step_size, gamma=self.gamma
        )
        self.model.train()
        start_time = time.time()

        # create an empty dataframe with the losses
        losses_pd = pd.DataFrame(columns=[])
        # Train loop
        bs, T, C, H, W = next(iter(self.train_dataloader)).shape
        print("-" * 40)
        for idx, epoch in enumerate(range(self.num_epochs)):
            print(f"Epoch {epoch} :")
            print(f"Learning rate: {self.scheduler.get_last_lr()}")
            epoch_loss = []
            for i, batch in enumerate(self.train_dataloader):
                if i >= self.batches_per_epoch:
                    break
                # step 1 : sample a batch of data
                # batch = next(iter(self.train_dataloader))     # shape (bs, T, c+1, h, w), T = 24
                batch = batch.view(bs * T, C, H, W)  # shape (bs * T, c+1, h, w)
                x = batch[:, :-1, :, :].to(self.device)  # shape (bs * T, c, h, w)
                y = batch[:, -1, :, :].to(self.device)  # shape (bs * T, h, w)
                y = (y >= 2).float()  # shape (bs * T, h, w)

                # step 2 : make prediction
                pred = self.model(x)  # shape (bs, h, w)

                # step 3 : adapt model parameters
                self.optimizer.zero_grad()
                # tensor_weights = torch.ones_like(y) + (self.pos_weight - 1) * (y == 1).float()
                loss = self.loss_fn(pred, y)

                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss.item())

                # pbar.set_description(f'batch {i}, loss: {loss.item():.4f}')

            # step 4 : update learning rate
            self.scheduler.step()

            # save losses
            losses_pd[f"epoch_{idx}"] = epoch_loss
            # save model if intermediate_save=True in kwargs
            if kwargs.get("intermediate_save", False):
                losses_pd.to_csv(self.model.save_path / f"{self.model.name}_losses.csv")
                torch.save(
                    self.model.state_dict(),
                    self.model.save_path / f"{self.model.name}_epoch_{idx}.pth",
                )
                # remove the previous model if the file exists
                previous_model_path = (
                    self.model.save_path / f"{self.model.name}_epoch_{idx - 1}.pth"
                )
                if os.path.exists(previous_model_path):
                    os.remove(previous_model_path)

            # print loss
            print(f"loss: {sum(epoch_loss) / len(epoch_loss):.4f}")
            print(f"training time : {time.time() - start_time}")
            print("-" * 40)

        torch.save(
            self.model.state_dict(), self.model.save_path / f"{self.model.name}.pth"
        )
        losses_pd.to_csv(self.model.save_path / f"{self.model.name}_losses.csv")
        self.model.eval()
        print("Training finished")

    def recalibrate(self):
        if hasattr(self.model, "recalibration_method"):
            if self.model.recalibration_method == "platt_scaling":
                self.platt_recalibration()
            else:
                print(
                    f"Recalibration method {self.model.recalibration_method} not implemented. Defaulting to no recalibration."
                )
        else:
            print(
                "No valid recalibration method specified in the model. Defaulting to no recalibration."
            )

    def platt_recalibration(self):
        start_time = time.time()
        y_preds = []
        y_trues = []
        if self.calibration_early_stopping:
            n_batches = self.n_early
        else:
            n_batches = len(self.calibration_dataloader)

        self.model.eval()
        bs, T, C, H, W = next(iter(self.calibration_dataloader)).shape
        with torch.no_grad():
            for i, batch in enumerate(self.calibration_dataloader):
                if (
                    self.calibration_early_stopping and i >= n_batches
                ):  # use only 200 batches for recalibration
                    break
                if i % 100 == 0:
                    print(
                        f"{i}/{n_batches}, recalibration duration : {time.time() - start_time}"
                    )
                batch = batch.view(bs * T, C, H, W)  # shape (bs * T, c+2, h, w)
                x = batch[:, :-1, :, :].to(self.device)
                y = batch[:, -1, :, :].to(self.device)
                y = (y >= 2).float()
                pred = self.model(x, apply_recalibration=False)
                y_preds.append(pred.cpu())
                y_trues.append(y.cpu())

                # check the y_preds are not NaN
                if torch.isnan(pred).any():
                    print("NaN values found in y_preds")
                    # get the indices of the NaN values
                    print(f"Index of batch with NaN values: {i}")

        y_preds = torch.cat(y_preds).reshape(-1, 1)
        y_trues = torch.cat(y_trues).ravel()
        # fit a logistic regression
        self.recalibration = LogisticRegression(solver="lbfgs")
        self.recalibration.fit(y_preds.numpy(), y_trues.numpy())

        # add the recalibration to the model
        self.model.recalibration_layer.a = nn.Parameter(
            torch.tensor(
                self.recalibration.coef_, dtype=torch.float32, device=self.device
            )
        )
        self.model.recalibration_layer.b = nn.Parameter(
            torch.tensor(
                self.recalibration.intercept_, dtype=torch.float32, device=self.device
            )
        )
        # save the model
        torch.save(
            self.model.state_dict(), self.model.save_path / f"{self.model.name}.pth"
        )

        print(f"Recalibration finished ! Total time : {time.time() - start_time}")
        print(
            f"Platt recalibration coefficients: {self.recalibration.coef_}, intercept: {self.recalibration.intercept_}"
        )

    def test(self, extremes=False) -> float:
        """Can be used for test and validation"""
        start_time = time.time()
        self.model.to(self.device)
        self.model.eval()
        if extremes:
            loader = self.test_extremes_dataloader
        else:
            loader = self.test_dataloader
        if self.test_early_stopping:
            n_batches = min(self.n_early, len(loader))
        else:
            n_batches = len(loader)
        self.metrics = Metrics(
            kernel_size=1,
            save_path=self.eval_path,
            reduction="mean",
            n_batches=n_batches,
            extremes=extremes,
        )
        print(f"Testing model {self.model.name} on {n_batches} batches........")
        bs, T, C, H, W = next(iter(loader)).shape
        test_start_time = time.time()
        season = None
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if self.test_early_stopping and i >= self.n_early:
                    break
                if self.seasons is not None:
                    season = self.seasons[i]
                if (i + 1) % 100 == 0:
                    print(
                        f"{i}/{min(n_batches, len(loader))}, test duration : {time.time() - test_start_time}"
                    )
                batch = batch.view(bs * T, C, H, W)  # shape (bs * T, c+2, h, w)
                x = batch[:, :-1, :, :].to(self.device)
                y_true = batch[:, -1, :, :].to(self.device)
                y_true = (y_true >= 2).float()
                pred = self.model(x)
                # pred = self.model(x, season)    # for baseline model eval
                self.metrics.update_all(y_true, pred, season)

        self.metrics.compute_all()
