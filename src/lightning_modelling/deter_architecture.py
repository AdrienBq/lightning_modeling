import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ConditionalVectorField(nn.Module, ABC):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass


class ResidualLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        """
        res = x.clone()  # (bs, c, h, w)

        # Initial conv block
        x = self.block1(x)  # (bs, c, h, w)

        # Second conv block
        x = self.block2(x)  # (bs, c, h, w)

        # Add back residual
        x = x + res  # (bs, c, h, w)

        return x


class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int):
        super().__init__()
        self.downsample = nn.Conv2d(
            channels_in, channels_out, kernel_size=3, stride=2, padding=1
        )
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(channels_out) for _ in range(num_residual_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        """
        # Downsample: (bs, c_in, h, w) -> (bs, c_out, h // 2, w // 2)
        x = self.downsample(x)

        # Pass through residual blocks: (bs, c_in, h, w) -> (bs, c_in, h, w)
        for block in self.res_blocks:
            x = block(x)

        return x


class Midcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(channels) for _ in range(num_residual_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        """
        # Pass through residual blocks: (bs, c, h, w) -> (bs, c, h, w)
        for block in self.res_blocks:
            x = block(x)

        return x


class Decoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualLayer(channels_out) for _ in range(num_residual_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        """
        # Upsample: (bs, c_in, h, w) -> (bs, c_out, 2 * h, 2 * w)
        x = self.upsample(x)

        # Pass through residual blocks: (bs, c_out, h, w) -> (bs, c_out, 2 * h, 2 * w)
        for block in self.res_blocks:
            x = block(x)

        return x


class PlattScaling(nn.Module):
    """Platt scaling recalibration: a learned 1-parameter logistic map.

    Applies ``sigmoid(a * x + b)`` with learnable scale ``a`` and bias ``b`` to
    rescale a model's logits/scores into better-calibrated probabilities. The
    parameters are fit on a held-out calibration set (see
    ``Trainer.platt_recalibration``).
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([[1.0]]))  # scale
        self.b = nn.Parameter(torch.tensor([0.0]))  # bias

    def forward(self, x):
        # x shape: [B, H, W] or [B, H, W]
        return torch.sigmoid(self.a * x + self.b)


class HistogramBinning(nn.Module):
    """DEPRECATED - Histogram-binning recalibration: 
    Map each score to a per-bin probability.

    Predictions are passed through a sigmoid, bucketed into ``num_bins`` bins
    defined by ``bin_edges`` (finer near 0 and 1 where lightning probabilities
    concentrate), and each bin is assigned a calibrated probability learned from
    the observed positive fraction in that bin.

    INVARIANT: ``num_bins`` must equal ``len(bin_edges)`` (44). ``bin_edges`` is
    hard-coded to 44 entries here, so overriding ``num_bins`` via kwargs without
    also changing ``bin_edges`` will break the bin lookup in ``forward``.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_bins = kwargs.get("num_bins", 44)
        self.bin_edges = torch.tensor(
            [
                0,
                1e-12,
                1e-11,
                1e-10,
                1e-9,
                1e-8,
                1e-7,
                1e-6,
                1e-5,
                1e-4,
                0.0005,
                1e-3,
                0.005,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                0.975,
                0.99,
                0.995,
                0.999,
                1 - 1e-4,
                1 - 1e-5,
                1 - 1e-6,
            ]
        )
        self.points_per_bin = nn.Parameter(
            torch.zeros(self.num_bins)
        )  # initialized to zero
        self.positive_points_per_bin = nn.Parameter(
            torch.zeros(self.num_bins)
        )  # initialized to zero
        self.bin_probs = nn.Parameter(
            torch.ones(self.num_bins) / self.num_bins
        )  # Learnable bin probabilities

    def forward(self, x):
        x = torch.sigmoid(x)
        # for each pixel, find the corresponding bin and replace it with the bin probability
        bin_indices = (
            torch.bucketize(x, self.bin_edges, right=True) - 1
        )  # get bin index for each pixel
        return self.bin_probs[bin_indices]


class Unet(ConditionalVectorField):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = kwargs.get("name", "FirstUnet")
        self.save_path = kwargs.get("save_path", None)
        self.num_residual_layers = kwargs.get("num_residual_layers", 1)
        self.channels = kwargs.get("channels", [32, 64, 128])
        self.recalibration_method = kwargs.get("recalibration_method", None)
        self.removed_features = kwargs.get("removed_features", [])

        # Initial padding to have an image of size 112 x 160
        self.pad = nn.ReflectionPad2d((5, 6, 5, 6))

        # Initial convolution: (bs, 5, H, W) -> (bs, c_0, H, W)
        self.init_conv = nn.Sequential(
            nn.Conv2d(
                max(5 - len(self.removed_features), 1),
                self.channels[0],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.channels[0]),
            nn.SiLU(),
            ResidualLayer(self.channels[0]),
        )

        # Encoders, Midcoders, and Decoders
        encoders = []

        decoders = []
        for curr_c, next_c in zip(self.channels[:-1], self.channels[1:]):
            encoders.append(Encoder(curr_c, next_c, self.num_residual_layers))
            decoders.append(Decoder(next_c, curr_c, self.num_residual_layers))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(self.channels[-1], self.num_residual_layers)

        # Final convolution
        self.final_conv = nn.Sequential(
            ResidualLayer(self.channels[0]),
            nn.Conv2d(self.channels[0], 1, kernel_size=3, padding=1),
        )

        # recalibration
        if self.recalibration_method == "platt_scaling":
            self.recalibration_layer = PlattScaling()
        elif self.recalibration_method == "histogram_binning":
            self.recalibration_layer = HistogramBinning(**kwargs)
        else:
            self.recalibration_layer = (
                nn.Sigmoid()
            )  # default to sigmoid if no recalibration

    def forward(
        self, x: torch.Tensor, apply_recalibration: bool = True
    ) -> torch.Tensor:
        """
        Args:
        - x: (bs, 5 - len(self.removed_features), 101, 149) --> possible to downsample 4 times
        Returns:
        - light(x): (bs, 1, 101, 149)
        """
        if len(self.removed_features) > 0:
            if len(self.removed_features) == 5:
                x = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
            elif len(self.removed_features) < 5:
                remove = torch.tensor(self.removed_features, device=x.device)
                keep = torch.ones(x.size(1), dtype=torch.bool, device=x.device)
                keep[remove] = False
                x = x[:, keep, :, :]
            else:
                raise ValueError(
                    "removed_features should be a list of integers between 0 and 4"
                )
        x = x.to(torch.float32)

        # Initial padding
        x = self.pad(
            x
        )  # (bs, 5 - len(self.removed_features), 101, 149) -> (bs, 5 - len(self.removed_features), 112, 160)

        # Initial convolution
        x = self.init_conv(x)  # (bs, c_0, 112, 160)

        residuals = []

        # Encoders
        for encoder in self.encoders:
            residuals.append(x.clone())
            x = encoder(x)  # (bs, c_i, h, w) -> (bs, c_{i+1}, h // 2, w //2)

        # Midcoder
        x = self.midcoder(x)

        # Decoders
        for decoder in self.decoders:
            x = decoder(x)  # (bs, c_i, h, w) -> (bs, c_{i-1}, 2 * h, 2 * w)
            res = residuals.pop()  # (bs, c_i, h, w)
            x = x + res

        # Final convolution
        x = self.final_conv(x).squeeze(1)  # (bs, 112, 160)
        x = x[:, 5 : 5 + 101, 5 : 5 + 149]  # crop to the original size

        if self.training or not apply_recalibration:
            return x

        x = self.recalibration_layer(x)

        return x


class FullyConnectedNet_1d(nn.Module):
    def __init__(self, **kwargs):
        super(FullyConnectedNet_1d, self).__init__()
        self.name = kwargs.get("name", "MLP")
        self.save_path = kwargs.get("save_path", None)
        self.recalibration_method = kwargs.get("recalibration_method", None)
        self.removed_features = kwargs.get("removed_features", [])

        self.c = kwargs.get("c", max(1, 5 - len(self.removed_features)))
        self.h = kwargs.get("h", 101)
        self.w = kwargs.get("w", 149)
        self.hidden_dims = kwargs.get("hidden_dims", [256])
        input_dim = self.c
        output_dim = 1

        hidden_layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            hidden_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        hidden_layers.append(nn.Linear(prev_dim, output_dim))
        self.hidden_layers = nn.ModuleList(hidden_layers)

        self.net = nn.Sequential(*self.hidden_layers)

        self.recalibration_layer = PlattScaling()

    def forward(self, x, apply_recalibration: bool = True):  # x: (batch_size, c, h, w)
        if len(self.removed_features) > 0:
            if len(self.removed_features) == 5:
                x = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
            elif len(self.removed_features) < 5:
                remove = torch.tensor(self.removed_features, device=x.device)
                keep = torch.ones(x.size(1), dtype=torch.bool, device=x.device)
                keep[remove] = False
                x = x[:, keep, :, :]
            else:
                raise ValueError(
                    "removed_features should be a list of integers between 0 and 4"
                )
        x = x.to(torch.float32)

        x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        out = self.net(x_flat)  # (batch_size * h * w)
        out = out.view(x.shape[0], x.shape[2], x.shape[3])  # (batch_size, h, w)

        if self.training or not apply_recalibration:
            return out
        out = self.recalibration_layer(out)
        return out
