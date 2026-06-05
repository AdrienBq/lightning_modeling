import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import matplotlib.cm as cm
import cartopy.crs as ccrs
from scipy.fft import fft2, fftshift
import os
import torch
import cmocean


class LightningPlotMultiModel:
    def __init__(self, lightning_hours, metadata_json, title, **kwargs):
        self.lightning_hours = lightning_hours
        self.json_metadata = metadata_json
        self.plot_title = title
        self.save_path = kwargs.get("save_path", None)
        self.file_name = kwargs.get("file_name", "lightning_comparison.png")
        self.levels = kwargs.get("levels", None)
        self.full_errors = kwargs.get("full_errors", None)
        self.conditionned_errors = kwargs.get("conditionned_errors", None)

        if self.levels is None:
            self.levels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        self.model_colors = [
            "#000000",  # black
            "#1e9684",  # cyan
            "#4576bf",  # blue
            "#f21111",  # red
            "#e88831",  # orange
            "#8036a8",  # purple
        ]

        self.base_colors = [
            "#FFFFFF",
            "#FFF5A6",
            "#FFE37B",
            "#FACA57",
            "#F5AD37",
            "#F08C1E",
            "#E36C16",
            "#D24D17",
            "#B33117",
            "#992015",
        ]
        self.colors = self._create_colors()
        self.cmap = ListedColormap(self.colors, "lightnings")
        self.norm = BoundaryNorm(self.levels, self.cmap.N)

        self._extract_metadata()
        self._create_figure()
        self._plot()

    def _create_colors(self):
        return np.concatenate(
            [
                np.concatenate(
                    [
                        np.linspace(cm.colors.to_rgba(c), cm.colors.to_rgba(d), 6)[:-1]
                        for (c, d) in zip(self.base_colors[:2], self.base_colors[1:3])
                    ]
                ),
                np.concatenate(
                    [
                        np.linspace(cm.colors.to_rgba(c), cm.colors.to_rgba(d), 9)[:-1]
                        for (c, d) in zip(self.base_colors[2:-1], self.base_colors[3:])
                    ]
                ),
            ]
        )

    def _extract_metadata(self):
        self.min_lat = int(float(self.json_metadata["min_latitude"]))
        self.max_lat = int(float(self.json_metadata["max_latitude"]))
        self.min_lon = int(float(self.json_metadata["min_longitude"]))
        self.max_lon = int(float(self.json_metadata["max_longitude"]))
        self.res = float(self.json_metadata["spatial_resolution"].split()[0])
        self.extent = [self.min_lon, self.max_lon, self.min_lat, self.max_lat]

        self.lon = np.arange(self.min_lon, self.max_lon + self.res, self.res)
        self.lat = np.arange(self.min_lat, self.max_lat + self.res, self.res)

    def _create_figure(self):
        self.fig = plt.figure(figsize=(20, 7))
        self.gs = self.fig.add_gridspec(
            2, 4, width_ratios=[1, 1.3, 1.3, 1.3], height_ratios=[1, 1], wspace=0.001
        )

        self.ax1 = self.fig.add_subplot(self.gs[:, 0])
        self.ax2 = self.fig.add_subplot(self.gs[0, 1], projection=ccrs.EuroPP())
        self.ax3 = self.fig.add_subplot(self.gs[0, 2], projection=ccrs.EuroPP())
        self.ax4 = self.fig.add_subplot(self.gs[0, 3], projection=ccrs.EuroPP())
        self.ax5 = self.fig.add_subplot(self.gs[1, 1], projection=ccrs.EuroPP())
        self.ax6 = self.fig.add_subplot(self.gs[1, 2], projection=ccrs.EuroPP())
        self.ax7 = self.fig.add_subplot(self.gs[1, 3], projection=ccrs.EuroPP())
        self.axes = [
            self.ax1,
            self.ax2,
            self.ax3,
            self.ax4,
            self.ax5,
            self.ax6,
            self.ax7,
        ]
        for i, ax in enumerate(self.axes[1:]):
            ax.set_extent([-5, 20, 30, 55], crs=ccrs.PlateCarree())
            ax.coastlines()
            # draw gridlines with labels only on the left and bottom axes
            gls = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.7)
            gls.top_labels = False
            gls.right_labels = False
            if i in range(3):
                gls.bottom_labels = False
            if i in [1, 2, 4, 5]:
                gls.left_labels = False
            # ax.gridlines(linestyle="--", alpha=0.7)
            ax.set_aspect("equal")
            # ax.add_feature(cfeature.OCEAN,facecolor=(0.5,0.5,0.5))
            # ax.add_feature(cfeature.LAND)
            # ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=1)

    def _plot(self):
        power_spectrums = []
        models = ["Observations", "LR", "GAM", "XGB", "MLP", "U-Net"]
        panels = [
            "(a) 1D Power Spectrum",
            "(b) Observations",
            "(c) LR",
            "(d) GAM",
            "(e) XGB",
            "(f) MLP",
            "(g) U-Net",
        ]

        for i, ax in enumerate(self.axes[1:]):
            lightning = self.lightning_hours[i]
            ps = compute_1d_power_spectrum(lightning)
            power_spectrums.append(ps)
            pred_lightning = ax.imshow(
                lightning,
                extent=self.extent,
                origin="upper",
                cmap=self.cmap,
                norm=self.norm,
                alpha=1,
                transform=ccrs.PlateCarree(),
            )
            if i >= 1 and self.full_errors is not None:
                if self.conditionned_errors is not None:
                    error_text = f"RMSE:\n  All: {self.full_errors[i - 1]:.2f}\n  Obs>0: {self.conditionned_errors[i - 1]:.2f}"
                    ax.text(
                        0.05,
                        0.95,
                        error_text,
                        transform=ax.transAxes,
                        fontsize=14,
                        color="black",
                        ha="left",
                        va="top",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.9
                        ),
                    )
                else:
                    error_text = f"RMSE: {self.full_errors[i - 1]:.2f}"
                    ax.text(
                        0.05,
                        0.95,
                        error_text,
                        transform=ax.transAxes,
                        fontsize=14,
                        color="black",
                        ha="left",
                        va="top",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.9
                        ),
                    )

        # Use ax1's height as a reference
        hori_pad = 0.05
        ref_box = self.ax2.get_position()
        ref_box.y0 -= 0.05
        self.ax1.set_position(
            [
                self.ax1.get_position().x0,
                self.ax1.get_position().y0,
                self.ax1.get_position().width,
                self.ax1.get_position().height - 0.04,
            ]
        )

        # Set all axes to match the same vertical height
        for i, ax in enumerate(self.axes[1:4]):
            ax.set_position(
                [
                    ax.get_position().x0 - i * hori_pad,
                    ref_box.y0,
                    ax.get_position().width,
                    ref_box.height,
                ]
            )
        for i, ax in enumerate(self.axes[4:]):
            ax.set_position(
                [
                    ax.get_position().x0 - i * hori_pad,
                    ref_box.y0 - ref_box.height - 0.03,
                    ax.get_position().width,
                    ref_box.height,
                ]
            )

        # Get ax2's position in figure coordinates
        bot_right_box = self.ax7.get_position()

        # Define colorbar size and position
        cbar_width = 0.015
        cbar_pad = 0.02  # padding between ax2 and colorbar

        cbar_x = bot_right_box.x1 + cbar_pad
        cbar_y = bot_right_box.y0
        cbar_height = ref_box.y0 - bot_right_box.y0 + ref_box.height

        # Create colorbar axis and add the colorbar
        cax = self.fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
        cbar = self.fig.colorbar(pred_lightning, cax=cax, orientation="vertical")
        cbar.set_label(
            "Number of hours of lightning", fontsize=20, rotation=270, labelpad=25
        )
        cbar.ax.tick_params(labelsize=16)

        # L = np.sqrt((100/4)**2 + (148/4)**2) * 120
        # lamb = L / np.arange(1, len(obs_ps) + 1)
        for i, ps in enumerate(power_spectrums):
            self.ax1.plot(ps, label=models[i], color=self.model_colors[i])
        self.ax1.legend(fontsize=14, loc="lower left")
        self.ax1.set_xlabel("Wavenumber", fontsize=16)
        # ax3.set_xlabel('Wavelength')
        self.ax1.set_ylabel("Power", fontsize=16)
        self.ax1.set_yscale("log")
        self.ax1.set_xscale("log")
        self.ax1.set_xlim(0.8, len(power_spectrums[0]))
        self.ax1.yaxis.grid(True, linestyle="--", alpha=0.7)
        self.ax1.xaxis.grid(True, linestyle="--", alpha=0.7)

        self.fig.text(0.47, 0.95, self.plot_title, fontsize=24, ha="center", va="top")
        # self.ax1.set_title('1D Power Spectrum', fontsize=20)
        for i, ax in enumerate(self.axes):
            ax.set_title(panels[i], fontsize=20)

    def show(self):
        plt.show()

    def save(self, dpi=300):
        if self.save_path is None:
            raise ValueError("Save path is not specified.")
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
        self.fig.savefig(self.save_path / self.file_name, bbox_inches="tight", dpi=dpi)
        plt.close(self.fig)


def compute_1d_power_spectrum(image):
    # Compute 2D FFT and shift zero frequency to center
    f_transform = fftshift(fft2(image))

    # Compute power spectrum (magnitude squared)
    power_spectrum = np.abs(f_transform) ** 2

    # Create a radial distance map
    y, x = np.indices(image.shape)
    center = np.array(image.shape) // 2
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int32)

    # Compute 1D power spectrum by averaging over rings
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)  # Avoid division by zero

    return radial_profile


class SaliencyPlotPaper:
    def __init__(
        self,
        input,
        model,
        metadata_json,
        lat,
        lon,
        title,
        device,
        save_path=None,
        file_name=None,
        levels=None,
    ):
        self.json_metadata = metadata_json
        self.plot_title = title
        self.save_path = save_path
        self.file_name = file_name
        self.model = model
        self.input = input
        self.lat_saliency = lat
        self.lon_saliency = lon
        self.device = device

        # self.input_names = ['lifted index', 'mixing ratio', 'relative humidity', 'convective precipitation', 'land-sea mask']
        self.input_names = ["MU_LI", "MU_MIXR", "500850_RH", "CP", "LSM"]

        if levels is None:
            self.levels = np.array(
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            )
        else:
            self.levels = levels
        self.base_colors = [
            "#FFFFFF",
            "#FFF5A6",
            "#FFE37B",
            "#FACA57",
            "#F5AD37",
            "#F08C1E",
            "#E36C16",
            "#D24D17",
            "#B33117",
            "#992015",
        ]
        self.colors = self._create_colors()
        self.cmap = ListedColormap(self.colors, "lightnings")
        self.norm = BoundaryNorm(self.levels, self.cmap.N)

        self._extract_metadata()
        self._create_figure()
        self._compute_saliency()
        self._plot()

    def _create_colors(self):
        return np.concatenate(
            [
                np.concatenate(
                    [
                        np.linspace(cm.colors.to_rgba(c), cm.colors.to_rgba(d), 6)[:-1]
                        for (c, d) in zip(self.base_colors[:2], self.base_colors[1:3])
                    ]
                ),
                np.concatenate(
                    [
                        np.linspace(cm.colors.to_rgba(c), cm.colors.to_rgba(d), 9)[:-1]
                        for (c, d) in zip(self.base_colors[2:-1], self.base_colors[3:])
                    ]
                ),
            ]
        )

    def _extract_metadata(self):
        self.min_lat = int(float(self.json_metadata["min_latitude"]))
        self.max_lat = int(float(self.json_metadata["max_latitude"]))
        self.min_lon = int(float(self.json_metadata["min_longitude"]))
        self.max_lon = int(float(self.json_metadata["max_longitude"]))
        self.res = float(self.json_metadata["spatial_resolution"].split()[0])
        self.extent = [self.min_lon, self.max_lon, self.min_lat, self.max_lat]

        self.lon = np.arange(self.min_lon, self.max_lon + self.res, self.res)
        self.lat = np.arange(self.min_lat, self.max_lat + self.res, self.res)

    def _compute_saliency(self):
        i = int((self.max_lat - self.lat_saliency) / self.res)
        j = int((self.lon_saliency - self.min_lon) / self.res)

        # we don't need gradients w.r.t. weights for a trained model
        for param in self.model.parameters():
            param.requires_grad = False

        # set model in eval mode
        self.model.eval()
        self.input.to(self.device)

        # we want to calculate gradient of higest score w.r.t. input
        # so set requires_grad to True for input
        self.input.requires_grad = True
        # forward pass to calculate predictions
        self.preds = self.model(self.input)
        score = self.preds[:, i, j].mean()
        print(
            "score at (lat, lon) = ({:.2f}, {:.2f}): {:.6f}".format(
                self.lat_saliency, self.lon_saliency, score.item()
            )
        )
        # backward pass to get gradients of score predicted class w.r.t. input image
        score.backward()
        # get max along channel axis
        slc = torch.abs(self.input.grad[0])
        # normalize to [0..1]
        # self.slc = (slc - slc.min())/(slc.max()-slc.min())
        self.slc = slc

    def _create_figure(self):
        self.fig = plt.figure(figsize=(10, 30))
        self.gs = self.fig.add_gridspec(6, 2, wspace=0.3)

        self.axes = []
        for i in range(6):
            for j in range(2):
                self.axes.append(
                    self.fig.add_subplot(self.gs[i, j], projection=ccrs.EuroPP())
                )
        for ax in self.axes:
            ax.set_extent([-5, 20, 30, 55], crs=ccrs.PlateCarree())
            ax.coastlines(color="grey")
            gls = ax.gridlines(
                draw_labels=True, linestyle="--", alpha=0.7, color="gray"
            )
            ax.set_aspect("equal")
            gls.top_labels = False
            gls.right_labels = False
            # ax.add_feature(cfeature.OCEAN,facecolor=(0.5,0.5,0.5))
            # ax.add_feature(cfeature.LAND)
            # ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=1)

    def _plot(self):
        lightning_pred = self.preds.squeeze(1).detach().cpu().numpy()[0]
        full_saliency = torch.linalg.vector_norm(self.slc, dim=0).detach().cpu().numpy()

        max_saliency = full_saliency.max()
        saliency_norm = plt.Normalize(vmin=0, vmax=max_saliency)
        input_cmaps = [cm.RdBu_r, cm.RdBu_r, cm.RdBu_r, cmocean.cm.rain, cm.BrBG_r]

        plot_lightning = self.axes[0].imshow(
            lightning_pred,
            extent=self.extent,
            origin="upper",
            cmap=self.cmap,
            norm=self.norm,
            alpha=1,
            transform=ccrs.PlateCarree(),
        )

        # add self.saliency_lat, self.saliency_lon marker to plot_lightning
        self.axes[0].plot(
            self.lon_saliency,
            self.lat_saliency,
            marker="x",
            color="black",
            markersize=10,
            transform=ccrs.PlateCarree(),
        )

        plot_full_saliency = self.axes[1].imshow(
            full_saliency,
            extent=self.extent,
            origin="upper",
            cmap=cm.hot,
            norm=saliency_norm,
            alpha=1,
            transform=ccrs.PlateCarree(),
        )
        input_plots = []
        saliency_plots = []
        for i in range(1, 6):
            input = self.input[0, i - 1, :, :].detach().cpu().numpy()
            if i in range(1, 4):
                absolute_max = np.max(np.abs(input))
                norm = plt.Normalize(vmin=-absolute_max, vmax=absolute_max)
            else:
                norm = plt.Normalize(vmin=0, vmax=input.max())
            slc = self.slc[i - 1].detach().cpu().numpy()
            input_plots.append(
                self.axes[2 * i].imshow(
                    input,
                    extent=self.extent,
                    origin="upper",
                    cmap=input_cmaps[i - 1],
                    norm=norm,
                    alpha=1,
                    transform=ccrs.PlateCarree(),
                )
            )
            self.axes[2 * i].plot(
                self.lon_saliency,
                self.lat_saliency,
                marker="x",
                color="black",
                markersize=10,
                transform=ccrs.PlateCarree(),
            )

            saliency_plots.append(
                self.axes[2 * i + 1].imshow(
                    slc,
                    extent=self.extent,
                    origin="upper",
                    cmap=cm.hot,
                    norm=saliency_norm,
                    alpha=1,
                    transform=ccrs.PlateCarree(),
                )
            )

        # Use the first and last axes position to set the colorbars ref
        top_left_box = self.axes[0].get_position()
        top_right_box = self.axes[1].get_position()
        # bot_left_box = self.axes[-2].get_position()
        # bot_right_box = self.axes[-1].get_position()

        # Define colorbar size and position
        cbar_height = top_left_box.height
        cbar_pad = 0.01
        cbar_width = 0.02

        cbar_proba_x = top_left_box.x0 - 8 * cbar_pad
        cbar_proba_y = top_left_box.y0
        # Create colorbars axis and add the colorbar
        c_proba_ax = self.fig.add_axes(
            [cbar_proba_x, cbar_proba_y, cbar_width, cbar_height]
        )
        cbar_proba = self.fig.colorbar(
            plot_lightning, cax=c_proba_ax, orientation="vertical", ticklocation="left"
        )
        cbar_proba.set_label("Probability of lightning", fontsize=15)
        cbar_proba.ax.tick_params(labelsize=14)
        self.axes[0].set_title("(a) Lightning prediction", fontsize=18)

        cbar_saliency_left_x = top_right_box.x0 + top_right_box.width + 2 * cbar_pad
        cbar_saliency_left_y = top_right_box.y0

        c_saliency_left_ax = self.fig.add_axes(
            [cbar_saliency_left_x, cbar_saliency_left_y, cbar_width, cbar_height]
        )
        cbar_saliency_left = self.fig.colorbar(
            plot_full_saliency,
            cax=c_saliency_left_ax,
            orientation="vertical",
            ticklocation="right",
        )
        cbar_saliency_left.set_label("Total gradient", fontsize=15)
        cbar_saliency_left.ax.tick_params(labelsize=14)
        self.axes[1].set_title("(b) Total saliency map", fontsize=18)

        for i in range(1, 6):
            box1 = self.axes[2 * i].get_position()
            cbar_input_x = box1.x0 - 8 * cbar_pad
            cbar_input_y = box1.y0

            c_input_ax = self.fig.add_axes(
                [cbar_input_x, cbar_input_y, cbar_width, cbar_height]
            )
            cbar_input = self.fig.colorbar(
                input_plots[i - 1],
                cax=c_input_ax,
                orientation="vertical",
                ticklocation="left",
            )
            cbar_input.set_label(f"{self.input_names[i - 1]}", fontsize=15)
            cbar_input.ax.tick_params(labelsize=14)
            self.axes[2 * i].set_title(
                f"({chr(97 + 2 * i)}) {self.input_names[i - 1]} input", fontsize=18
            )

            box2 = self.axes[2 * i + 1].get_position()
            cbar_input_x = box2.x0 + top_right_box.width + 2 * cbar_pad
            cbar_input_y = box2.y0

            c_saliency_ax = self.fig.add_axes(
                [cbar_input_x, cbar_input_y, cbar_width, cbar_height]
            )
            cbar_saliency = self.fig.colorbar(
                plot_full_saliency,
                cax=c_saliency_ax,
                orientation="vertical",
                ticklocation="right",
            )
            cbar_saliency.set_label(f"{self.input_names[i - 1]} gradient", fontsize=15)
            cbar_saliency.ax.tick_params(labelsize=14)
            self.axes[2 * i + 1].set_title(
                f"({chr(98 + 2 * i)}) {self.input_names[i - 1]} saliency", fontsize=18
            )

        self.fig.text(0.5, 0.92, self.plot_title, fontsize=20, ha="center", va="top")
        # add legend for the marker
        # self.fig.text(0.1, 0.9, 'X: Sal
        if self.save_path is not None and self.file_name is not None:
            plt.savefig(self.save_path / self.file_name, dpi=300, bbox_inches="tight")
        plt.show()
