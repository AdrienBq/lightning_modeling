import os
import pandas as pd


import torch
from torch.utils.data import Dataset
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomPTDataset(Dataset):
    def __init__(self, root_dir, sample_ids, **kwargs):
        self.samples_dir = os.path.join(root_dir, "samples")
        self.sample_files = [
            f for f in os.listdir(self.samples_dir) if f.endswith(".pt")
        ]
        self.sample_ids = sample_ids
        self.scaler_path = kwargs.get(
            "scaler_path", os.path.join(root_dir, "scalers", "final", "scaler_full.pkl")
        )

        # Optional: load metadata
        self.metadata_csv = None
        self.metadata_json = None
        csv_path = os.path.join(root_dir, "metadata.csv")
        json_path = os.path.join(root_dir, "metadata.json")
        if os.path.exists(json_path):
            import json

            with open(json_path, "r") as f:
                self.metadata_json = json.load(f)
        if os.path.exists(csv_path):
            import pandas as pd

            self.metadata_csv = pd.read_csv(csv_path).iloc[sample_ids]
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, "rb") as f:
                scaler = pickle.load(f)
            self.transform = scaler

    def __len__(self):
        return len(self.sample_ids)

    def normalize(self, batch):
        """
        Normalize a batch of shape (bs, C, H, W)
        """
        mean = torch.tensor(
            self.transform.mean_, dtype=batch.dtype, device=batch.device
        ).view(1, -1, 1, 1)
        scale = torch.tensor(
            self.transform.scale_, dtype=batch.dtype, device=batch.device
        ).view(1, -1, 1, 1)
        return (batch - mean) / scale

    def __getitem__(self, idx):
        sample_path = os.path.join(
            self.samples_dir, self.sample_files[self.sample_ids[idx]]
        )
        sample = torch.load(
            sample_path
        )  # tensor of shape (batch_size, channels, height, width) = (bs, 6, 101, 149)

        # Optional: attach metadata
        meta_csv = (
            self.metadata_csv.iloc[idx] if self.metadata_csv is not None else None
        )

        # Apply transforms if any
        if self.transform:
            sample[:, :-1, :, :] = self.normalize(sample[:, :-1, :, :])

        # mask the last channel : if value >= 2 put one, zero otherwise
        # sample[-1] = (sample[-1] >= 2).float()

        return sample  # or return (sample, meta) if needed


def create_train_test(dataset_path, train_years, test_years, **kwargs):
    scaler_path = kwargs.get(
        "scaler_path", os.path.join(dataset_path, "scalers", "final", "scaler_full.pkl")
    )
    season = kwargs.get("season", None)
    full_dataset = CustomPTDataset(
        root_dir=dataset_path, sample_ids=range(5843), scaler_path=scaler_path
    )

    metadata = full_dataset.metadata_csv.copy()
    metadata["year"] = metadata["date"].astype(str).str[:4].astype(int)

    if season is None or season == "all":
        train_ids = metadata.loc[metadata["year"].isin(train_years), "id"].values
        test_ids = metadata.loc[metadata["year"].isin(test_years), "id"].values
    elif season == "summer":
        train_ids = metadata.loc[
            (metadata["year"].isin(train_years))
            & (metadata["date"].str[5:7].astype(int).isin([6, 7, 8])),
            "id",
        ].values
        test_ids = metadata.loc[
            (metadata["year"].isin(test_years))
            & (metadata["date"].str[5:7].astype(int).isin([6, 7, 8])),
            "id",
        ].values
    elif season == "winter":
        train_ids = metadata.loc[
            (metadata["year"].isin(train_years))
            & (metadata["date"].str[5:7].astype(int).isin([12, 1, 2])),
            "id",
        ].values
        test_ids = metadata.loc[
            (metadata["year"].isin(test_years))
            & (metadata["date"].str[5:7].astype(int).isin([12, 1, 2])),
            "id",
        ].values
    elif season == "spring":
        train_ids = metadata.loc[
            (metadata["year"].isin(train_years))
            & (metadata["date"].str[5:7].astype(int).isin([3, 4, 5])),
            "id",
        ].values
        test_ids = metadata.loc[
            (metadata["year"].isin(test_years))
            & (metadata["date"].str[5:7].astype(int).isin([3, 4, 5])),
            "id",
        ].values
    elif season == "autumn":
        train_ids = metadata.loc[
            (metadata["year"].isin(train_years))
            & (metadata["date"].str[5:7].astype(int).isin([9, 10, 11])),
            "id",
        ].values
        test_ids = metadata.loc[
            (metadata["year"].isin(test_years))
            & (metadata["date"].str[5:7].astype(int).isin([9, 10, 11])),
            "id",
        ].values
    else:
        raise ValueError(f"Invalid season: {season}")

    assert len(set(train_ids) & set(test_ids)) == 0

    train_dataset = CustomPTDataset(
        root_dir=dataset_path, sample_ids=train_ids, scaler_path=scaler_path
    )

    test_dataset = CustomPTDataset(
        root_dir=dataset_path, sample_ids=test_ids, scaler_path=scaler_path
    )

    return full_dataset, train_dataset, test_dataset


def create_train_val_test(dataset_path, train_years, val_years, test_years, **kwargs):
    scaler_path = kwargs.get(
        "scaler_path", os.path.join(dataset_path, "scalers", "scaler_final.pkl")
    )
    full_dataset = CustomPTDataset(
        root_dir=dataset_path, sample_ids=range(5843), scaler_path=scaler_path
    )

    metadata = full_dataset.metadata_csv.copy()
    metadata["year"] = metadata["date"].astype(str).str[:4].astype(int)

    train_ids = metadata.loc[metadata["year"].isin(train_years), "id"].values
    val_ids = metadata.loc[metadata["year"].isin(val_years), "id"].values
    test_ids = metadata.loc[metadata["year"].isin(test_years), "id"].values

    # assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0

    train_dataset = CustomPTDataset(
        root_dir=dataset_path, sample_ids=train_ids, scaler_path=scaler_path
    )

    validation_dataset = CustomPTDataset(
        root_dir=dataset_path, sample_ids=val_ids, scaler_path=scaler_path
    )

    test_dataset = CustomPTDataset(
        root_dir=dataset_path, sample_ids=test_ids, scaler_path=scaler_path
    )

    return full_dataset, train_dataset, validation_dataset, test_dataset


def get_seasons(years):
    validation_dates = pd.concat(
        [
            pd.Series(
                pd.date_range(
                    f"{y}-01-02" if y == 2008 else f"{y}-01-01",
                    f"{y}-12-31",
                    freq="D",
                )
            )
            for y in years
        ]
    )
    seasons = validation_dates.dt.month.map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    ).to_list()
    return seasons
