import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle


# Number of days in the full original dataset (2008-01-02 .. 2023-12-31). Only a
# subset of these samples is shipped with the demo (the 49 extreme days); the
# full-length range is still used to build the id-indexed metadata below.
N_TOTAL_DAYS = 5843


class CustomPTDataset(Dataset):
    """Dataset of daily gridded samples stored as ``.pt`` tensors.

    Each item is a tensor of shape ``(24, 6, 101, 149)`` = (hours, channels,
    latitudes, longitudes); channels 0-4 are predictors and channel 5 is the
    lightning observation. Items are addressed by position in the sorted list of
    available ``.pt`` files, i.e. ``sample_files[sample_ids[idx]]`` — note the
    demo ships only the 49 extreme-day files, so ``sample_ids`` are positional
    indices into that list. If a scaler exists at ``scaler_path`` it is applied
    to the predictor channels in ``__getitem__``.
    """

    def __init__(self, root_dir, sample_ids, **kwargs):
        self.samples_dir = os.path.join(root_dir, "samples")
        self.sample_files = sorted(
            f for f in os.listdir(self.samples_dir) if f.endswith(".pt")
        )
        self.sample_ids = sample_ids
        self.scaler_path = kwargs.get(
            "scaler_path", os.path.join(root_dir, "scaler", "scaler_full.pkl")
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
            self.metadata_csv = pd.read_csv(csv_path).iloc[sample_ids]
        self.transform = None
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

        # Apply transforms if any
        if self.transform:
            sample[:, :-1, :, :] = self.normalize(sample[:, :-1, :, :])

        return sample 


def create_train_test(dataset_path, train_years, test_years, **kwargs):
    """Build (full, train, test) datasets split by year.

    Loads the full metadata, assigns each day to the train or test split by its
    year, and optionally restricts to a single meteorological ``season``
    (``"summer"``, ``"winter"``, ``"spring"``, ``"autumn"``, or
    ``None``/``"all"`` for no filter). Returns three ``CustomPTDataset`` objects:
    the full dataset, the train split, and the test split. Recognised
    ``kwargs``: ``scaler_path`` (path to the fitted scaler) and ``season``.
    """
    scaler_path = kwargs.get(
        "scaler_path", os.path.join(dataset_path, "scaler", "scaler_full.pkl")
    )
    season = kwargs.get("season", None)
    full_dataset = CustomPTDataset(
        root_dir=dataset_path, sample_ids=range(N_TOTAL_DAYS), scaler_path=scaler_path
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


def get_seasons(years):
    """Return the meteorological season label for every day in ``years``.

    Produces a flat list (one entry per day, chronologically) mapping each day
    to ``"winter"``, ``"spring"``, ``"summer"``, or ``"autumn"`` by month. The
    per-day labels line up with the metadata ``id`` order, so they can be
    indexed by sample id to stratify metrics by season.
    """
    validation_dates = pd.concat(
        [
            pd.Series(
                pd.date_range(
                    # The dataset starts on 2008-01-02, so 2008 begins a day
                    # late; every other year starts on Jan 1st.
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
