# Dataset

This folder contains the dataset of extreme events used in the notebooks. It contains the following files and folders

---
## Single files

### extreme_days_top_0.05.csv

This file contains the ids of the most extreme days. They correspond to the full original dataset which is not showed here. This file is useful to load the right samples in the notebooks.

---
### metadata.csv and metadata.json

These files contain metadata on the full original dataset. They are also used to load the right samples in the notebooks.


---
## Climatology

This folder contains netcdfs of the seasonal and yearly climatology of lightning observations. The observations are data from the Met Office **ATDnet**.

The climatologies were computed on the training set which consists of the years 2009, 2010, 2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021 and 2022.

Each netcdf contains one map of average number of hours of lightning over the seasons (winter, spring, summer, autumn or year).

> ATDnet data provided by the Met Office (2026) under a [CC-BY-SA license](https://creativecommons.org/licenses/by-sa/4.0/).

---
## Samples

This folder contains the most extreme days of the test dataset consists of the years 2008, 2015 and 2023.

The most extreme events are defined in terms of number of lightning strikes during the day over the studied domain.

See the README.md file inside the folder for further description.

---

## Scaler

Contains the scaler used to normalize the samples when loading each sample.
The scaler was computed using the full original training dataset.
