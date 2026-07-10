# Data Samples

This folder contains tensor samples (`.pt` files) used for model training and evaluation. It consists of the 49 most extreme days of the test dataset. Each sample is a PyTorch tensor of shape **(24 timesteps, 6 channels, 101 latitudes, 149 longitudes)**, sourced from two distinct contributors described below.

---

## Data Sources

### Channels 1–5 — Atmospheric Convective Parameters (thundeR)

Five channels are derived from atmospheric convective parameters computed using the **thundeR** R package.

- Channel 1 : Most unstable lifted index (MU_LI)
- Channel 2 : Most unstable mixing ratio (MU_MIXR)
- Channel 3 : Mean realative humidity between 500 and 850 hPa (RH_500850)
- Channel 4 : Convective precipitation (cp)
- Channel 5 : Land-sea mask (lsm)

> Czernecki, B., Taszarek, M., & Szuster, P. (2025). *thunder: Computation and Visualisation of Atmospheric Convective Parameters* (R package version 1.1.5). https://bczernecki.github.io/thundeR/

**BibTeX:**
```bibtex
@manual{thunder2025,
  title  = {thunder: Computation and Visualisation of Atmospheric Convective Parameters},
  author = {Bartosz Czernecki and Mateusz Taszarek and Piotr Szuster},
  year   = {2025},
  note   = {R package version 1.1.5},
  url    = {https://bczernecki.github.io/thundeR/},
}
```

---

### Channel 6 — Lightning Observation (Met Office ATDnet)

The 6th channel contains lightning observation data from the Met Office **ATDnet** (Arrival Time Difference network).
The observations are re-gridded to the ERA5 grid so every point indicates the number of lightning strikes that were recorded by the network within one hour and one grid-cell (0.25° by 0.25°).

> ATDnet data provided by the Met Office (2026) under a [CC-BY-SA license](https://creativecommons.org/licenses/by-sa/4.0/).

Please note: the CC-BY-SA attribution notice required by the Met Office applies to any file, dataset, or derivative work that incorporates this channel. Because the license attribution cannot be embedded in the binary `.pt` tensor files themselves, it is recorded here as the authoritative notice for this dataset.

---

## Channel Summary

| Channel | Source | Description |
|---------|--------|-------------|
| 1–5 | thundeR (Czernecki et al., 2025) | Atmospheric convective parameters |
| 6 | Met Office ATDnet (2026) | Lightning location data |
