# From logistic regression to deep learning : machine learning modeling of lightnings in reanalysis data

This repository is linked to this paper : https://egusphere.copernicus.org/preprints/2026/egusphere-2026-4363/.

---

## Abstract

Most lightning parameterization schemes rely on local approaches where the predictors are the atmospheric variables in the same grid cell as the output. To validate the hypothesis that large-scale thunderstorm clusters – such as mesoscale convective systems – are driven by broad spatial predictor patterns, we model lightning occurrence using architectures capable of processing surrounding grid-cell data rather than relying solely on local point-based inputs. This study develops a deep convolutional neural network (U-Net) to model lightning occurrence across Europe using ERA5 reanalysis data. The model is trained on 13 years of data and evaluated with a leave-one-year-out cross-validation strategy. We compare the performance of the U-Net to several local machine learning models of increasing complexity, including logistic regression, generalized additive model, extreme gradient boosting, and multi-layer perceptron. We find that the U-Net outperforms all single grid cell models in overall performance and on the most extreme events. Through a feature importance study, we find that the most important predictors depend on the model type. Finally we show with a spatial sensitivity study that the U-Net captures mesoscale patterns driving lightning occurrence.

---

## Structure of the repo

```
----| data        samples of the most extreme days from the test dataset
----| models      models described in the paper
----| notebooks   3 notebooks to reproduce and expand the paper's results
----| src         source files used in the notebooks
```

- **data/** — Samples of the most extreme days from the test dataset. See [`data/README.md`](data/README.md) for details on the data format and channel descriptions.
- **models/** — The models described in the paper. See the paper and the README.md in the notebooks folder for more details.
- **notebooks/** — Three notebooks to reproduce and expand some of the paper's results. See [`notebooks/README.md`](notebooks/README.md) for a description of each notebook and how to run them.
- **src/** — Source files and utility functions used across the notebooks.

---

## Installation

### uv build (fastest)

```bash
git clone git@github.com:AdrienBq/lightning_modelling.git
cd lightning_modelling
uv sync
```

### pip build

```bash
git clone git@github.com:AdrienBq/lightning_modelling.git
cd lightning_modelling

# Portable install (uses the loose version ranges in pyproject.toml; CPU-friendly)
pip install .

# OR, to reproduce the authors' exact environment (a fully pinned freeze that
# includes GPU/CUDA wheels — heavier, and intended for a CUDA machine):
pip install -r requirements.txt
```

> **Note:** `pip install .` / `uv sync` and `requirements.txt` are **not** equivalent.
> The first resolves the loose ranges from `pyproject.toml` and works on CPU-only
> machines; `requirements.txt` is a full pinned freeze of the authors' GPU environment.
> The demo notebooks run on CPU, so the portable install is recommended for most users.

---

## Data sources & acknowledgements

### Atmospheric convective parameters — thundeR

Channels 1–5 of the input tensors are derived from atmospheric convective parameters computed with the **thundeR** R package. This postprocessed data was provided by Mateusz Taszarek under contribution from a grant from the Polish National Science Centre (2020/39/D/ST10/00768) and can be made available. Contact him (mateusz.taszarek@amu.edu.pl) for usage information. 


### Lightning location data — Met Office ATDnet

Channel 6 contains lightning location data from the Met Office ATDnet (Arrival Time Difference network):

> ATDnet data provided by the Met Office (2026) under a [CC-BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).

---

## License

The code in this repository is released under the [MIT License](LICENSE.txt).

The data samples in `data/` include a channel derived from Met Office ATDnet data, provided under a [CC-BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/). Any derivative work incorporating that channel must comply with the terms of CC-BY-SA 4.0.

---

## Citation

If you use this code or data in your work, please cite:

```bibtex
@article{Burq2026,
  title = {From logistic regression to deep learning: machine learning modeling of lightning in ERA5 reanalysis data},
  url = {http://dx.doi.org/10.5194/egusphere-2026-4363},
  DOI = {10.5194/egusphere-2026-4363},
  publisher = {Copernicus GmbH},
  author = {Burq,  Adrien and Vrac,  Mathieu and Jouhaud,  Jean and Xing,  Victor and Bouvier,  Victor and Forcadell,  Vincent and Taszarek,  Mateusz and Faranda,  Davide},
  year = {2026},
  month = Aug 
}
```

```bibtex
@manual{thunder2025,
  title  = {thunder: Computation and Visualisation of Atmospheric Convective Parameters},
  author = {Bartosz Czernecki and Mateusz Taszarek and Piotr Szuster},
  year   = {2025},
  note   = {R package version 1.1.5},
  url    = {https://bczernecki.github.io/thundeR/},
}
```

```bibtex
@article{atdnet2006,
author = {Gaffard, Catherine and Nash, John and Atkinson, N. and Bennett, Alec and Callaghan, Greg and Hibbett, Eric and Turp, Myles and Schulz, Wolfgang},
year = {2008},
month = {01},
pages = {},
title = {Observing Lightning Around the Globe from the Surface},
journal = {The Preprints, 20th Interna-tional Lightning Detection Conference}
}
```
