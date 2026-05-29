# From logistic regression to deep learning : machine learning modeling of lightnings in reanalysis data

This repository is linked to the paper [INSERT PAPER LINK TO PREPRINT ONCE SUBMITTED].

---

## Abstract

*[Insert abstract here once submitted.]*

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

Clone the repository and install the dependencies:

```bash
git clone [INSERT REPO URL]
cd [INSERT REPO NAME]
pip install -r requirements.txt
```

---

## Data sources & acknowledgements

### Atmospheric convective parameters — thundeR

Channels 1–5 of the input tensors are derived from atmospheric convective parameters computed with the **thundeR** R package:

> Czernecki, B., Taszarek, M., & Szuster, P. (2025). *thunder: Computation and Visualisation of Atmospheric Convective Parameters* (R package version 1.1.5). https://bczernecki.github.io/thundeR/

```bibtex
@manual{thunder2025,
  title  = {thunder: Computation and Visualisation of Atmospheric Convective Parameters},
  author = {Bartosz Czernecki and Mateusz Taszarek and Piotr Szuster},
  year   = {2025},
  note   = {R package version 1.1.5},
  url    = {https://bczernecki.github.io/thundeR/},
}
```

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

*[Insert BibTeX citation once the preprint is available.]*
