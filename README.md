# NfgTransformer: Equivariant Representation Learning for Normal-form Games

This repository provides a reference implementation of the network architecture
described in the ICLR 2024 paper
[NfgTransformer: Equivariant Representation Learning for Normal-form Games](https://openreview.net/forum?id=4YESQqIys7).

## Installation

No installation is needed when interacting with the `run_experiment.ipynb`
notebook as it installs this package from GitHub sources directly.

For local installation, following the steps below:

Clone the repository:

`git clone https://github.com/google-deepmind/nfg_transformer.git`

Switch to the project directory:

`cd nfg_transformer`

Install dependencies:

`pip install -e .`

You can then run the tests to verify that all modules are working as intended
(requires `pytest` to be installed):

`python -m pytest nfg_transformer/*test.py`

## Usage

The NfgTransformer offers general-purpose equivariant representation learning of
normal-form games and can be used for equilibrium solving, max-deviation gain
estimation and payoff prediction of n-player general-sum normal-form games.

`run_experiment.ipynb` implements a self-contained supervised learning
experiment for all these tasks and we recommend following along the notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/nfg_transformer/blob/master/run_experiment.ipynb)

## Citing this work

To cite this work:

```bibtex
@inproceedings{
anonymous2024nfgtransformer,
  title={NfgTransformer: Equivariant Representation Learning for Normal-form Games},
  author={Anonymous},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=4YESQqIys7}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
