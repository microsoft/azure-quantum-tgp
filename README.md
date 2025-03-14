# Topological gap protocol: `azure-quantum-tgp`
<img width="308" alt="tgp" align="left" src="https://user-images.githubusercontent.com/6897215/196533626-f573acab-15d3-4fe9-932e-12cae7cc251f.png">

This code performs the analysis as reported in _"InAs-Al Hybrid Devices Passing the Topological Gap Protocol"_ by Microsoft Azure Quantum.

The paper is available on [10.1103/PhysRevB.107.245423](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.245423) and [arXiv:2207.02472](https://arxiv.org/abs/2207.02472).

See the Jupyter notebooks
* [notebooks/stage-one-analysis.ipynb](notebooks/stage-one-analysis.ipynb) as a step-by-step example of the ***Stage 1*** analysis,
* [notebooks/stage-two-analysis.ipynb](notebooks/stage-two-analysis.ipynb) as a step-by-step example of the ***Stage 2*** analysis,
* [notebooks/yield-analysis.ipynb](notebooks/yield-analysis.ipynb) which performs the yield analysis on a large set of simulation data,
* [notebooks/fridge-calibration.ipynb](notebooks/fridge-calibration.ipynb) which shows the fridge calibration data used for high-frequency corrections,
* and [notebooks/paper-figures.ipynb](notebooks/paper-figures.ipynb) which performs _all_ the analysis and generates the plots that appear in the paper.

## Data

We store the raw data in this repository using Git LFS in the [`data/`](data) folder.
Install [Git LFS](https://git-lfs.github.com/) before cloning this repository.

The `data/simulated/yield` folder is 17 GB and is only used in the [`notebooks/yield-analysis.ipynb` Jupyter notebook](notebooks/yield-analysis.ipynb).

To clone the repo *without* this data, use (with `ssh`):
```bash
git lfs clone git@github.com:microsoft/azure-quantum-tgp.git --exclude="data/simulated/yield"
```
or with `https`:
```bash
git lfs clone https://github.com/microsoft/azure-quantum-tgp.git --exclude="data/simulated/yield"
```

## Installation

Install `azure-quantum-tgp` from PyPI with
```bash
pip install azure-quantum-tgp
```

or clone this repository and do a developer install with
```
conda create --name tgp python=3.10  # create a new conda env
conda activate tgp
pip install -e ".[test]"
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
