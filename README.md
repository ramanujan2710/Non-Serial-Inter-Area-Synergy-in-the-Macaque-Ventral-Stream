#  Non-Serial Inter-Area Synergy in the Macaque Ventral Stream

This repository contains the code and Jupyter Notebooks for the NeurIPS submission.

## Files
- `CGMIMonkeyN.ipynb`
- `MonkeyF.ipynb`
- `MonkeyN.ipynb`
- `internal_pid.py`

## Environment and Dependencies
All code was designed to be executed on **Google Colaboratory (Colab)** using an NVIDIA A100 GPU with the High-RAM configuration. 
The notebooks rely on the default Google Colab Python 3 environment packages (including `numpy`, `scipy`, `matplotlib`, and standard deep learning libraries). 

## Data
The neural population dataset used in these notebooks is derived from prior published work (accessible via *Neuron*, 2024: https://doi.org/10.1016/j.neuron.2024.12.003). 
To reproduce the analyses, ensure the dataset is downloaded and the paths in the Google Drive mount (`/content/drive/MyDrive/...`) are updated to point to your local or Drive-hosted copies of the data.

## Paper
This code accompanies:

> Arun Ram Ponnambalam, Krishnan Venkiteswaran Pottore.
> *Inferotemporal Cortex Collaborates Before It Codes: Non-Serial Inter-Area Synergy in the Macaque Ventral Stream.*
> Advances in Neural Information Processing Systems (NeurIPS), 2026 (poster, to appear).

The paper is not yet publicly available; a link will be added here when it is.

```bibtex
@inproceedings{ponnambalam2026inferotemporal,
  title     = { Non-Serial Inter-Area Synergy in the Macaque Ventral Stream},
  author    = {Ponnambalam, Arun Ram and Pottore, Krishnan Venkiteswaran},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
