---
title: Thera Arbitrary-Scale Super-Resolution
emoji: 🔥
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Thera Arbitrary-Scale Super-Resolution
This is an interactive demo for our paper "Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields
" [(arXiV link)](https://arxiv.org/pdf/2311.17643) [(code link)](https://github.com/prs-eth/thera).

## Run locally
If you want to run the demo locally, you need a Python 3.10 environment (e.g., installed via conda) on Linux as well as an NVIDIA GPU. Then install packages via pip:
```bash
> pip install --upgrade pip
> pip install -r requirements.txt
```

Then, start the Gradio server like this:
```bash
> python app.py
```

The server should bind to port `7860` by default.

## Useful XLA flags
* Disable pre-allocation of entire VRAM: `XLA_PYTHON_CLIENT_PREALLOCATE=false`
* Disable jitting for debugging: `JAX_DISABLE_JIT=1`

## Citation

If you found our work helpful, consider citing our paper 😊:

```
@article{becker2025thera,
  title={Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields},
  author={Becker, Alexander and Daudt, Rodrigo Caye and Narnhofer, Dominik and Peters, Torben and Metzger, Nando and Wegner, Jan Dirk and Schindler, Konrad},
  journal={arXiv preprint arXiv:2311.17643},
  year={2025}
}
```