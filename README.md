# NN Downscaler

Initial exploration of image downsampling methods, starting with classical
nearest-neighbor, bilinear, cubic, and Lanczos approaches in a Jupyter notebook.

The next phase is to develop neural-network methods for learned downsampling and
upsampling, using these baseline methods as references for visual and numerical
comparison.

## Contents

- `downsampling_methods_exploration.ipynb`: baseline downsampling exploration
  and visual comparisons on the Set5 dataset.

## Setup

This project currently uses a local Python virtual environment and Jupyter.
Install the notebook dependencies as needed before running the exploration:

```bash
pip install datasets pillow torch matplotlib numpy notebook
```
