# NN Downscaler

Neural image downscaling experiments for learning compact, reconstructable image
representations.

The model receives a high-resolution RGB image, predicts a 2x lower-resolution
RGB image, and trains a decoder to reconstruct the original image from that
predicted bottleneck. The interesting question is not just whether the learned
downscale looks like bicubic or Lanczos, but whether it can keep details that are
useful when the image needs to be brought back up again.

## Contents

- `train.py`: trains the RGB downscaler on DIV2K crops and writes checkpoints.
- `compare.py`: generates side-by-side visual comparisons for classical and
  neural downsampling methods.
- `nn_downscaler.ipynb`: earlier notebook exploration.
- `downsampling_methods_exploration.ipynb`: baseline resampling experiments.

## Experiment

The current model has three pieces:

- an encoder that compresses the HR RGB input by 2x;
- an LR head that emits the learned RGB downsample;
- a decoder that reconstructs HR RGB using only the learned LR output.

Training uses DIV2K HR images. Each batch samples `512x512` HR crops and creates
`256x256` LR targets with bicubic resizing. The loss combines:

- LR loss: how closely the learned downsample matches the bicubic target;
- HR loss: how well the decoder reconstructs the original RGB crop.

This keeps the bottleneck honest: the decoder does not receive hidden encoder
features, so reconstruction quality depends on the actual downscaled image.

## Usage

Install the core dependencies in a Python environment:

```bash
pip install torch torchvision datasets matplotlib numpy pillow notebook tqdm
```

Train the model:

```bash
python train.py
```

Generate comparison images from local DIV2K validation images:

```bash
python compare.py --model checkpoints/checkpoint-9.pth
```

By default, `compare.py` uses full-resolution DIV2K validation images from
`data/div2k/DIV2K_valid_HR`, writes outputs to `comparison_outputs/`, and labels
each downsampling method with its runtime. Use `--crop-size 1024` for quicker
preview grids, or leave the default `--crop-size 0` for the largest available
source images.

## Results

Comparison grids show the HR source next to nearest, bilinear, bicubic, Lanczos,
and neural RGB downscales. They are meant for close visual inspection of edge
behavior, texture retention, color stability, and runtime.

![Street scene comparison](comparison_outputs/sample_004_comparison.png)

![Architecture comparison](comparison_outputs/sample_008_comparison.png)
