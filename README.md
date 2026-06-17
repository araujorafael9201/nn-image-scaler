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
- `dataset.py`: DIV2K download/path helpers and crop dataset.
- `model.py`: checkpoint-compatible RGB downscaler model definition.
- `nn_downscaler.ipynb`: notebook-local experimental training path.
- `downsampling_methods_exploration.ipynb`: baseline resampling experiments.

## Experiment

The current model has three pieces:

- an encoder that compresses the HR RGB input by 2x;
- an LR head that emits the learned RGB downsample;
- a decoder that reconstructs HR RGB using only the learned LR output.

Training uses DIV2K HR images. On first run, `train.py` extracts three random
`512x512` crops from each source image and saves them to `data/div2k/patches_512/`
as full-quality PNGs. Subsequent runs skip this step and load from the cache.
During training, each `__getitem__` call lazily opens one of the pre-extracted
patches and picks randomly among the three, so the model sees different crops each
epoch without holding the full dataset in RAM.

Each batch creates `256x256` LR targets from those HR patches via bicubic
resizing. The loss combines:

- LR loss: how closely the learned downsample matches the bicubic target;
- HR loss: how well the decoder reconstructs the original RGB crop.

This keeps the bottleneck honest: the decoder does not receive hidden encoder
features, so reconstruction quality depends on the actual downscaled image.

## Usage

Install the core dependencies in a Python environment:

```bash
pip install torch torchvision matplotlib numpy pillow notebook tqdm
```

Train the model:

```bash
python train.py
```

All training options are configurable via CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--n-residual-blocks` | `4` | Number of residual blocks in the model |
| `--checkpoint` | — | State dict (`.pth`) to warm-start from before training |
| `--batch-size` | `48` | Training and validation batch size |
| `--epochs` | `50` | Total number of training epochs |
| `--checkpoint-interval` | `5` | Epochs between validation runs and checkpoint saves |

Generate comparison images from local test inputs:

```bash
python compare.py --model path/to/model.pth
```

Use `--model` to point at the checkpoint or `model.pth` file produced by your
own training run. By default, `compare.py` reads images from `test_inputs/`,
writes outputs to `comparison_outputs/`, and labels each downsampling or
upscaling method with its runtime. The default inputs are curated stress images
with small text, grids, diagonal lines, fine texture, hard color edges, and
repeating patterns. It uses the full image by default. Use `--crop-size 1024` to
explicitly take a centered crop before comparison, or `--max-samples 0` to
compare every image in the input directory.

## Results

`compare.py` writes two comparison grids per sample:

- Downscaling: full-resolution input beside low-resolution outputs from nearest,
  bilinear, bicubic, Lanczos, and the neural downscaler.
- Upscaling: full-resolution input is first downsampled once with Lanczos; that
  same LR image is then upscaled by nearest, bilinear, bicubic, Lanczos, and the
  neural decoder.

The upscaling grid is intentionally decoder-only for the neural method. The
neural downscaler does not create the LR input for that comparison.

![Diagonal line downscale comparison](comparison_outputs/sample_001_comparison.png)

![Small text upscale comparison](comparison_outputs/sample_007_upscale_comparison.png)
