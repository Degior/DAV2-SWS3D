# Depth Anything V2 for US3D object-height estimation

This directory fine-tunes a Depth Anything V2 encoder to predict object height
from overhead RGB imagery. The maintained datasets are:

- `us3d`: height-only baseline;
- `us3dwh`: optional multi-task experiment with scale and angle heads.

Heights are read from 16-bit rasters, converted from centimetres to metres, and
the value `65535` is treated as nodata. A height of zero is valid background.

## Environment

```bash
pip install -r requirements.txt
```

Training requires CUDA and is launched through `torchrun`. The repository does
not contain the US3D data or private split files.

## Split formats

`us3d` requires only the first two whitespace-separated columns:

```text
/path/to/rgb.tif /path/to/height.tif [anything else is ignored]
```

It neither validates nor opens any trailing columns. `us3dwh` requires exactly
three columns and reads VFLOW JSON to train the optional scale and angle heads:

```text
/path/to/rgb.tif /path/to/height.tif /path/to/vflow.json
```

Metadata schema:

```json
{
  "scale": 1.0,
  "angle": 0.0
}
```

`scale` must be positive. Angles default to radians; pass
`--angle-unit degrees` when the JSON uses degrees. The angle is treated as a
circular azimuth. Horizontal flip is disabled for `us3dwh` because its correct
angle transformation depends on the metadata coordinate convention.

## Training

Edit the paths and hyperparameters in `dist_train.sh`, then run:

```bash
bash dist_train.sh
```

Equivalent single-GPU command:

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --dataset us3d \
  --encoder vitl \
  --train-split dataset/splits/us3d/train.txt \
  --val-split dataset/splits/us3d/val.txt \
  --pretrained-from ../checkpoints/depth_anything_v2_vitl.pth \
  --max-depth 250 \
  --save-path exp/us3d_baseline
```

Zero-height pixels are included when `--min-depth 0` is used. To reproduce the
old behaviour, set `--min-depth 0.001`.

For the auxiliary-head ablation:

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --dataset us3dwh \
  --angle-unit degrees \
  --lambda-scale 0.05 \
  --lambda-angle 0.05 \
  --pretrained-from ../checkpoints/depth_anything_v2_vitl.pth \
  --save-path exp/us3d_aux
```

Training writes:

- `latest.pth`: most recent model and optimizer state;
- `best.pth`: checkpoint with the lowest validation RMSE;
- TensorBoard scalars, including separate scale/angle losses and metrics for
  `us3dwh`.

Continue an interrupted run with `--resume exp/us3d_baseline/latest.pth`.
Do not combine `--resume` with `--pretrained-from`.

## Inference

Both raw Depth Anything state dicts and checkpoints produced by `train.py` are
supported. Auxiliary-head checkpoints are detected automatically.

```bash
python run.py \
  --encoder vitl \
  --load-from exp/us3d_baseline/best.pth \
  --max-depth 250 \
  --img-path /path/to/images \
  --outdir predictions \
  --save-numpy
```

The `.npy` files contain raw predicted heights in metres. PNG visualization uses
a fixed `[--vis-min, --vis-max]` range so colours remain comparable between
images; `--vis-max` defaults to `--max-depth`.

## Validation metrics

MAE, RMSE and bias include valid zero-height pixels. Ratio and logarithmic depth
metrics use only strictly positive target heights. For reliable experiments,
split train and validation data by city/AOI rather than by neighbouring tiles.
