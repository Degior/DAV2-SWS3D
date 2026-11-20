#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
checkpoint="$PWD/exp/us3d_fold5_agl_crop518_val518_21001381/best.pth"
control_root="$PWD/exp/astra_round1_20261118_184340_360017/stats/control"
test -s "$checkpoint"
test -s "$control_root/latest/summary.json"
test -s "$control_root/best/summary.json"
run_dir="$PWD/exp/astra_decoder_lr_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run_dir"

../depth/bin/python - "$checkpoint" <<'PY'
import math
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=True)
if ckpt["epoch"] + 1 != 334 or ckpt.get("experiment_steps", 0):
    raise SystemExit("Expected the original AGL checkpoint at epoch 334")
rates = {group["name"]: group["lr"] for group in ckpt["optimizer"]["param_groups"]}
if not math.isclose(rates["backbone"], 5e-7) or not math.isclose(rates["depth_head"], 5e-6):
    raise SystemExit(f"Unexpected starting learning rates: {rates}")
PY

job=$(CHECKPOINT="$checkpoint" RUN_DIR="$run_dir" sbatch --parsable --export=ALL \
    --partition=hiperf --nodelist=tesla-a100 \
    --nodes=1 --ntasks=1 --gres=gpu:a100:8 \
    --cpus-per-task=16 --mem=80000 --time=06:00:00 \
    --job-name=astra-decoder-lr --chdir="$PWD" \
    --output="$run_dir/train_%j.log" <<'SLURM'
#!/bin/bash
set -euo pipefail
export OPENCV_LOG_LEVEL=ERROR OMP_NUM_THREADS=1

../depth/bin/python -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 train.py \
    --dataset us3d --encoder vitl \
    --train-split dataset/splits/us3d/train_fold5.txt \
    --val-split dataset/splits/us3d/val_fold5.txt \
    --resume "$CHECKPOINT" --save-path "$RUN_DIR" \
    --epochs 383 --train-steps 2000 --reset-best \
    --bs 16 --micro-batch-size 8 \
    --lr 5e-7 --resume-lr 5e-7 --lr-scheduler constant \
    --resume-decoder-lr 1.5e-5 --decoder-lr-ramp-steps 200 \
    --height-loss l1-agl --no-agl-focal --no-mag-scale --agl-height-weights 1 1 1 \
    --no-rotate90 --photometric-prob 0 \
    --height-activation softplus --attention-backend sdpa \
    --sampling-mode native --img-size 518 \
    --val-img-size 518 --val-tile-overlap 126 --val-tile-batch-size 4 \
    --height-crop-prob 0.6 --height-crop-mid-first-prob 0.5 \
    --hflip-prob 0.5 --min-depth 0.1 --zero-height-after-epochs 15 --max-depth 250 \
    --num-workers 2 --seed 42
SLURM
)
job=${job%%;*}
printf "Training job: %s\n" "$job"

evaluation=$(RUN_DIR="$run_dir" CONTROL_ROOT="$control_root" sbatch --parsable --export=ALL \
    --partition=hiperf --nodelist=tesla-a100 \
    --nodes=1 --ntasks=1 --gres=gpu:a100:1 \
    --cpus-per-task=4 --mem=24000 --time=02:00:00 \
    --dependency="afterok:$job" --job-name=astra-decoder-eval --chdir="$PWD" \
    --output="$run_dir/eval_%j.log" <<'SLURM'
#!/bin/bash
set -euo pipefail
export OPENCV_LOG_LEVEL=INFO OMP_NUM_THREADS=2

for kind in latest best; do
    ../depth/bin/python -u collect_height_stats.py \
        --checkpoint "$RUN_DIR/$kind.pth" \
        --val-split dataset/splits/us3d/val_fold5.txt \
        --tile-size 518 --tile-overlap 126 --tile-batch-size 4 \
        --num-workers 2 --no-figures --output "$RUN_DIR/stats/$kind"
done

../depth/bin/python - "$RUN_DIR" "$CONTROL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

run, control = map(Path, sys.argv[1:])


def read_stats(path):
    report = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    images = [json.loads(line)["image"] for line in (path / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]
    if not report["complete_split"] or len(images) != 592 or len(set(images)) != 592:
        raise SystemExit(f"Incomplete fold 5 evaluation: {path}")
    return report, images


results = {}
for kind in ("latest", "best"):
    candidate, images = read_stats(run / "stats" / kind)
    reference, control_images = read_stats(control / kind)
    if images != control_images or candidate["config"] != reference["config"]:
        raise SystemExit("Evaluation images or inference settings differ from the control")
    if kind == "latest" and (candidate["experiment_steps"], reference["experiment_steps"]) != (2000, 2000):
        raise SystemExit("Expected 2000 updates in both runs")
    metrics = candidate["height_groups"]["all_valid"]["mean_per_image"]
    baseline = reference["height_groups"]["all_valid"]["mean_per_image"]
    rmse_gain = baseline["rmse"] - metrics["rmse"]
    mae_change = metrics["mae"] - baseline["mae"]
    results[kind] = {
        "control_metrics": baseline, "decoder_lr_metrics": metrics,
        "control_steps": reference["experiment_steps"], "decoder_lr_steps": candidate["experiment_steps"],
        "rmse_gain_m": rmse_gain, "mae_change_m": mae_change,
        "passes_practical_threshold": rmse_gain >= 0.02 and mae_change <= 0.01,
    }
    print(f"{kind}: RMSE gain {rmse_gain:+.6f} m, MAE change {mae_change:+.6f} m")
(run / "comparison.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
PY
SLURM
)
printf "Evaluation job: %s\n" "${evaluation%%;*}"
printf "Experiment directory: %s\n" "$run_dir"
