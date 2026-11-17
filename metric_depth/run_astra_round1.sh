#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
checkpoint="$PWD/exp/us3d_fold5_agl_crop518_val518_21001381/best.pth"

../depth/bin/python - "$checkpoint" <<'PY'
import math
import sys
import torch
from dataset.us3d import read_us3d_split
from prepare_agl_city_stats import validate_agl_city_stats

checkpoint = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=True)
if not {"model", "optimizer", "epoch", "args"} <= checkpoint.keys():
    raise SystemExit("A complete training checkpoint is required")
args = checkpoint["args"]
expected = {"dataset": "us3d", "encoder": "vitl", "sampling_mode": "native",
            "img_size": 518, "height_activation": "softplus",
            "attention_backend": "sdpa", "height_loss": "l1-agl"}
for key, value in expected.items():
    if args.get(key) != value:
        raise SystemExit(f"Unexpected checkpoint {key}: {args.get(key)}")
if checkpoint["epoch"] + 1 != 334 or checkpoint.get("experiment_steps", 0):
    raise SystemExit("Expected the original best AGL checkpoint from epoch 334")
if args.get("mag_scale") or args.get("agl_focal"):
    raise SystemExit("Expected ordinary AGL without auxiliary heads")
validate_agl_city_stats(args["agl_city_stats"], "dataset/splits/us3d/train_fold5.txt", 250.0)
train = read_us3d_split("dataset/splits/us3d/train_fold5.txt")
val = read_us3d_split("dataset/splits/us3d/val_fold5.txt")
if len(train) != 5331 or len(val) != 592:
    raise SystemExit("Expected the 5331/592 fold 5 split")
if set(row[0] for row in train) & set(row[0] for row in val):
    raise SystemExit("Training and validation images overlap")
rates = [group["lr"] for group in checkpoint["optimizer"]["param_groups"]]
if len(rates) != 2 or not all(math.isclose(a, b) for a, b in zip(rates, (5e-7, 5e-6))):
    raise SystemExit(f"Unexpected optimizer learning rates: {rates}")
print("Verified source epoch 334, fold 5 and optimizer state", flush=True)
PY

experiment_root="$PWD/exp/astra_round1_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$experiment_root/logs"

baseline=$(EXPERIMENT_ROOT="$experiment_root" CHECKPOINT="$checkpoint" \
    sbatch --parsable --export=ALL \
        --partition=hiperf --nodelist=tesla-a100 \
        --nodes=1 --ntasks=1 --gres=gpu:a100:1 \
        --cpus-per-task=4 --mem=24000 --time=02:00:00 \
        --job-name=astra-baseline --chdir="$PWD" \
        --output="$experiment_root/logs/baseline_%j.log" <<'SLURM'
#!/bin/bash
set -euo pipefail
export OPENCV_LOG_LEVEL=INFO OMP_NUM_THREADS=2
../depth/bin/python -u collect_height_stats.py \
    --checkpoint "$CHECKPOINT" \
    --val-split dataset/splits/us3d/val_fold5.txt \
    --tile-size 518 --tile-overlap 126 --tile-batch-size 4 \
    --num-workers 2 --no-figures \
    --output "$EXPERIMENT_ROOT/stats/baseline"
SLURM
)
baseline=${baseline%%;*}
printf 'Baseline validation job: %s\n' "$baseline"

submit_training() {
    local name="$1" rotate="$2" photo="$3" mid="$4" high="$5" dependency="$6"
    EXPERIMENT_ROOT="$experiment_root" CHECKPOINT="$checkpoint" \
    RUN_NAME="$name" ROTATE_FLAG="$rotate" PHOTO_PROB="$photo" \
    MID_WEIGHT="$mid" HIGH_WEIGHT="$high" \
    sbatch --parsable --export=ALL \
        --partition=hiperf --nodelist=tesla-a100 \
        --nodes=1 --ntasks=1 --gres=gpu:a100:8 \
        --cpus-per-task=16 --mem=80000 --time=06:00:00 \
        --dependency="afterok:$dependency" --job-name="astra-$name" \
        --chdir="$PWD" --output="$experiment_root/logs/${name}_%j.log" <<'SLURM'
#!/bin/bash
set -euo pipefail
export OPENCV_LOG_LEVEL=ERROR OMP_NUM_THREADS=1
../depth/bin/python -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 train.py \
    --dataset us3d --encoder vitl \
    --train-split dataset/splits/us3d/train_fold5.txt \
    --val-split dataset/splits/us3d/val_fold5.txt \
    --resume "$CHECKPOINT" --save-path "$EXPERIMENT_ROOT/$RUN_NAME" \
    --epochs 383 --train-steps 2000 --reset-best \
    --bs 16 --micro-batch-size 8 \
    --lr 5e-7 --resume-lr 5e-7 --lr-scheduler constant \
    --height-loss l1-agl --no-agl-focal --no-mag-scale \
    --agl-height-weights 1 "$MID_WEIGHT" "$HIGH_WEIGHT" \
    "$ROTATE_FLAG" --photometric-prob "$PHOTO_PROB" \
    --height-activation softplus --attention-backend sdpa \
    --sampling-mode native --img-size 518 \
    --val-img-size 518 --val-tile-overlap 126 --val-tile-batch-size 4 \
    --height-crop-prob 0.6 --height-crop-mid-first-prob 0.5 \
    --hflip-prob 0.5 --min-depth 0.1 --zero-height-after-epochs 15 --max-depth 250 \
    --num-workers 2 --seed 42
SLURM
}

dependency="$baseline"
for name in control height_weighted rotate90 photometric; do
    case "$name" in
        control) rotate=--no-rotate90; photo=0; mid=1; high=1 ;;
        height_weighted) rotate=--no-rotate90; photo=0; mid=1.5; high=2 ;;
        rotate90) rotate=--rotate90; photo=0; mid=1; high=1 ;;
        photometric) rotate=--no-rotate90; photo=0.5; mid=1; high=1 ;;
    esac
    job=$(submit_training "$name" "$rotate" "$photo" "$mid" "$high" "$dependency")
    dependency=${job%%;*}
    printf '%s training job: %s\n' "$name" "$dependency"
done

evaluation=$(EXPERIMENT_ROOT="$experiment_root" sbatch --parsable --export=ALL \
    --partition=hiperf --nodelist=tesla-a100 \
    --nodes=1 --ntasks=1 --gres=gpu:a100:1 \
    --cpus-per-task=4 --mem=24000 --time=08:00:00 \
    --dependency="afterok:$dependency" --job-name=astra-evaluate \
    --chdir="$PWD" --output="$experiment_root/logs/evaluation_%j.log" <<'SLURM'
#!/bin/bash
set -euo pipefail
export OPENCV_LOG_LEVEL=INFO OMP_NUM_THREADS=2
for name in control height_weighted rotate90 photometric; do
    for kind in best latest; do
        ../depth/bin/python -u collect_height_stats.py \
            --checkpoint "$EXPERIMENT_ROOT/$name/$kind.pth" \
            --val-split dataset/splits/us3d/val_fold5.txt \
            --tile-size 518 --tile-overlap 126 --tile-batch-size 4 \
            --num-workers 2 --no-figures \
            --output "$EXPERIMENT_ROOT/stats/$name/$kind"
    done
done

../depth/bin/python - "$EXPERIMENT_ROOT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
def read(path):
    data = json.loads((path / "summary.json").read_text())
    records = [json.loads(line) for line in (path / "per_image.jsonl").read_text().splitlines()]
    if not data["complete_split"] or data["validation_images"] != 592:
        raise SystemExit(f"Incomplete validation: {path}")
    return data, records

baseline, _ = read(root / "stats/baseline")
result = {
    "baseline": baseline["height_groups"]["all_valid"]["mean_per_image"],
    "criterion": "RMSE gain >= 0.02 m over matching control; MAE increase <= 0.01 m",
    "comparisons": {},
}
for kind in ("best", "latest"):
    control, control_records = read(root / "stats/control" / kind)
    reference = control["height_groups"]["all_valid"]["mean_per_image"]
    rows = {}
    for name in ("control", "height_weighted", "rotate90", "photometric"):
        data, records = read(root / "stats" / name / kind)
        if [r["image"] for r in records] != [r["image"] for r in control_records]:
            raise SystemExit("Per-image validation order differs")
        if kind == "latest" and data.get("experiment_steps") != 2000:
            raise SystemExit(f"Incorrect step budget: {name}")
        metrics = data["height_groups"]["all_valid"]["mean_per_image"]
        gain = reference["rmse"] - metrics["rmse"]
        mae_change = metrics["mae"] - reference["mae"]
        differences = [r["all_valid"]["metrics"]["rmse"] - c["all_valid"]["metrics"]["rmse"]
                       for r, c in zip(records, control_records)]
        rows[name] = {
            "metrics": metrics, "rmse_gain_vs_control_m": gain, "mae_change_vs_control_m": mae_change,
            "passes_practical_threshold": name != "control" and gain >= 0.02 and mae_change <= 0.01,
            "paired_rmse_wins": sum(value < 0 for value in differences),
            "paired_rmse_mean_delta_m": statistics.mean(differences),
            "experiment_steps": data.get("experiment_steps"),
            "height_groups": data["height_groups"], "cities": data["cities"],
        }
        print(f"{kind} {name}: RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, "
              f"gain={gain:+.6f}, MAE change={mae_change:+.6f}", flush=True)
    result["comparisons"][kind] = rows
(root / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"Saved comparison: {root / 'comparison.json'}", flush=True)
PY
SLURM
)
printf 'Final evaluation job: %s\n' "${evaluation%%;*}"
printf 'Experiment directory: %s\n' "$experiment_root"

