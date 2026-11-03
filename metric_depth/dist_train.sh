#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

export OPENCV_LOG_LEVEL=ERROR

epochs=50
bs=4
gpus=1
lr=0.0000005
lr_scheduler=constant
encoder=vitl
dataset=us3d # use us3dwh only for the scale/angle ablation
img_size=518
sampling_mode=native
val_tile_overlap=126
min_depth=0.1
zero_height_after_epochs=15
max_depth=250
pretrained_from=../checkpoints/depth_anything_v2_${encoder}.pth
save_path=exp/us3d_native_zero_after15_${now}
port=20596

mkdir -p $save_path

torchrun \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$port \
    train.py \
    --epochs $epochs \
    --encoder $encoder \
    --bs $bs \
    --lr $lr \
    --lr-scheduler $lr_scheduler \
    --save-path $save_path \
    --dataset $dataset \
    --img-size $img_size \
    --sampling-mode $sampling_mode \
    --val-tile-overlap $val_tile_overlap \
    --min-depth $min_depth \
    --zero-height-after-epochs $zero_height_after_epochs \
    --max-depth $max_depth \
    --pretrained-from $pretrained_from \
    2>&1 | tee -a $save_path/$now.log
