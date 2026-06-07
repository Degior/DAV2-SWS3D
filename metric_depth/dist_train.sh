#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

export OPENCV_LOG_LEVEL=ERROR

epochs=50
bs=4
gpus=1
lr=0.0000005
lr_scheduler=constant  # constant или poly
encoder=vitl
dataset=us3d # vkitti
img_size=518
min_depth=0.001
max_depth=250 # 80 for virtual kitti
pretrained_from=../checkpoints/depth_anything_v2_${encoder}.pth
save_path=exp/us3d_berhu_1 # exp/vkitti
port=20596

mkdir -p $save_path

python3 -m torch.distributed.launch \
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
    --min-depth $min_depth \
    --max-depth $max_depth \
    --pretrained-from $pretrained_from \
    --port $port \
    2>&1 | tee -a $save_path/$now.log
