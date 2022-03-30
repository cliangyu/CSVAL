#!/usr/bin/env bash
PORT=${PORT:-53256}

export CUDA_VISIBLE_DEVICES=2,3

declare -a StringArray=("derma")

for dataset in ${StringArray[@]}; do
    CONFIG="configs/selfsup/simclr/simclr_resnet50_2xb2048-coslr-200e_${dataset}.py"
    echo $CONFIG

    python -m torch.distributed.run --nproc_per_node=2 --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --seed 0 \
    --launcher pytorch

done
