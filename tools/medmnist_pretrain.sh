#!/usr/bin/env bash
PORT=${PORT:-53256}

declare -a StringArray=("derma" "blood" "path")

for dataset in ${StringArray[@]}; do
    CONFIG="configs/selfsup/simclr/simclr_resnet50_2xb2048-coslr-200e_${dataset}.py"
    echo $CONFIG

    python -m torch.distributed.run --nproc_per_node=2 --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --seed 0 \
    --launcher pytorch

done
