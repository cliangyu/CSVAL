#!/usr/bin/env bash
PORT=${PORT:-53256}

export CUDA_VISIBLE_DEVICES=2

declare -a StringArray=("organa" "derma" "path" "blood")

for dataset in ${StringArray[@]}; do
    # CONFIG="configs/selfsup/simclr/simclr_resnet50_2xb2048-coslr-200e_${dataset}.py"
    CONFIG="configs/selfsup/mocov2/mocov2_resnet50_1xb4096-coslr-200e_${dataset}.py"
    # CONFIG="configs/selfsup/simclr/simclr_resnet50_4xb128-coslr-200e_inlt.py"
    echo $CONFIG

    python -m torch.distributed.run --nproc_per_node=1 --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --seed 0 \
    --launcher pytorch

done
