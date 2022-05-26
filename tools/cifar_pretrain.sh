#!/usr/bin/env bash
PORT=${PORT:-53256}

export CUDA_VISIBLE_DEVICES=1

# declare -a StringArray=("organa" "path" "blood")
# declare -a StringArray=("cifar10")
declare -a StringArray=("cifar10lt")

for dataset in ${StringArray[@]}; do
    # CONFIG="configs/selfsup/mocov2/mocov2_resnet50_1xb4096-coslr-200e_${dataset}.py"
    CONFIG="configs/selfsup/mocov2/mocov2_resnet50_1xb512-coslr-800e_${dataset}.py"
    echo $CONFIG

    python -m torch.distributed.run --nproc_per_node=1 --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG \
    --seed 0 \
    --launcher pytorch

done
