#!/usr/bin/env bash
PORT=${PORT:-29500}

declare -a Dataset=("path" "derma" "blood")
declare -a Metric=("ambiguity" "easy" "hard")

for dataset in ${Dataset[@]}; do
    for metric in ${Metric[@]}; do
        CONFIG="configs/selfsup/simclr/simclr_resnet50_4xb1024-coslr-200e_${dataset}.py"
        echo $CONFIG
        CHECKPOINT="work_dirs/selfsup/simclr_resnet50_4xb1024-coslr-200e_${dataset}/latest.pth"
        echo $CHECKPOINT
        DATASET_CONFIG="configs/benchmarks/classification/medmnist/${dataset}mnist.py"
        echo $DATASET_CONFIG
        SORTED_IDX_FILE="/media/ntu/volume1/home/s121md302_06/data_selection/${metric}_sorted_idx/${dataset}mnist_sorted_idx.npy"
        echo $SORTED_IDX_FILE

        python -m torch.distributed.run --nproc_per_node=1 --master_port=$PORT \
        $(dirname "$0")/analysis_tools/visualize_umap.py $CONFIG \
        --checkpoint $CHECKPOINT \
        --dataset_config $DATASET_CONFIG \
        --layer_ind 4 \
        --pool_type adaptive \
        --max_num_sample_plot 20000 \
        --sorted_idx_file $SORTED_IDX_FILE \
        --plot_name "${dataset}_${metric}" \
        --launcher pytorch
    done
done
