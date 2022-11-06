#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

PORT=${PORT:-29500}

declare -a StringArray=("cifar10lt")
declare -a METRICS=("easy" "ambiguous" "hard")
declare -a EPOCH=110
declare -a TEMPI=("0.05")

for tempi in ${TEMPI[@]}; do
    for epoch in ${EPOCH[@]}; do
        for dataset in ${StringArray[@]}; do
            CONFIG="configs/selfsup/mocov2/mocov2_resnet50_1xb512-coslr-800e_${dataset}.py"
            echo $CONFIG
            WORK_DIR="work_dirs/selfsup/mocov2_resnet50_1xb512-coslr-800e_${dataset}"
            echo $WORK_DIR
            CHECKPOINT="${WORK_DIR}/latest.pth"
            # CHECKPOINT="${WORK_DIR}/epoch_${epoch}.pth"
            echo $CHECKPOINT
            DATASET_CONFIG="configs/benchmarks/classification/cifar/${dataset}.py"
            echo $DATASET_CONFIG

            python -m torch.distributed.run --nproc_per_node=1 --master_port=$PORT \
            $(dirname "$0")/analysis_tools/visualize_umap.py $CONFIG \
            --work_dir $WORK_DIR \
            --checkpoint $CHECKPOINT \
            --dataset_config $DATASET_CONFIG \
            --layer_ind 4 \
            --pool_type adaptive \
            --launcher pytorch \
            --no_point_selected \
            --plot_name "${dataset}_${epoch}" \
            --overwrite_features \


            python $(dirname "$0")/analysis_tools/visualize_cartography.py $CONFIG \
            --work_dir "${WORK_DIR}/cartography" \
            --dataset_config=$DATASET_CONFIG \
            --temperature $tempi

            for metric in ${METRICS[@]}; do
                IDX_FILE="${WORK_DIR}/data_selection/${dataset}_${metric}_sorted_idx.npy"
                echo $IDX_FILE

                python $(dirname "$0")/data_selection/select_cartography.py $CONFIG \
                --work_dir "${WORK_DIR}/data_selection" \
                --pseudo_labels "${WORK_DIR}/clustering_pseudo_labels/${dataset}.npy" \
                --training_dynamics "${WORK_DIR}/cartography/training_${tempi}_td_metrics.jsonl" \
                --dataset_config=$DATASET_CONFIG \
                --metric=$metric

                python -m torch.distributed.run --nproc_per_node=1 --master_port=$PORT \
                $(dirname "$0")/analysis_tools/visualize_umap.py $CONFIG \
                --work_dir $WORK_DIR \
                --checkpoint $CHECKPOINT \
                --dataset_config $DATASET_CONFIG \
                --layer_ind 4 \
                --max_num_sample_plot 100000 \
                --pool_type adaptive \
                --sorted_idx_file $IDX_FILE \
                --plot_name "${dataset}_${metric}" \
                --launcher pytorch
            done
        done
    done
done
