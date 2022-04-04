#!/usr/bin/env bash
PORT=${PORT:-29500}

# declare -a StringArray=("path" "blood" "derma")
declare -a StringArray=("blood")
declare -a METRICS=("easy" "ambiguous" "hard")
declare -a TEMPI=("0.01")
# declare -a K=("100" "200")

for tempi in ${TEMPI[@]}; do
    # for k in ${K[@]}; do
        for dataset in ${StringArray[@]}; do
            # CONFIG="configs/selfsup/simclr/simclr_resnet50_2xb2048-coslr-200e_${dataset}.py"
            CONFIG="configs/selfsup/mocov2/mocov2_resnet50_1xb512-coslr-200e_${dataset}.py"
            echo $CONFIG
            # WORK_DIR="work_dirs/selfsup/simclr_resnet50_2xb2048-coslr-200e_tempi${tempi}_${dataset}"
            WORK_DIR="work_dirs/selfsup/mocov2_resnet50_1xb512-coslr-200e_${dataset}"
            echo $WORK_DIR
            CHECKPOINT="${WORK_DIR}/latest.pth"
            echo $CHECKPOINT
            DATASET_CONFIG="configs/benchmarks/classification/medmnist/${dataset}mnist.py"
            echo $DATASET_CONFIG

            python -m torch.distributed.run --nproc_per_node=1 --master_port=$PORT \
            $(dirname "$0")/analysis_tools/visualize_umap.py $CONFIG \
            --work_dir $WORK_DIR \
            --checkpoint $CHECKPOINT \
            --dataset_config $DATASET_CONFIG \
            --layer_ind 4 \
            --pool_type adaptive \
            --launcher pytorch

            python $(dirname "$0")/analysis_tools/visualize_cartography.py $CONFIG \
            --work_dir "${WORK_DIR}/cartography" \
            --dataset_config=$DATASET_CONFIG

            for metric in ${METRICS[@]}; do
                IDX_FILE="${WORK_DIR}/data_selection/${dataset}mnist_${metric}_sorted_idx.npy"
                echo $IDX_FILE

                python $(dirname "$0")/data_selection/select_cartography.py $CONFIG \
                --work_dir "${WORK_DIR}/data_selection" \
                --pseudo_labels "${WORK_DIR}/clustering_pseudo_labels/${dataset}mnist_train.npy" \
                --training_dynamics "${WORK_DIR}/cartography/training_td_metrics.jsonl" \
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
    # done
done
