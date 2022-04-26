#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

PORT=${PORT:-29500}

# declare -a StringArray=("path" "blood" "derma")
# declare -a StringArray=("organa" "path" "blood" "derma")
declare -a StringArray=("derma")
declare -a METRICS=("easy" "ambiguous" "hard")
declare -a EPOCH=110
# declare -a EPOCH=$(seq 5 5 30)
declare -a TEMPI=("0.05")
# declare -a TEMPI=("0.2" "0.1" "0.05" "0.01")
# declare -a K=("100" "200")

for tempi in ${TEMPI[@]}; do
    # for k in ${K[@]}; do
    for epoch in ${EPOCH[@]}; do
        for dataset in ${StringArray[@]}; do
            # CONFIG="configs/selfsup/simclr/simclr_resnet50_2xb2048-coslr-200e_${dataset}.py"
            CONFIG="configs/selfsup/mocov2/mocov2_resnet50_1xb4096-coslr-200e_${dataset}.py"
            # CONFIG="configs/selfsup/mocov2/mocov2_resnet50_1xb4096-coslr-100e_organa_wo_repeat.py"
            echo $CONFIG
            # WORK_DIR="work_dirs/selfsup/simclr_resnet50_2xb2048-coslr-200e_tempi${tempi}_${dataset}"
            # WORK_DIR="work_dirs/selfsup/mocov2_resnet50_1xb4096-coslr-100e_organa_wo_repeat"
            WORK_DIR="work_dirs/selfsup/mocov2_resnet50_1xb4096-coslr-200e_${dataset}"
            echo $WORK_DIR
            # CHECKPOINT="${WORK_DIR}/latest.pth"
            CHECKPOINT="${WORK_DIR}/epoch_${epoch}.pth"
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
            --launcher pytorch \
            --no_point_selected \
            --plot_name "${dataset}_${epoch}" \
            --overwrite_features \
            --overwrite_pseudo_label

            python $(dirname "$0")/analysis_tools/visualize_cartography.py $CONFIG \
            --work_dir "${WORK_DIR}/cartography" \
            --dataset_config=$DATASET_CONFIG \
            --temperature $tempi

            for metric in ${METRICS[@]}; do
                IDX_FILE="${WORK_DIR}/data_selection/${dataset}mnist_${metric}_sorted_idx.npy"
                echo $IDX_FILE

                python $(dirname "$0")/data_selection/select_cartography.py $CONFIG \
                --work_dir "${WORK_DIR}/data_selection" \
                --pseudo_labels "${WORK_DIR}/clustering_pseudo_labels/${dataset}mnist_train.npy" \
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
    # done
done
