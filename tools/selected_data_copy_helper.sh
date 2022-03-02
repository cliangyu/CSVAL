#!/usr/bin/env bash

declare -a StringArray=("path" "pneumonia" "oct" "breast" "derma" "blood" )

for dataset in ${StringArray[@]}; do
    # SELECTION_FILES="work_dirs/selfsup/simclr_resnet50_4xb1024-coslr-200e_${dataset}/data_selection/."
    # TARGET_DIR="${HOME}/data_selection/${dataset}mnist"

    SELECTION_FILES="work_dirs/selfsup/simclr_resnet50_4xb1024-coslr-200e_${dataset}/data_selection/${dataset}mnist_sorted_idx.npy"
    TARGET_DIR="${HOME}/data_selection/easy_sorted_idx"

    mkdir -p $TARGET_DIR
    cp -R $SELECTION_FILES $TARGET_DIR

done
