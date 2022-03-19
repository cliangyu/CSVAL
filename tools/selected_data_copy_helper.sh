#!/usr/bin/env bash

declare -a StringArray=("path" "pneumonia" "oct" "breast" "derma" "blood" )
declare -a METRICS=("easy" "ambiguous" "hard")

for dataset in ${StringArray[@]}; do
    for metric in ${METRICS[@]}; do
        IDX_FILE="work_dirs/selfsup/simclr_resnet50_2xb2048-coslr-200e_${dataset}/data_selection/${dataset}mnist_${metric}_sorted_idx.npy"
        TARGET_DIR="${HOME}/data_selection/20220320"

        mkdir -p $TARGET_DIR
        cp -R $IDX_FILE $TARGET_DIR
    done
done
