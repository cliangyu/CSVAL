# A Guide to Your First Choice: Addressing Cold Start Problem in Vision Active Learning

This repo is developed based on [open-mmlab/mmselfsup](https://github.com/open-mmlab/mmselfsup).
## Installlation
Please see [mmselfsup INSTALLATION](https://mmselfsup.readthedocs.io/en/latest/install.html).

## Dataset Preparation
Datasets can be downloaded at [MedMNIST v2](https://medmnist.com/).

CIFAR-10 TBC.

## Pretrain
Pretrain on all MedMNIST datasets
```
bash tools/medmnist_pretrain.sh
```

## Select initial query
Select initial queries on all MedMNIST datasets
```
bash tools/medmnist_postprocess.sh
```
