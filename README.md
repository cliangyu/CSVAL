# A Guide to Your First Choice: Addressing Cold Start Problem in Vision Active Learning

This repo is developed based on [open-mmlab/mmselfsup](https://github.com/open-mmlab/mmselfsup).
## Installlation
Please see [mmselfsup INSTALLATION](https://mmselfsup.readthedocs.io/en/latest/install.html).

## Dataset Preparation

All datasets can be auto downloaded in this repo.

MedMNIST can also be downloaded at [MedMNIST v2](https://medmnist.com/).

CIFAR-10-LT is generated in this repo with a fixed seed.

## Pretrain
Pretrain on all MedMNIST datasets
```
bash tools/medmnist_pretrain.sh
```

Pretrain on CIFAR-10-LT
```
bash tools/cifar_pretrain.sh
```

## Select initial query
Select initial queries on all MedMNIST datasets
```
bash tools/medmnist_postprocess.sh
```

Select initial queries on CIFAR-10-LT
```
bash tools/cifar_postprocess.sh
```
