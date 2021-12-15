# DeepCluster

## Deep Clustering for Unsupervised Learning of Visual Features

<!-- [ABSTRACT] -->

Clustering is a class of unsupervised learning methods that has been extensively applied and studied in computer vision. Little work has been done to adapt it to the end-to-end training of visual features on large scale datasets. In this work, we present DeepCluster, a clustering method that jointly learns the parameters of a neural network and the cluster assignments of the resulting features. DeepCluster iteratively groups the features with a standard clustering algorithm, k-means, and uses the subsequent assignments as supervision to update the weights of the network.

<!-- [IMAGE] -->
<div align="center">
<img  />
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{caron2018deep,
  title={Deep clustering for unsupervised learning of visual features},
  author={Caron, Mathilde and Bojanowski, Piotr and Joulin, Armand and Douze, Matthijs},
  booktitle={ECCV},
  year={2018}
}
```

## Models and Benchmarks

**Back to [model_zoo.md](../../../docs/model_zoo.md) to download models.**

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models were trained on ImageNet1k dataset.

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are  Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                   | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ---------------------------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [sobel_resnet50_8xb64-steplr-200e](deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py) | feature5   | 74.26 | 29.37 | 37.99 | 45.85 | 55.57 | 62.48 | 66.15 | 70.00 | 71.37 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-90e.py](../../benchmarks/classification/imagenet/resnet50_mhead_8xb32-steplr-90e_in1k.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [file name]() for details of config.

| Self-Supervised Config                                                                   | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| ---------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [sobel_resnet50_8xb64-steplr-200e](deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py) | 12.78    | 30.81    | 43.88    | 57.71    | 51.68    | 46.92   |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k_voc0712.py](../../benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k_voc0712.py) for details of config.

| Self-Supervised Config                                                                   | AP50 |
| ---------------------------------------------------------------------------------------- | ---- |
| [sobel_resnet50_8xb64-steplr-200e](deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py) |      |  |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x_coco.py](../../benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py) for details of config.

| Self-Supervised Config                                                                   | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ---------------------------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [sobel_resnet50_8xb64-steplr-200e](deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py) |          |           |           |           |            |            |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [fcn_r50-d8_512x512_20k_voc12aug.py](../../benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py) for details of config.

| Self-Supervised Config                                                                   | mIOU  |
| ---------------------------------------------------------------------------------------- | ----- |
| [sobel_resnet50_8xb64-steplr-200e](deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py) | 59.69 |