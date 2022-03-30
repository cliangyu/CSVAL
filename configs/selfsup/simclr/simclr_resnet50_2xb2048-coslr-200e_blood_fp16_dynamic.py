_base_ = [
    '../_base_/datasets/medmnist/bloodmnist_simclr.py',
    '../_base_/simclr_runtime.py',
]

# fp16
fp16 = dict(loss_scale='dynamic')
