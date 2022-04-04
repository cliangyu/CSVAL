_base_ = [
    '../_base_/moco_runtime.py',
    '../_base_/datasets/medmnist/dermamnist_simclr.py',
]

data = dict(
    drop_last=True,
    samples_per_gpu=4096,
    workers_per_gpu=32,
)
