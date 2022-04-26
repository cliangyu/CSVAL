_base_ = [
    '../_base_/moco_runtime.py',
    '../_base_/datasets/medmnist/organamnist_simclr.py',
]

data = dict(
    drop_last=True,
    samples_per_gpu=4096,
    workers_per_gpu=32,
)

model = dict(backbone=dict(in_channels=1))

runner = dict(type='EpochBasedRunner', max_epochs=100)
