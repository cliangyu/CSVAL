data_source = 'CIFAR10'
dataset_type = 'SingleViewDataset'
name = data_source.lower()
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    extract=dict(
        type='SingleViewDataset',
        data_source=dict(
            type=data_source,
            data_prefix='data/cifar10',
        ),
        pipeline=[
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]))
