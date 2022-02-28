data_source = 'PathMNIST'
dataset_type = 'SingleViewDataset'
name = 'pathmnist_train'
img_norm_cfg = dict(mean=[.5], std=[.5])

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    extract=dict(
        type='SingleViewDataset',
        data_source=dict(
            type=data_source,
            data_prefix='data/medmnist',
            split='train',
        ),
        pipeline=[
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]))
