data_source = 'PathMNIST'
dataset_type = 'SingleViewDataset'
name = 'pathmnist_val'
img_norm_cfg = dict(mean=[.5], std=[.5])

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    extract=dict(
        type='SingleViewDataset',
        data_source=dict(
            type=data_source,
            data_prefix='data/medmnist',
            split='val',
        ),
        pipeline=[
            # dict(type='Resize', size=256),
            # dict(type='CenterCrop', size=224),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]))
