# dataset settings
data_source = 'CIFAR10'
# N = 25
N = 1
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])

train_pipeline = [
    dict(type='RandomResizedCrop', size=32),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    # dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type='RandomHorizontalFlip'),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=2048,  # total 4096=2048*2
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(
            type=dataset_type,
            data_source=dict(
                type=data_source,
                data_prefix='data/cifar10',
            ),
            num_views=[2],
            pipelines=[train_pipeline],
            prefetch=prefetch,
        )))
