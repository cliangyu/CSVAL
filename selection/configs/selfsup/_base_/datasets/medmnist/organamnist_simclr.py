# dataset settings
data_source = 'OrganAMNIST'
N = 37
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[.5], std=[.5])

# Big Self-Supervised Models
# Advance Medical Image Classification,
# CheXpert augmentation
train_pipeline = [
    dict(type='RandomResizedCrop', size=28),
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
    dict(type='RandomRotation', degrees=45),
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
    workers_per_gpu=16,
    train=dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(
            type=dataset_type,
            data_source=dict(
                type=data_source,
                data_prefix='data/medmnist',
                split='train',
            ),
            num_views=[2],
            pipelines=[train_pipeline],
            prefetch=prefetch,
        )))
