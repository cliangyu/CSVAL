# dataset settings
data_source = 'PneumoniaMNIST'
N = 270
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[.5], std=[.5])
color_jitter_strength = 0.5

train_pipeline = [
    dict(type='RandomResizedCrop', size=28),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomRotation', degrees=45),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8 * color_jitter_strength,
                contrast=0.8 * color_jitter_strength,
                saturation=0.8 * color_jitter_strength,
                hue=0.2 * color_jitter_strength)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
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
                data_prefix='data/medmnist',
                split='train',
            ),
            num_views=[2],
            pipelines=[train_pipeline],
            prefetch=prefetch,
        )))
