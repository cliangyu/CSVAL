# dataset settings
data_source = 'PathMNIST'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[.5], std=[.5])
train_pipeline1 = [
    dict(type='RandomResizedCrop', size=28, scale=(0.2, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='Solarization', p=0.),
    dict(type='RandomHorizontalFlip'),
]
train_pipeline2 = [
    dict(type='RandomResizedCrop', size=28, scale=(0.2, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='Solarization', p=0.2),
    dict(type='RandomHorizontalFlip'),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline1.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])
    train_pipeline2.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=1024,  # total 4096=1024*4
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/medmnist',
            split='train',
        ),
        num_views=[1, 1],
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch,
    ))
model = dict(
    backbone=dict(
        type='VisionTransformer',
        arch='mocov3-small',  # embed_dim = 384
        img_size=28,
        patch_size=4,
        stop_grad_conv1=True), )
