# dataset settings
data_source = 'PathMNIST'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[.5], std=[.5])
train_pipeline = [
    dict(type='RandomResizedCrop', size=28),
    dict(type='RandomHorizontalFlip'),
]
test_pipeline = [
    # dict(type='Resize', size=256),
    # dict(type='CenterCrop', size=224),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])
    test_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=128,  # 128*1
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/medmnist',
            split='train',
        ),
        pipeline=train_pipeline,
        prefetch=prefetch),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/medmnist',
            split='val',
        ),
        pipeline=test_pipeline,
        prefetch=prefetch),
    test=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/medmnist',
            split='test',
        ),
        pipeline=test_pipeline,
        prefetch=prefetch),
)
evaluation = dict(interval=10, topk=(1, 5))
