_base_ = [
    '../_base_/moco_runtime.py',
    '../_base_/datasets/cifar/cifar10_simclr.py',
]

data = dict(
    drop_last=True,
    samples_per_gpu=512,
    workers_per_gpu=32,
)

optimizer = dict(
    type='SGD',
    #  lr=0.03, # base lr
    lr=0.06,  # lr at bs of 4096
    weight_decay=5e-4,
    momentum=0.9)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# model settings
model = dict(
    queue_len=4096, head=dict(type='ContrastiveHead', temperature=0.1))
