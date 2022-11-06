_base_ = [
    'simclr_resnet50_4xb1024-coslr-200e_path.py',
]

model = dict(head=dict(type='ContrastiveHead', temperature=0.2))
