import os

import numpy as np
import torchvision

os.chdir(
    '/media/ntu/volume1/home/s121md302_06/workspace/code/mmselfsup/data/cifar10'  # noqa
)
save_dir = '/media/ntu/volume1/home/s121md302_06/workspace/code/mmselfsup/work_dirs/montage'  # noqa

data_flag = 'cifar10lt_rho100'

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True)
idx = np.load(
    '/media/ntu/volume1/home/s121md302_06/workspace/code/mmselfsup/work_dirs/selfsup/mocov2_resnet50_1xb512-coslr-800e_cifar10lt/cifar10lt_100_idx.npy'  # noqa
)


def montage(imgs, length=[7, 12], replace=False, save_folder=None):
    from medmnist.utils import montage2d

    n_sel = length[0] * length[1]
    sel = np.random.choice(len(imgs), size=n_sel, replace=replace)
    montage_img = montage2d(length=length, imgs=imgs, n_channels=3, sel=sel)

    return montage_img


montage = montage(imgs=trainset.data[idx], length=[21, 12])
save_file = os.path.join(save_dir, f'montage_{data_flag}.png')
montage.save(save_file, format='png')
