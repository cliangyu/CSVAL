import os

import medmnist
import torchvision.transforms as transforms
from medmnist import INFO

os.chdir(
    '/media/ntu/volume1/home/s121md302_06/workspace/code/mmselfsup/data/medmnist'  # noqa
)
save_dir = '/media/ntu/volume1/home/s121md302_06/workspace/code/mmselfsup/work_dirs/montage'  # noqa

for data_flag in ['pathmnist', 'dermamnist', 'bloodmnist', 'organamnist']:
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    download = True

    # load the data
    train_dataset = DataClass(
        split='train', transform=data_transform, download=download)
    test_dataset = DataClass(
        split='test', transform=data_transform, download=download)

    pil_dataset = DataClass(split='train', download=download)

    montage = train_dataset.montage(length=[21, 12])

    save_file = os.path.join(save_dir, f'montage_{data_flag}.png')
    montage.save(save_file, format='png')
