# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from tqdm import tqdm

from mmselfsup.datasets import build_dataset
from mmselfsup.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='save')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument(
        '--work_dir', type=str, default=None, help='the dir to save results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--dataset_config',
        default='configs/benchmarks/classification/umap_imagenet.py',
        help='extract dataset config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])

    # create work_dir and init the logger before other steps
    work_dir = osp.join(cfg.work_dir, '')
    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    log_file = osp.join(work_dir, 'extract.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # build the dataloader
    dataset_cfg = mmcv.Config.fromfile(args.dataset_config)
    dataset = build_dataset(dataset_cfg.data.extract)
    gt_labels = dataset.data_source.get_gt_labels()

    # save images
    for idx, gt_label in tqdm(enumerate(gt_labels)):
        img = dataset.data_source.get_img(idx)
        save_name = osp.join(work_dir, f'{idx}_{gt_label}.jpg')
        img.save(save_name)

    logger.info(f'Saved iamges to {work_dir}')


if __name__ == '__main__':
    main()
