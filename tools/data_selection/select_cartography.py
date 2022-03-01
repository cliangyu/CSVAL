# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import random

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pandas as pd
from mmcv import Config

from mmselfsup.apis import set_random_seed
from mmselfsup.datasets import build_dataset
from mmselfsup.utils import get_root_logger


def check_best_patial(plist, num_train):
    print('\n', len([int(num_train * i / 100000.0) for i in plist]))
    print([int(num_train * i / 100000.0) for i in plist])


def plot_class_dist(labels,
                    percentage,
                    save_dir,
                    logger,
                    title=None,
                    figsize=[40, 40],
                    ylim=(None, None)):
    classes, counts = np.unique(labels, return_counts=True)
    logger.info(f'class counts:{counts}')
    df_rel = pd.DataFrame(columns=['classes', 'counts'])
    df_rel['classes'], df_rel['counts'] = classes, counts
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.autolayout'] = True
    df_rel.plot(
        x='classes',
        y='counts',
        kind='bar',
        stacked=True,
        title=title,
        legend=None,
        figsize=figsize,
        colormap='Reds_r',
        xlabel=None,
        xticks=None,
        width=0.9)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.ylim(ylim)
    plt.savefig(
        osp.join(save_dir, f'p-0.{percentage}_distribution_histogram.png'))
    plt.clf()
    plt.cla()
    plt.close('all')


def parse_args():
    parser = argparse.ArgumentParser(
        description='select samples with uniformity and cartography metrics')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work_dir', type=str, default=None, help='the dir to save results')
    parser.add_argument(
        '--dataset_config',
        default='configs/benchmarks/classification/tsne_pathmnist.py',
        help='extract dataset config file path')
    parser.add_argument('--pseudo_labels', help='pseudo labels file path')
    parser.add_argument(
        '--training_dynamics', help='training dynamics file path')
    parser.add_argument(
        '--metric',
        choices=['easy', 'ambiguous', 'hard'],
        default='ambiguous',
        help='cartography metric to select data')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # get pseudo labels
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0],
                                'data_selection')
    mmcv.mkdir_or_exist(cfg.work_dir)
    log_file = osp.join(cfg.work_dir, 'select.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    dataset_cfg = mmcv.Config.fromfile(args.dataset_config)
    dataset = build_dataset(dataset_cfg.data.extract)
    gt_labels = dataset.data_source.get_gt_labels()

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, ')
        set_random_seed(args.seed)

    if args.pseudo_labels is None:
        args.pseudo_labels = osp.join(
            './work_dirs', work_type,
            osp.splitext(osp.basename(args.config))[0], 'tsne',
            'clustering_pseudo_labels', f'{dataset_cfg.name}.npy')
    pseudo_labels = np.load(args.pseudo_labels)
    num_pseudo_class = len(np.unique(pseudo_labels))
    # rank each pseudo class with cartography metrics
    if args.training_dynamics is None:
        args.training_dynamics = osp.join(
            './work_dirs', work_type,
            osp.splitext(osp.basename(args.config))[0], 'cartography',
            'training_td_metrics.jsonl')
    training_dynamics = pd.read_json(
        path_or_buf=args.training_dynamics, lines=True)
    selection_df = training_dynamics.sort_values('idx')
    selection_df['pseudo_label'] = pseudo_labels
    selection_df['gt_label'] = gt_labels

    # plist = [i for i in range(10, 100, 10)] + \
    plist = [i for i in range(15, 100, 5)] + \
        [i for i in range(100, 1000, 50)] + \
        [i for i in range(1000, 10000, 500)] + \
        [i for i in range(10000, 100000, 5000)]
    check_best_patial(plist=plist, num_train=len(dataset))
    zfill = 5
    p = [str(i).zfill(zfill) for i in plist]

    if args.metric == 'easy':
        metric = 'confidence'
        ascending = False
    elif args.metric == 'ambiguous':
        metric = 'variability'
        ascending = False
    elif args.metric == 'hard':
        metric = 'confidence'
        ascending = True
    else:
        raise ValueError('metric not supported')

    selection_df = selection_df.sort_values(metric, ascending=ascending)
    plot_dir = osp.join(cfg.work_dir, 'distribution_histogram')
    mmcv.mkdir_or_exist(plot_dir)

    plt.rcParams.update({'figure.max_open_warning': 0})

    for ind, num_select in enumerate(plist):
        # sample uniformly from pseudo classes.
        bucket = np.array_split(range(num_select), num_pseudo_class)
        random.shuffle(bucket)
        indices = []
        for label, indice_placeholder in enumerate(bucket):
            indices.extend(selection_df[selection_df['pseudo_label'] == label]
                           ['idx'][:len(indice_placeholder)])
        indices = np.array(indices)

        # save results
        dataset_name = dataset_cfg.name.split('_')[0]
        percentage = p[ind]
        save_dir = osp.join(cfg.work_dir,
                            f'{dataset_name}-p0.{percentage}.npy')
        np.save(save_dir, indices)
        logger.info(f'{len(indices)} samples saved to {save_dir}')

        # plot gt_label distribution
        plot_class_dist(
            labels=selection_df[selection_df.idx.isin(indices)]['gt_label'],
            percentage=percentage,
            logger=logger,
            save_dir=plot_dir)


if __name__ == '__main__':
    main()
