# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pandas as pd
import seaborn as sns
from mmcv import Config

from mmselfsup.apis import set_random_seed
from mmselfsup.datasets import build_dataset
from mmselfsup.utils import get_root_logger


def plot_class_dist(labels,
                    percentage,
                    save_dir,
                    logger,
                    title=None,
                    figsize=[40, 40],
                    ylim=(None, None)):
    labels[labels == 0] = 1e-4
    sns.set(style='whitegrid', font_scale=3, context='paper')
    figsize = [25.4, 57.2]
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.autolayout'] = True
    my_pal = [
        '#e60049', '#0bb4ff', '#50e991', '#e6d800', '#9b19f5', '#ffa300',
        '#dc0ab4', '#b3d4ff', '#00bfa0'
    ]
    num_unique_labels = len(labels)
    pal = sns.color_palette(my_pal, n_colors=num_unique_labels)

    ax = sns.barplot(  # noqa F841
        y=list(map(str, np.arange(num_unique_labels))),
        x=labels,
        palette=pal,
        ci=None,
    )
    # ax.bar_label(ax.containers[0]) # annotate the bars
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
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


def convert_array_to_cnt_array(ori_labels, num_classes):
    unique, counts = np.unique(ori_labels, return_counts=True)
    count_dict = dict(zip(unique, counts))
    labels = np.zeros(num_classes)
    for _class in count_dict.keys():
        labels[_class] = count_dict[_class]
    return labels


def main():
    args = parse_args()
    # get pseudo labels
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        work_type = args.config.split('/')[1]
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
    unique, counts = np.unique(gt_labels, return_counts=True)
    num_real_classes = len(unique)
    labels = convert_array_to_cnt_array(gt_labels, num_real_classes)
    plot_dir = osp.join(cfg.work_dir, 'distribution_histogram_full')
    mmcv.mkdir_or_exist(plot_dir)
    # plot gt_label distribution
    plot_class_dist(
        labels=labels, percentage='1', logger=logger, save_dir=plot_dir)

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, ')
        set_random_seed(args.seed)

    if args.pseudo_labels is None:
        args.pseudo_labels = osp.join(
            './work_dirs', work_type,
            osp.splitext(osp.basename(args.config))[0],
            'clustering_pseudo_labels', f'{dataset_cfg.name}.npy')
    pseudo_labels = np.load(args.pseudo_labels)
    num_pseudo_class = len(np.unique(pseudo_labels))
    # rank each pseudo class with cartography metrics
    if args.training_dynamics is None:
        args.training_dynamics = osp.join(
            './work_dirs', work_type,
            osp.splitext(osp.basename(args.config))[0], 'cartography',
            'training_td_metrics.jsonl')
    logger.info(f'Load training dynamics from {args.training_dynamics}')
    training_dynamics = pd.read_json(
        path_or_buf=args.training_dynamics, lines=True)
    selection_df = training_dynamics.sort_values('idx')
    selection_df['pseudo_label'] = pseudo_labels
    selection_df['gt_label'] = gt_labels

    # define percentage list
    dataset_name = dataset_cfg.name.split('_')[0]
    if dataset_name in ['pathmnist']:
        plist = [i for i in range(15, 100, 5)] + \
            [i for i in range(100, 1000, 50)] + \
            [i for i in range(1000, 10000, 500)] + \
            [i for i in range(10000, 100000, 5000)]
    elif dataset_name in [
            'organamnist', 'pneumoniamnist', 'bloodmnist', 'dermamnist'
    ]:
        plist = [i for i in range(100, 1000, 100)] + \
            [i for i in range(1000, 10000, 1000)] + \
            [i for i in range(10000, 100000, 10000)]
    elif dataset_name in ['tissuemnist', 'octmnist']:
        plist = [i for i in range(10, 100, 10)] + \
            [i for i in range(100, 1000, 100)] + \
            [i for i in range(1000, 10000, 1000)] + \
            [i for i in range(10000, 100000, 10000)]
    elif dataset_name in ['breastmnist', 'retinamnist']:
        plist = [i for i in range(500, 10000, 500)] + \
            [i for i in range(10000, 100000, 5000)]

    num_train = len(dataset)
    zfill = 5
    p = [str(i).zfill(zfill) for i in plist]
    num_select_list = [int(num_train * i / 100000.0) for i in plist]
    logger.info(f'{len(num_select_list)} ratios will be selected.')
    logger.info(num_select_list)

    # rank by metrics
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
    plot_dir = osp.join(cfg.work_dir, f'distribution_histogram_{args.metric}')
    mmcv.mkdir_or_exist(plot_dir)
    sample_by_percentage_dir = osp.join(cfg.work_dir, f'sample_{args.metric}')
    mmcv.mkdir_or_exist(sample_by_percentage_dir)

    plt.rcParams.update({'figure.max_open_warning': 0})

    # generate the whole sorting list
    pseudo_label_order = np.arange(num_pseudo_class)
    np.random.shuffle(pseudo_label_order)
    label_dict = {}
    largest_cluster_size = 0
    for pseudo_label in pseudo_label_order:
        label_dict[pseudo_label] = np.array(
            selection_df[selection_df['pseudo_label'] == pseudo_label]['idx'])
        largest_cluster_size = len(label_dict[pseudo_label]) if len(
            label_dict[pseudo_label]
        ) > largest_cluster_size else largest_cluster_size
    sample_list = np.array([], dtype=np.int8)
    for i in range(largest_cluster_size):
        for pseudo_label in pseudo_label_order:
            try:
                sample_list = np.append(sample_list,
                                        label_dict[pseudo_label][i])
            except (KeyError, IndexError):
                pass
    assert len(sample_list) == len(dataset)
    save_dir = osp.join(cfg.work_dir,
                        f'{dataset_name}_{args.metric}_sorted_idx.npy')
    logger.info(f'All samples saved to '
                f'{dataset_name}_{args.metric}_sorted_idx.npy')
    np.save(save_dir, sample_list)

    for ind, num_select in enumerate(num_select_list):
        indices = sample_list[:num_select]

        # save results
        percentage = p[ind]
        save_dir = osp.join(sample_by_percentage_dir,
                            f'{dataset_name}-p0.{percentage}.npy')
        np.save(save_dir, indices)
        logger.info(f'{len(indices)} samples saved to {save_dir}')

        labels = convert_array_to_cnt_array(
            selection_df[selection_df.idx.isin(indices)]['gt_label'],
            num_real_classes)

        # plot gt_label distribution
        plot_class_dist(
            labels=labels,
            percentage=percentage,
            logger=logger,
            save_dir=plot_dir)


if __name__ == '__main__':
    main()
