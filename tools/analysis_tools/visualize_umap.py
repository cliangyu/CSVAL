# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import pickle

import matplotlib
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import seaborn as sns
import torch
import umap
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from sklearn.cluster import KMeans

from mmselfsup.apis import set_random_seed
from mmselfsup.datasets import build_dataloader, build_dataset
from mmselfsup.models import build_algorithm
from mmselfsup.models.utils import ExtractProcess
from mmselfsup.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='UMAP visualization')
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
        '--layer_ind',
        type=str,
        default='0,1,2,3,4',
        help='layer indices, separated by comma, e.g., "0,1,2,3,4"')
    parser.add_argument(
        '--pool_type',
        choices=['specified', 'adaptive'],
        default='specified',
        help='Pooling type in :class:`MultiPooling`')
    parser.add_argument(
        '--max_num_class',
        type=int,
        default=20,
        help='the maximum number of classes to apply UMAP algorithms, now the'
        'function supports maximum 20 classes')
    parser.add_argument(
        '--max_num_sample_plot',
        type=int,
        default=20000,
        help='the maximum number of samples to plot.')
    parser.add_argument(
        '--num_selected_sample',
        type=int,
        default=30,
        help='the number of visualized selected samples.')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--sorted_idx_file', type=str, help='sorted idx .npy file.')
    parser.add_argument(
        '--no_point_selected',
        action='store_true',
        help='plot all the points, no point selected.')
    parser.add_argument(
        '--overwrite_features',
        action='store_true',
        help='whether to overwrite features.')
    parser.add_argument(
        '--overwrite_pseudo_label',
        action='store_true',
        help='whether to overwrite pseudo labels.')
    parser.add_argument(
        '--plot_name', type=str, help='file name of the saved plots.')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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

    # clustering settings
    parser.add_argument(
        '-k',
        type=int,
        default=30,
        help='k for k-means clustering, the categories of pseudo labels.')
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
        work_type = args.config.split('/')[1]
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])

    # get out_indices from args
    layer_ind = [int(idx) for idx in args.layer_ind.split(',')]
    cfg.model.backbone.out_indices = layer_ind

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir and init the logger before other steps
    umap_work_dir = osp.join(cfg.work_dir, '')
    mmcv.mkdir_or_exist(osp.abspath(umap_work_dir))
    log_file = osp.join(umap_work_dir, 'extract.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset_cfg = mmcv.Config.fromfile(args.dataset_config)
    dataset = build_dataset(dataset_cfg.data.extract)
    gt_labels = dataset.data_source.get_gt_labels()

    # compress dataset, select that the label is less then max_num_class
    tmp_infos = []
    for i in range(len(dataset)):
        if dataset.data_source.data_infos[i]['gt_label'] < args.max_num_class:
            tmp_infos.append(dataset.data_source.data_infos[i])
    dataset.data_source.data_infos = tmp_infos

    # extract features
    if osp.isfile(f'{umap_work_dir}features/features.pkl'
                  ) and not args.overwrite_features:
        with open(f'{umap_work_dir}features/features.pkl', 'rb') as f:
            features = pickle.load(f)
    else:
        logger.info(f'Apply UMAP to visualize {len(dataset)} samples.')

        if 'imgs_per_gpu' in cfg.data:
            cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=dataset_cfg.data.samples_per_gpu,
            workers_per_gpu=dataset_cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model
        model = build_algorithm(cfg.model)
        model.init_weights()

        # model is determined in this priority: init_cfg > checkpoint > random
        if hasattr(cfg.model.backbone, 'init_cfg'):
            if getattr(cfg.model.backbone.init_cfg, 'type',
                       None) == 'Pretrained':
                logger.info(f'Use pretrained model: '
                            f'{cfg.model.backbone.init_cfg.checkpoint}'
                            f'to extract features')
        elif args.checkpoint is not None:
            logger.info(
                f'Use checkpoint: {args.checkpoint} to extract features')
            load_checkpoint(model, args.checkpoint, map_location='cpu')
        else:
            logger.info(
                'No pretrained or checkpoint is given, use random init.')

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)

        # build extraction processor and run
        extractor = ExtractProcess(
            pool_type=args.pool_type,
            backbone='resnet50',
            layer_indices=layer_ind)
        features = extractor.extract(
            model, data_loader, distributed=distributed)

        # save features
        mmcv.mkdir_or_exist(f'{umap_work_dir}features/')
        logger.info(f'Save features to {umap_work_dir}features/')
        if distributed:
            rank, _ = get_dist_info()
            if rank == 0:
                for key, val in features.items():
                    output_file = \
                        f'{umap_work_dir}features/{dataset_cfg.name}_{key}.npy'
                    np.save(output_file, val)
        else:
            for key, val in features.items():
                output_file = \
                    f'{umap_work_dir}features/{dataset_cfg.name}_{key}.npy'
                np.save(output_file, val)
        with open(f'{umap_work_dir}features/features.pkl', 'wb') as f:
            pickle.dump(features, f)

    # clustering
    if osp.isfile(
            f'{umap_work_dir}clustering_pseudo_labels/{dataset_cfg.name}.npy'
    ) and not args.overwrite_pseudo_label:
        clustering_pseudo_labels = np.load(
            f'{umap_work_dir}clustering_pseudo_labels/{dataset_cfg.name}.npy')
    else:
        logger.info('Running clustering......')
        # Features are normalized during clustering
        kmeans = KMeans(args.k, random_state=0)
        reducer = umap.UMAP()
        result = reducer.fit_transform(features['feat5'])
        clustering_pseudo_labels = kmeans.fit(result).predict(result)
        # save clustering_pseudo_labels
        mmcv.mkdir_or_exist(f'{umap_work_dir}clustering_pseudo_labels/')
        output_file = \
            f'{umap_work_dir}clustering_pseudo_labels/{dataset_cfg.name}.npy'
        np.save(output_file, clustering_pseudo_labels)

    # build UMAP model
    reducer = umap.UMAP()

    # run and get results
    mmcv.mkdir_or_exist(f'{umap_work_dir}saved_pictures/')
    logger.info('Running UMAP......')
    if len(dataset) > args.max_num_sample_plot:
        indices = np.random.permutation(
            len(dataset))[:args.max_num_sample_plot]
    else:
        indices = np.arange(len(dataset))
    dataset_name = dataset_cfg.name.split('_')[0]

    # calculate sorted index
    if args.sorted_idx_file is None:
        args.sorted_idx_file = osp.join(umap_work_dir, 'data_selection',
                                        f'{dataset_name}_easy_sorted_idx.npy')
    sorted_idx = np.load(args.sorted_idx_file)
    selected_idx = sorted_idx[np.isin(sorted_idx,
                                      indices)][:args.num_selected_sample]

    for key, val in features.items():
        output_file = osp.join(f'{umap_work_dir}', 'features',
                               f'{dataset_cfg.name}_{key}_umap.npy')
        if osp.isfile(output_file) and not args.overwrite_features:
            result = np.load(output_file)
        else:
            result = reducer.fit_transform(val)
            np.save(output_file, result)
        res_min, res_max = result.min(0), result.max(0)
        res_norm = (result - res_min) / (res_max - res_min)

        plt.figure(figsize=(10, 10))
        my_pal = [
            '#e60049', '#0bb4ff', '#50e991', '#e6d800', '#9b19f5', '#ffa300',
            '#dc0ab4', '#b3d4ff', '#00bfa0', '#fdcce5', '#1a53ff'
        ]
        pal = ListedColormap(
            sns.color_palette(my_pal,
                              n_colors=len(np.unique(gt_labels))).as_hex())
        if args.no_point_selected:
            plt.scatter(
                res_norm[indices, 0],
                res_norm[indices, 1],
                alpha=0.1,
                s=15,
                c=gt_labels[indices],
                cmap=pal)
        else:
            # plot round scatter plot for unselected samples
            plt.scatter(
                res_norm[np.setdiff1d(indices, selected_idx), 0],
                res_norm[np.setdiff1d(indices, selected_idx), 1],
                alpha=0.1,
                s=15,
                c=gt_labels[np.setdiff1d(indices, selected_idx)],
                cmap=pal)
            # plot cross scatter plot for selected samples
            plt.scatter(
                res_norm[selected_idx, 0],
                res_norm[selected_idx, 1],
                alpha=1.0,
                s=500,
                marker='X',
                linewidth=3,
                edgecolors='black',
                c=gt_labels[selected_idx],
                cmap=pal)

        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.tight_layout()
        if args.plot_name is None:
            gt_label_plot_name = osp.join(f'{umap_work_dir}', 'saved_pictures',
                                          f'{key}_gt_labels.png')
        else:
            gt_label_plot_name = osp.join(
                f'{umap_work_dir}', 'saved_pictures',
                f'{key}_{args.plot_name}_gt_labels.png')
        plt.savefig(gt_label_plot_name)

        # plot pseudo labels
        light_cmap = cm.get_cmap('nipy_spectral', args.k)
        # modify colormap
        alpha = .5
        colors = []
        for ind in range(light_cmap.N):
            c = []
            for x in light_cmap(ind)[:3]:
                c.append(x * alpha)
            colors.append(tuple(c))
        dark_cmap = matplotlib.colors.ListedColormap(colors, name='dark')
        plt.figure(figsize=(10, 10))
        if args.no_point_selected:
            plt.scatter(
                res_norm[indices, 0],
                res_norm[indices, 1],
                alpha=0.1,
                s=15,
                c=clustering_pseudo_labels[indices],
                cmap=light_cmap)
        else:
            plt.scatter(
                res_norm[np.setdiff1d(indices, selected_idx), 0],
                res_norm[np.setdiff1d(indices, selected_idx), 1],
                alpha=1.0,
                s=15,
                c=clustering_pseudo_labels[np.setdiff1d(indices,
                                                        selected_idx)],
                cmap=light_cmap)
            plt.scatter(
                res_norm[selected_idx, 0],
                res_norm[selected_idx, 1],
                alpha=1.0,
                s=500,
                marker='X',
                linewidth=3,
                c=clustering_pseudo_labels[selected_idx],
                cmap=dark_cmap)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.tight_layout()
        plt.savefig(
            osp.join(f'{umap_work_dir}',
                     f'saved_pictures/{key}_kmeans_psuedo_labels.png'))
        if args.plot_name is None:
            psuedo_label_plot_name = osp.join(f'{umap_work_dir}',
                                              'saved_pictures',
                                              f'{key}_pseudo_labels.png')
        else:
            psuedo_label_plot_name = osp.join(
                f'{umap_work_dir}', 'saved_pictures',
                f'{key}_{args.plot_name}_pseudo_labels.png')
        plt.savefig(psuedo_label_plot_name)
    logger.info(f'Saved results to {umap_work_dir}saved_pictures/')


if __name__ == '__main__':
    main()
