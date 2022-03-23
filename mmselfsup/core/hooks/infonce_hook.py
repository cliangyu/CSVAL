# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pandas as pd
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class InfoNCEHook(Hook):
    """Hook for logging InfoNCE prob.

    This hook records InfoNCE prob between each sample's positive pair.
    The similarity score will be saved in ``runner.work_dir``.

    Args:
        interval (int, optional): the interval to save the queue.
            Defaults to 1.
    """

    def __init__(self, interval=1, **kwargs):
        self.interval = interval

    def before_train_epoch(self, runner):
        self.training_dynamics = dict(idx=[], prob=[])

    def after_train_iter(self, runner):
        for var_name, var_value in runner.model.module.training_dynamics.items(
        ):
            self.training_dynamics[var_name].extend(var_value)

    def after_train_epoch(self, runner):
        logger = runner.logger
        if self.training_dynamics is not None and self.every_n_epochs(
                runner, self.interval):
            ids = np.repeat(self.training_dynamics['idx'], 2)
            self.log_training_dynamics(
                output_dir=runner.work_dir,
                epoch=runner.epoch,
                ids=ids,
                prob=self.training_dynamics['prob'],
                logger=logger,
                split='training')

    def log_training_dynamics(self,
                              output_dir,
                              epoch,
                              ids,
                              prob,
                              logger,
                              split='training'):
        """Save training dynamics (InfoNCE prob) from given epoch as records of
        a `.jsonl` file."""
        td_df = pd.DataFrame({
            'idx': ids,
            f'prob_epoch_{epoch}': prob,
        })

        logging_dir = osp.join(output_dir, f'{split}_dynamics')
        # create directory for logging training dynamics
        mmcv.mkdir_or_exist(logging_dir)
        epoch_file_name = osp.join(logging_dir,
                                   f'dynamics_epoch_{epoch}.jsonl')
        td_df.to_json(epoch_file_name, lines=True, orient='records')
        logger.info(
            f'{split.capitalize()} dynamics logged to {epoch_file_name}')
