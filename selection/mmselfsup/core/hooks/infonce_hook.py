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
        self.temperature_list = [0.2, 0.1, 0.05, 0.01]

    def before_train_epoch(self, runner):
        self.training_dynamics = {}
        for temperature in self.temperature_list:
            self.training_dynamics[temperature] = dict(idx=[], prob=[])

    def after_train_iter(self, runner):
        for temperature in self.temperature_list:
            for var_name, var_value in runner.model.module.training_dynamics[
                    temperature].items():
                self.training_dynamics[temperature][var_name].extend(var_value)

    def after_train_epoch(self, runner):
        logger = runner.logger
        if self.training_dynamics is not None and self.every_n_epochs(
                runner, self.interval):
            for temperature in self.temperature_list:
                training_dynamics = self.training_dynamics[temperature]
                if len(training_dynamics['idx']) == len(
                        training_dynamics['prob']):  # for MoCo
                    ids = training_dynamics['idx']
                else:
                    ids = np.repeat(training_dynamics['idx'], 2)  # for SimCLR
                self.log_training_dynamics(
                    output_dir=runner.work_dir,
                    epoch=runner.epoch,
                    ids=ids,
                    prob=training_dynamics['prob'],
                    logger=logger,
                    split='training',
                    temperature=temperature)

    def log_training_dynamics(self,
                              output_dir,
                              epoch,
                              ids,
                              prob,
                              logger,
                              split='training',
                              temperature=0.01):
        """Save training dynamics (InfoNCE prob) from given epoch as records of
        a `.jsonl` file."""
        td_df = pd.DataFrame({
            'idx': ids,
            f'prob_epoch_{epoch}': prob,
        })

        logging_dir = osp.join(output_dir, f'{split}_dynamics',
                               str(temperature))
        # create directory for logging training dynamics
        mmcv.mkdir_or_exist(logging_dir)
        epoch_file_name = osp.join(logging_dir,
                                   f'dynamics_epoch_{epoch}.jsonl')
        td_df.to_json(epoch_file_name, lines=True, orient='records')
        logger.info(
            f'{split.capitalize()} dynamics logged to {epoch_file_name}')
