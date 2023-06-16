import os
import pickle, gzip
import torch
import pytorch_lightning as pl
from time import time

import smdebug.pytorch as smd


save_config = smd.SaveConfig()
smd.Collection
hook = smd.Hook(
    out_dir="/home/ubuntu/resnet-sagemaker/pytorch/debugger_logs/",
    export_tensorboard = True,
    save_config = save_config
)


class PlSageMakerLogger(pl.Callback):
    
    def __init__(self, frequency=10):
        self.frequency=frequency
        self.step = 0
        self.epoch = 0

    # def on_train_start(self, trainer, pl_module):
    #     hook.register_module(pl_module)


    
    def on_train_epoch_start(self, trainer, module, *args, **kwargs):
        self.inner_step = 1
        self.epoch += 1
        self.step_time_start = time()
    
    @pl.utilities.rank_zero_only
    def on_train_batch_end(self, trainer, module, *args, **kwargs):
        if self.inner_step%self.frequency==0:
            print("Step : {} of epoch {}".format(self.inner_step, self.epoch))
            print("Training Losses:")
            print(' '.join(["{0}: {1:.4f}".format(key, float(value)) \
                            for key,value in trainer.logged_metrics.items()]))
            step_time_end = time()
            print("Step time: {0:.2f} milliseconds".format((step_time_end - self.step_time_start)/self.frequency * 1000))
            self.step_time_start = step_time_end
        self.inner_step += 1
        self.step += 1
        
    @pl.utilities.rank_zero_only
    def on_validation_end(self, trainer, module, *args, **kwargs):
        print("Validation")
        print(' '.join(["{0}: {1:.4f}".format(key, float(value)) \
                        for key,value in trainer.logged_metrics.items() if 'val' in key]))
