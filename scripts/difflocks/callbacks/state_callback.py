from callbacks.callback import *
import os
import torch


class StateCallback(Callback):

    def __init__(self):
        pass

    def after_forward_pass(self, phase, loss, loss_pos, loss_dir, loss_curv, **kwargs):
        phase.iter_nr+=1
        phase.samples_processed_this_epoch+=1
        phase.loss_acum_per_epoch+=loss
        phase.loss_pos_acum_per_epoch+=loss_pos
        phase.loss_dir_acum_per_epoch+=loss_dir
        phase.loss_curv_acum_per_epoch+=loss_curv


    def epoch_started(self, phase, **kwargs):
        phase.loss_acum_per_epoch=0.0
        phase.loss_pos_acum_per_epoch=0.0
        phase.loss_dir_acum_per_epoch=0.0
        phase.loss_curv_acum_per_epoch=0.0

    def epoch_ended(self, phase, **kwargs):

        phase.epoch_nr+=1

    def phase_started(self, phase, **kwargs):
        phase.samples_processed_this_epoch=0

    def phase_ended(self, phase, model, hyperparams, experiment_name, output_training_path, **kwargs):

        if (phase.epoch_nr%hyperparams.save_checkpoint_every_x_epoch==0) and hyperparams.save_checkpoint and phase.grad:
            model.save(output_training_path, experiment_name, hyperparams, phase.epoch_nr)


