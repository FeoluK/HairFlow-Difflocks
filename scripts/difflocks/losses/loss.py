import torch
from losses.losses import  compute_loss_dir_l1, compute_loss_curv_l1, compute_loss_l1, compute_loss_l2, compute_loss_kl
import numpy as np

class StrandVAELoss(torch.nn.Module):
    def __init__(
        self,
        eps: float = 1e-8,
        ):
        super().__init__()

        self.eps = eps
       
    def forward(self, phase, gt_dict, pred_dict, latent_dict, hyperparams):
        gt_strands= gt_dict["strand_positions"]
        pred_hair_strands= pred_dict["strand_positions"] 

        # w_kl=map_range_val(phase.epoch_nr, hyperparams.loss_kl_anneal_epoch_nr_start, hyperparams.loss_kl_anneal_epoch_nr_finish, 0.0, hyperparams.loss_kl_weight)


        loss_dict = {}
        
        #we always predict the xyz loss just because it's fast
        # loss_l2 = compute_loss_l2(gt_strands, pred_hair_strands)
        loss_pos = compute_loss_l1(gt_strands, pred_hair_strands)
        loss_dir = compute_loss_dir_l1(gt_strands, pred_hair_strands)
        loss_curv = compute_loss_curv_l1(gt_strands, pred_hair_strands)
        loss_kl = 0.0
        if "z_logstd" in latent_dict:
            loss_kl = compute_loss_kl(latent_dict["z_mean"], latent_dict["z_logstd"])
        loss = loss_pos*hyperparams.loss_pos_weight + loss_dir*hyperparams.loss_dir_weight + loss_curv*hyperparams.loss_curv_weight + loss_kl*hyperparams.loss_kl_weight
        # loss = loss_pos*hyperparams.loss_pos_weight + loss_dir*hyperparams.loss_dir_weight + loss_curv*hyperparams.loss_curv_weight + loss_kl*w_kl
        loss_dict['loss'] = loss
        loss_dict['loss_pos'] = loss_pos
        loss_dict['loss_dir'] = loss_dir
        loss_dict['loss_curv'] = loss_curv
        loss_dict['loss_kl'] = loss_kl


        return loss_dict



