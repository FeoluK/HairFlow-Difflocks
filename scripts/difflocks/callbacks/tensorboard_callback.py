from callbacks.callback import *
from torch.utils.tensorboard import SummaryWriter

class TensorboardCallback(Callback):

    def __init__(self, experiment_name):
        self.tensorboard_writer = SummaryWriter("tensorboard_logs/"+experiment_name)
        self.experiment_name=experiment_name
        

    def after_forward_pass(self, phase, loss=0, loss_pos=0, loss_dir=0, loss_curv=0, loss_kl=0, lr=0, z_deviation=None, z=None, z_no_eps=None, **kwargs):

        if phase.iter_nr%300==0 and phase.grad:
            self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/loss', loss.item(), phase.iter_nr)
            if loss_pos!=0:
                self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/loss_pos', loss_pos.item(), phase.iter_nr)
            # if loss_l1!=0:
                # self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/loss_l1', loss_l1.item(), phase.iter_nr)
            # if loss_l2!=0:
                # self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/loss_l2', loss_l2.item(), phase.iter_nr)
            if loss_dir!=0:
                self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/loss_dir', loss_dir.item(), phase.iter_nr)
            if loss_curv!=0:
                self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/loss_curv', loss_curv.item(), phase.iter_nr)
            if loss_kl!=0:
                self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/loss_kl', loss_kl.item(), phase.iter_nr)
            
            if lr!=0:
                self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/lr', lr, phase.iter_nr)

            if z_deviation is not None:
                self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/z_deviation', z_deviation.std(), phase.iter_nr)
            
            # if z is not None:
            #     self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/z_max', z.max(), phase.iter_nr)
            if z_no_eps is not None:
                self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/z_no_eps_mean', z_no_eps.mean(), phase.iter_nr)
                self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/z_no_eps_std', z_no_eps.std(), phase.iter_nr)


    def epoch_ended(self, phase, **kwargs):
        avg_loss_pos=phase.loss_pos_acum_per_epoch/phase.samples_processed_this_epoch
        avg_loss_dir=phase.loss_dir_acum_per_epoch/phase.samples_processed_this_epoch
        avg_loss_curv=phase.loss_curv_acum_per_epoch/phase.samples_processed_this_epoch

        if phase.grad==False and phase.loss_pos_acum_per_epoch!=0:
                self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/loss_pos_avg', avg_loss_pos.item(), phase.epoch_nr)
        if phase.grad==False and phase.loss_dir_acum_per_epoch!=0:
                self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/loss_dir_avg', avg_loss_dir.item(), phase.epoch_nr)
        if phase.grad==False and phase.loss_curv_acum_per_epoch!=0:
                self.tensorboard_writer.add_scalar('hair_forge/' + phase.name + '/loss_curv_avg', avg_loss_curv.item(), phase.epoch_nr)
        