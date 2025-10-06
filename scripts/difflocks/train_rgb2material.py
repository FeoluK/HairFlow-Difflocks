#!/usr/bin/env python3

import os
import argparse
from models.rgb_to_material import RGB2MaterialModel
from torch.utils.data import DataLoader
from callbacks.callback_utils import *
import numpy as np
import random
from losses.losses import *
from schedulers.pytorch_warmup.untuned import UntunedLinearWarmup
from tqdm import tqdm
import copy
from data_loader.dataloader import DiffLocksDataset

#path in order to import hair_synth
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#lambda
# CUDA_VISIBLE_DEVICES=4 python3 ./train_rgb2material.py --dataset_path=<PATH_DIFFLOCKS> --dataset_processed_path=<PATH_DIFFLOCKS_PROCESSED> --exp_info=rgb2mat_name_experiment



torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
# torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


class HyperParamsRGB2Mat:
    def __init__(self):
        self.nr_iters_to_train=500000 
        self.lr= 1e-3
        self.save_checkpoint=True
        self.save_checkpoint_every_x_epoch=10
        self.with_tensorboard=True
        self.with_visualizer=False
        self.viewer_config_path=os.path.join(SCRIPT_DIR, "./configs/strand_vae_train.toml")

      


def create_dataloaders(dataset_path, dataset_processed_path):
    difflocks_dataset_train = DiffLocksDataset(dataset_path, 
                                                processed_difflocks_path=dataset_processed_path,
                                                train=True, load_rgb_imgs=False, load_full_strands=False,\
                                                load_guide_strands=False, load_interpolated_strands=False, 
                                                load_cam=False,
                                                subsample_factor=1, #needed in order to load the dino latents from subsample1
                                                load_material=True,
                                                compute_tbn_full_strands=False,
                                                load_latents=True,
                                                latents_type_list=["dinov2"],
                                                load_latents_layers=[
                                                    ["final_latent"]
                                                ],
                                                #not really needd but we want to filter to only those samples that have a scalp texture because on local we don't have all of them downloaded
                                                # load_scalp_texture=True,
                                                # scalp_texture_resolution=64,

                                                check_validity=True,
                                                do_pedantic_checks=False, 
                                                overfit=False,
                                                train_ratio=0.9)
    difflocks_dataset_test = DiffLocksDataset(dataset_path, 
                                                processed_difflocks_path=dataset_processed_path,
                                                train=False, load_rgb_imgs=False, load_full_strands=False,\
                                                load_guide_strands=False, load_interpolated_strands=False, 
                                                load_cam=False,
                                                subsample_factor=1, #needed in order to load the dino latents from subsample1
                                                load_material=True,
                                                compute_tbn_full_strands=False,
                                                load_latents=True,
                                                latents_type_list=["dinov2"],
                                                load_latents_layers=[
                                                    ["final_latent"]
                                                ],
                                                #not really needd but we want to filter to only those samples that have a scalp texture because on local we don't have all of them downloaded
                                                # load_scalp_texture=True,
                                                # scalp_texture_resolution=64,

                                                check_validity=True,
                                                do_pedantic_checks=False, 
                                                overfit=False,
                                                train_ratio=0.9)

    loader_train = DataLoader(difflocks_dataset_train, batch_size=8, num_workers=8, shuffle=True, pin_memory=True, persistent_workers=True,
                              prefetch_factor=3)
    loader_test = DataLoader(difflocks_dataset_test, batch_size=8, num_workers=8, shuffle=False, pin_memory=True, persistent_workers=True,
                             prefetch_factor=3)

    return loader_train, loader_test

#
def compute_loss(phase, gt_dict, pred_dict, hyperparams):
        
    gt_material=gt_dict["material"] 
    pred_material=pred_dict["material"] 
    nr_batches=gt_material.shape[0]
    #pred material is usually in the range 0,1 but the first two values are slightly different so we rescale those
    
    pred_material[:,0]*=30
    pred_material[:,1]*=360

   
    loss_per_elem = ((gt_material-pred_material)**2)
    


    
    gt_melanin=gt_material[:,3]
    root_darkness_strength=gt_material[:,-1]
   

    #root_darkenss should be downweighted in loss if the melanin is high, so if the hair is dark, it doesn't matter if we predict the correct root_darkness
    root_darkness_weight = 1.0-gt_melanin

    loss_per_elem[:,0]*=0.0 #material_wave_scale
    loss_per_elem[:,1]*=0.0 #material_wave_phase_offset
    loss_per_elem[:,2]*=0.0 #material_wave_strength
    loss_per_elem[:,3]*=1.0 #material_melanin_amount
    loss_per_elem[:,4]*=1.0 #bsdf_melanin_redness
    loss_per_elem[:,5]*=0.0 #bsdf_roughness
    loss_per_elem[:,6]*=0.0 #bsdf_radial_roughness
    loss_per_elem[:,7]*=0.0 #bsdf_coat
    loss_per_elem[:,8]*=root_darkness_strength*root_darkness_weight #root_darkness_start
    loss_per_elem[:,9]*=root_darkness_strength*root_darkness_weight #root_darkness_end
    loss_per_elem[:,10]*=1.0*root_darkness_weight #root_darkness_strength

   
    loss = loss_per_elem.mean()

    loss_dict={}
    loss_dict["loss"]=loss

    return loss_dict


def prepare_gt_batch(batch, hyperparams, do_augmentation=False):
    gt_dict = {}

    
    gt_dict["dinov2_latents"]=batch["latents"]["dinov2"]["final_latent"].to(device)
    gt_dict["material"]=batch["material"].to(device)

    return gt_dict

# 





def train(args, hyperparams, loader_train, loader_test, experiment_name, output_training_path):

    cb=create_callbacks(with_tensorboard=hyperparams.with_tensorboard,\
                        with_visualizer=hyperparams.with_visualizer,\
                        viewer_config_path=hyperparams.viewer_config_path,\
                        experiment_name=experiment_name)

    #create phases
    phases= [
        Phase('train', loader_train, grad=True),
        Phase('test', loader_test, grad=False),
    ]

    #model 
    model = RGB2MaterialModel(
                    input_dim=1024,
                    out_dim=11,
                    hidden_dim=64,
                       ).to(args.device)
    # model = torch.compile(model)

  

    # #optimizer
    optimizer = torch.optim.AdamW (model.parameters(), amsgrad=False, lr=hyperparams.lr, weight_decay=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams.nr_iters_to_train)
    scheduler_warmup = UntunedLinearWarmup(optimizer)

    progress_bar = tqdm(range(0, hyperparams.nr_iters_to_train), desc="Training progress")

    loss_zero=torch.zeros((1)).to(device)


    is_in_training_loop=True
    while is_in_training_loop:

        for phase in phases:
            model.train(phase.grad)
            cb.phase_started(phase=phase)
            cb.epoch_started(phase=phase)


            #run epoch 
            for batch in iter(phase.loader):
                cb.before_forward_pass(phase=phase)

                #progress
                if phase.grad and phase.iter_nr%100==0 :
                    progress_bar.update(100)


                #world_to_local
                with torch.no_grad():
                    gt_dict = prepare_gt_batch(batch, hyperparams, do_augmentation=phase.grad) 
                    
                    input_dict=copy.deepcopy(gt_dict)
                    latents=input_dict["dinov2_latents"]
                    
                    #make a mask with ones so as to mask whole patches of the 1x1024x55x55 dino latents
                    mask=torch.ones_like(latents[:,0:1,:,:]) #N1hw
                    mask=torch.nn.functional.dropout(mask,0.1)
                    latents=latents*mask
                    #also dropout random elements of the latents so that pixels (or patchs) in the 55x55 have sometimes different values accross channels
                    latents=torch.nn.functional.dropout(latents,0.1)
                    input_dict["dinov2_latents"]=latents


                    
                
                pred_dict = model(input_dict)
              
                loss_dict = compute_loss(phase, gt_dict, pred_dict, hyperparams)
                loss=loss_dict["loss"]
                # print("loss",loss)
                

                if torch.isnan(loss):
                    print("found nan")
                    exit()


                #backward
                if phase.grad:
                    optimizer.zero_grad()
                    cb.before_backward_pass()
                    loss.backward()
                    cb.after_backward_pass()
                    optimizer.step()

                    with scheduler_warmup.dampening():
                        lr_scheduler.step()


              

                cb.after_forward_pass(phase=phase, loss=loss, 
                                      loss_pos=loss_zero, loss_dir=loss_zero, loss_curv=loss_zero,
                                    lr=optimizer.param_groups[0]['lr'])

            cb.epoch_ended(phase=phase)
            cb.phase_ended(phase=phase, model=model, hyperparams=hyperparams, experiment_name=experiment_name, output_training_path=output_training_path)

            if phase.grad and phase.iter_nr>=hyperparams.nr_iters_to_train:
                print("Done training!")
                is_in_training_loop=False
                model.save(output_training_path, experiment_name, hyperparams, phase.epoch_nr, info="final")
                exit(1)

        

def main():

    #argparse
    parser = argparse.ArgumentParser(description='Train sdf and color')
    parser.add_argument('--dataset_path', required=True, help='Path to the hair_synth dataset to train on')
    parser.add_argument('--dataset_processed_path', required=True, help='Path to the hair_synth processed dataset to train on')
    parser.add_argument('--exp_info', default="", help='Experiment info string useful for distinguishing one experiment for another')
    parser.add_argument('--device', default=device)
    args = parser.parse_args()

    #get the output path which will be at the root of the package 
    hair_forge_root=os.path.dirname(os.path.abspath(__file__))
    output_training_path=os.path.join(hair_forge_root, "out_training")
    os.makedirs(output_training_path, exist_ok=True)


    experiment_name="hair_forge"
    if args.exp_info:
        experiment_name+="_"+args.exp_info


    loader_train, loader_test= create_dataloaders(args.dataset_path, args.dataset_processed_path)

  
    hyperparams=HyperParamsRGB2Mat() 
    train(args, hyperparams, loader_train, loader_test, experiment_name, output_training_path)

    #finished training
    return


if __name__ == '__main__':
    main()
    
