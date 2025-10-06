#!/usr/bin/env python3

#creates latent representations of all the rgb images in the dataset, this will be useful when using them to condition the diffusion model

#./create_latents.py --subsample_factor 1 --dataset_path=<DATASET_PATH> --out_path <DATASET_PATH_PROCESSED>


import sys
import os
import argparse
import torch
import torchvision
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from models.strand_codec import StrandCodec
from torch.utils.data import DataLoader
import utils.resize_right.resize_right as resize_right
import utils.resize_right.interp_methods as interp_methods
from tqdm import tqdm
import torchvision.transforms as T
from data_loader.dataloader import DiffLocksDataset



torch.set_grad_enabled(False)


def horizontally_flip(batch):

    rgb_img=batch["rgb_img"]
    rgb_img=torchvision.transforms.functional.hflip(rgb_img)
    batch["rgb_img"]=rgb_img

    return batch


def generate_latents_dinov2(args, batch, preprocessor, model, output_latents_path):
    #encode img
    rgb_img=batch["rgb_img"].to(device)
    rgb_img=rgb_img[:,0:3,:,:]


    rgb_input = preprocessor(rgb_img).to(device)
    ret = model.forward_features(rgb_input)
    patch_tok = ret["x_norm_patchtokens"].clone()
    cls_tok = ret["x_norm_clstoken"].clone()

    # print("outputs",outputs)

    #only makes sense to write the last layer becuase it';s dinov2 and the other ones are not coarser representations
    cls_token=cls_tok
    # print("cls", cls_token.shape)
    patch_embeddings = patch_tok
    #reshape to [Batch_size, h, w, embedding]
    batch_size, num_patches, hidden_size = patch_embeddings.shape
    h = w = int(num_patches ** 0.5)  # Assuming the number of patches is a perfect square (e.g., 14x14)
    patch_embeddings_reshaped = patch_embeddings.reshape(batch_size, h, w, hidden_size)
    patch_embeddings_reshaped=patch_embeddings_reshaped.permute(0,3,1,2).contiguous() #Make it bchw
    #write last layer
    out_path_final_latent=os.path.join(output_latents_path, "final_latent.pt")
    torch.save(patch_embeddings_reshaped, out_path_final_latent)
    #write cls token which is like an embedding for the whole image
    out_path_cls_token=os.path.join(output_latents_path, "cls_token.pt")
    #writing cls token of size
    # print("cls_token",cls_token.shape)
    torch.save(cls_token, out_path_cls_token)

    # feat_pca = img_2_pca(patch_embeddings_reshaped)
    # torchvision.utils.save_image(feat_pca.squeeze(0), os.path.join(output_latents_path, "final_latent.png"))

   

    #write a file to signify that we are done with this folder
    #start with x so that rsync copies it last if we copy to local 
    open( os.path.join(output_latents_path, "x_done.txt"), 'a').close()
      

def main():

    #argparse
    parser = argparse.ArgumentParser(description='Create latents')
    parser.add_argument('--dataset_path', required=True, help='Path to the hair_synth dataset')
    parser.add_argument('--out_path', required=True, type=str, help='Where to output the processed hair_synth dataset')
    parser.add_argument('--subsample_factor', default=1, type=int, help='Subsample factor for the RGB img')
    parser.add_argument('--skip_validity_check', dest='check_validity', action='store_false', help='Wether to check for the validity of each hairstyle we read from the dataset. Some older dataset versions might need this turned to false')
    args = parser.parse_args()

    
    #v2 from torch
    image_size = int(768/(2**(args.subsample_factor-1)))
    print("Selected dino with img size", image_size)
    #going to the nearest multiple of 14 because 14 is the patch size
    if image_size==768:
        image_size=770
    else:
        print("I haven't implemented the other ones yet")
    latents_preprocessor = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    latents_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    latents_model.to(device)

    latents_model.eval()




    print("args.check_validity",args.check_validity)

    difflocks_dataset = DiffLocksDataset(args.dataset_path, 
                                          check_validity=args.check_validity,
                                          load_rgb_imgs=True,
                                          processed_difflocks_path = args.out_path,
                                          subsample_factor=args.subsample_factor,
                                            )
    loader = DataLoader(difflocks_dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True, persistent_workers=True)

    

    progress_bar = tqdm(range(0, len(difflocks_dataset)), desc="Training progress")

    for batch in loader:
        progress_bar.update()

        #make the output path
        output_latents_path=os.path.join(args.out_path, "processed_hairstyles", batch["file"][0], "latents_"+"dinov2"+"_subsample_"+str(args.subsample_factor))
        os.makedirs(output_latents_path, exist_ok=True)
        #check if we already created this one
        if not os.path.isfile( os.path.join(output_latents_path,"x_done.txt")):
        # if True:
            #if it doesn't exist or we can't load it we create it
            generate_latents_dinov2(args, batch, latents_preprocessor, latents_model, output_latents_path)
            

        # #generate also a flipped texture, the reason being that just flipping the rgb does not result in a flipped latents neceserily so we have to horizontally flip the data in the batch then encode a new flipped latent
        #make the output path
        output_latents_path=os.path.join(args.out_path, "processed_hairstyles", batch["file"][0], "latents_flipped_"+"dinov2"+"_subsample_"+str(args.subsample_factor))
        os.makedirs(output_latents_path, exist_ok=True)
        #check if we already created this one
        if not os.path.isfile( os.path.join(output_latents_path,"x_done.txt")):
        # if True:
            batch=horizontally_flip(batch)
            #if it doesn't exist or we can't load it we create it
            generate_latents_dinov2(args, batch, latents_preprocessor, latents_model, output_latents_path)
            
          

    #finished training
    return


if __name__ == '__main__':
    main()
    
