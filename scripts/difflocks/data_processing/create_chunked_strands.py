#!/usr/bin/env python3

#from the full_strands.npz it creates chunked versions of it they can be more easily loaded by the trainer for strand vae

# ./create_chunked_strands.py --dataset_path <DATASET_PATH>


import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import math
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from data_loader.dataloader import DiffLocksDataset


def main():

    #argparse
    parser = argparse.ArgumentParser(description='Create latents')
    parser.add_argument('--dataset_path', required=True, help='Path to the hair_synth dataset')
    args = parser.parse_args()


    difflocks_dataset = DiffLocksDataset(args.dataset_path, 
                                          check_validity=False,
                                          load_full_strands=True
                                            )
    loader = DataLoader(difflocks_dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True, persistent_workers=True)

    

    progress_bar = tqdm(range(0, len(difflocks_dataset)), desc="Training progress")

    for batch in loader:
        progress_bar.update()

        # print("batch",batch)

        path_hairstyle=batch["path"][0]
        print("path", path_hairstyle)

        positions=batch["full_strands"]["positions"].squeeze(0).cpu().numpy()
        root_uv=batch["full_strands"]["root_uv"].squeeze(0).cpu().numpy()
        root_normal=batch["full_strands"]["root_normal"].squeeze(0).cpu().numpy()
        # print("positions", positions.shape)
        # print("root_uv", root_uv.shape)
        # print("root_normal", root_normal.shape)


        select_nr_random_strands=10000
        nr_strands_per_chunk_list=[1000,100]

        #select random strands because the original 100K is too much 
        if select_nr_random_strands:
            nr_strands_left = positions.shape[0]
            per_curve_keep_random = np.random.choice(nr_strands_left, select_nr_random_strands, replace=False)
            positions=positions[per_curve_keep_random,:,:]
            root_uv=root_uv[per_curve_keep_random,:]
            root_normal=root_normal[per_curve_keep_random,:]


        #break the strands into chunks if needed and write those too
        nr_strands_total = positions.shape[0]
        if nr_strands_per_chunk_list:
            for nr_strands_cur_chunk in nr_strands_per_chunk_list:
                nr_chunks = math.ceil(nr_strands_total/nr_strands_cur_chunk)

                positions_chunked = np.array_split(positions, nr_chunks,axis=0)
                root_uv_chunked = np.array_split(root_uv, nr_chunks,axis=0)
                root_normal_chunked = np.array_split(root_normal, nr_chunks,axis=0)

                #make path for this chunked data
                chunked_data_path=os.path.join(path_hairstyle,"full_strands_chunked","nr_strands_"+str(nr_strands_cur_chunk))
                os.makedirs(chunked_data_path, exist_ok=True)
                
                #write each chunk
                for idx_chunk in range(nr_chunks):
                    npz_path = os.path.join(chunked_data_path,str(idx_chunk)+".npz")
                    np.savez(npz_path, positions=positions_chunked[idx_chunk],\
                            root_uv=root_uv_chunked[idx_chunk],\
                            root_normal=root_normal_chunked[idx_chunk])


    return


if __name__ == '__main__':
    main()
    
