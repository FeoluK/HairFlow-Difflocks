#!/usr/bin/env python3

# ./inference_difflocks.py \
#   --img_path=./samples/medium_11.png \
#   --out_path=./outputs_inference/ 

from inference.img2hair import DiffLocksInference
import subprocess
import os
import argparse


def run():

    #argparse
    parser = argparse.ArgumentParser(description='Get the weights of each dimensions after training a strand VAE')
    parser.add_argument('--strand_checkpoint_path', default="./checkpoints/strand_vae/strand_codec.pt", type=str, help='Path to the strandVAE checkpoint')
    parser.add_argument('--difflocks_checkpoint_path', default="./checkpoints/difflocks_diffusion/scalp_v9_40k_06730000.pth", type=str, help='Path to the difflocks checkpoint')
    parser.add_argument('--difflocks_config_path', default="./configs/config_scalp_texture_conditional.json", type=str, help='Path to the difflocks config')
    parser.add_argument('--rgb2mat_checkpoint_path', default="./checkpoints/rgb2material/rgb2material.pt", type=str,  help='Path to the rgb2material checkpoint')
    parser.add_argument('--blender_path', type=str, default="", help='Path to the blender executable')
    parser.add_argument('--blender_nr_threads', default=8, type=int, help='Number of threads for blender to use')
    parser.add_argument('--blender_strands_subsample', default=1.0, type=float, help='Amount of subsample of the strands(1.0=full strands, 0.5=half strands)')
    parser.add_argument('--blender_vertex_subsample', default=1.0, type=float, help='Amount of subsample of the vertices(1.0=all vertex, 0.5=half number of vertices per strand)')
    parser.add_argument('--alembic_resolution', default=7, type=int, help='Resolution of the exported alembic')
    parser.add_argument('--export_alembic', action='store_true', help='weather to export alembic or not')
    parser.add_argument('--do_shrinkwrap', action='store_true', help='applies a shrinkwrap modifier in blender that pushes the strands away from the scalp so they dont pass through the head')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the image to do inference on')
    parser.add_argument('--out_path', type=str, required=True, help='Path to the image to do inference on')
    args = parser.parse_args()

    print("args is", args)
    
    out_path="./outputs_inference/"

    difflocks= DiffLocksInference(args.strand_checkpoint_path, args.difflocks_config_path, args.difflocks_checkpoint_path, args.rgb2mat_checkpoint_path)


    #run----
    # img_path="./samples/medium_11.png"
    strand_points_world, hair_material_dict=difflocks.file2hair(args.img_path, args.out_path) 
    print("hair_material_dict",hair_material_dict)


    #create blender file and optionally an alembic file
    if args.blender_path!="":
        cmd=[args.blender_path, "-t", str(args.blender_nr_threads), "--background", "--python", "./inference/npz2blender.py", "--", "--input_npz", os.path.join(out_path,"difflocks_output_strands.npz"), "--out_path", args.out_path, "--strands_subsample", str(args.blender_strands_subsample), "--vertex_subsample", str(args.blender_vertex_subsample), "--alembic_resolution", str(args.alembic_resolution) ]
        if args.do_shrinkwrap:
            cmd.append("--shrinkwrap")
        if args.export_alembic:
            cmd.append("--export_alembic")
        subprocess.run(cmd, capture_output=False)

    print("Finished writing to ", args.out_path)

if __name__ == '__main__':

    run()
